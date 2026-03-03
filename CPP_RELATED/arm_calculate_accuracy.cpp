#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstring>

namespace fs = std::filesystem;

static const float MEAN[3] = {0.4914f, 0.4822f, 0.4465f};
static const float STD[3]  = {0.2470f, 0.2435f, 0.2616f};

static const int IMG_SIZE    = 32;
static const int NUM_CLASSES = 10;

static const std::vector<std::string> CLASS_NAMES = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};

static const std::map<std::string, int> CLASSNAME_TO_INDEX = {
    {"airplane",   0}, {"automobile", 1}, {"bird",  2}, {"cat",   3},
    {"deer",       4}, {"dog",        5}, {"frog",  6}, {"horse", 7},
    {"ship",       8}, {"truck",      9},
};

struct ValSample {
    std::vector<float> pixels;   /* CHW float32, already normalised */
    int                label;
};

/* ------------------------------------------------------------------ */
/*  Preprocess a single BGR cv::Mat (for directory-based loading)      */
/* ------------------------------------------------------------------ */
static void preprocess_to_chw(const cv::Mat &frame, std::vector<float> &out)
{
    cv::Mat img;
    cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(IMG_SIZE, IMG_SIZE));

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    for (int i = 0; i < 3; i++)
        channels[i] = (channels[i] - MEAN[i]) / STD[i];

    int spatial = IMG_SIZE * IMG_SIZE;
    out.resize(3 * spatial);
    for (int c = 0; c < 3; c++)
        std::memcpy(out.data() + c * spatial,
                    channels[c].ptr<float>(),
                    spatial * sizeof(float));
}

/* ------------------------------------------------------------------ */
/*  Load CIFAR-10 binary test batch                                    */
/*  Format: 10000 records, each = 1 byte label + 3072 bytes image     */
/*          (1024 R + 1024 G + 1024 B, row-major)                     */
/* ------------------------------------------------------------------ */
static std::vector<ValSample> load_cifar10_bin(const std::string &bin_path)
{
    std::vector<ValSample> samples;
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Cannot open " << bin_path << "\n";
        return samples;
    }

    const int record_size = 1 + 3 * IMG_SIZE * IMG_SIZE;   /* 3073 */
    std::vector<uint8_t> record(record_size);
    int spatial = IMG_SIZE * IMG_SIZE;

    while (ifs.read(reinterpret_cast<char *>(record.data()), record_size)) {
        ValSample s;
        s.label = static_cast<int>(record[0]);
        s.pixels.resize(3 * spatial);

        const uint8_t *raw = record.data() + 1;
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < spatial; i++) {
                float v = static_cast<float>(raw[c * spatial + i]) / 255.0f;
                s.pixels[c * spatial + i] = (v - MEAN[c]) / STD[c];
            }
        }
        samples.push_back(std::move(s));
    }

    return samples;
}

/* ------------------------------------------------------------------ */
/*  Load from directory:  val_dir/class_name/*.png                     */
/* ------------------------------------------------------------------ */
static std::vector<ValSample> load_from_directory(const std::string &val_dir)
{
    std::vector<ValSample> samples;

    for (const auto &class_dir : fs::directory_iterator(val_dir)) {
        if (!class_dir.is_directory())
            continue;

        std::string name = class_dir.path().filename().string();
        auto it = CLASSNAME_TO_INDEX.find(name);
        if (it == CLASSNAME_TO_INDEX.end()) {
            std::cerr << "Warning: skipping unknown class folder " << name << "\n";
            continue;
        }
        int label = it->second;

        for (const auto &entry : fs::directory_iterator(class_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp")
                continue;

            cv::Mat frame = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (frame.empty()) continue;

            ValSample s;
            s.label = label;
            preprocess_to_chw(frame, s.pixels);
            samples.push_back(std::move(s));
        }
    }

    std::sort(samples.begin(), samples.end(),
              [](const ValSample &a, const ValSample &b) {
                  return a.label < b.label;
              });

    return samples;
}

int main(int argc, char **argv)
{
    std::string model_path = "resnet18_cifar10.onnx";
    std::string data_path  = "test_batch.bin";

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) data_path  = argv[2];

    /* ---- Load ONNX model -------------------------------------------- */

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cifar10_accuracy");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(4);

    Ort::Session session(env, model_path.c_str(), session_options);

    auto input_names  = session.GetInputNames();
    auto output_names = session.GetOutputNames();
    const char *input_name  = input_names[0].c_str();
    const char *output_name = output_names[0].c_str();

    std::cout << "Model loaded: " << model_path << "\n";

    /* ---- Load test data --------------------------------------------- */

    std::vector<ValSample> samples;

    if (fs::is_directory(data_path))
        samples = load_from_directory(data_path);
    else
        samples = load_cifar10_bin(data_path);

    int total_images = static_cast<int>(samples.size());
    std::cout << "Test images loaded: " << total_images << "\n\n";

    if (total_images == 0) {
        std::cerr << "No images found at " << data_path << "\n";
        return 1;
    }

    /* ---- Inference loop --------------------------------------------- */

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, 3, IMG_SIZE, IMG_SIZE};

    int correct = 0;
    int total   = 0;
    int per_class_correct[NUM_CLASSES] = {};
    int per_class_total[NUM_CLASSES]   = {};

    double total_inference_ms = 0.0;

    for (int idx = 0; idx < total_images; idx++) {
        ValSample &sample = samples[idx];

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            sample.pixels.data(),
            sample.pixels.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto t0 = std::chrono::high_resolution_clock::now();

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &input_name,  &input_tensor,  1,
            &output_name, 1
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        total_inference_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        float *output = output_tensors[0].GetTensorMutableData<float>();
        int pred = static_cast<int>(
            std::max_element(output, output + NUM_CLASSES) - output);

        int gt = sample.label;
        per_class_total[gt]++;
        total++;

        if (pred == gt) {
            correct++;
            per_class_correct[gt]++;
        }

        if ((idx + 1) % 1000 == 0 || idx == total_images - 1) {
            double running_acc = 100.0 * correct / total;
            std::printf("[%5d / %5d]  running accuracy: %.2f%%\n",
                        idx + 1, total_images, running_acc);
        }
    }

    /* ---- Results ---------------------------------------------------- */

    double accuracy = 100.0 * correct / total;

    std::cout << "\n====== CIFAR-10 Validation Results ======\n";
    std::cout << "Total images : " << total   << "\n";
    std::cout << "Correct      : " << correct << "\n";
    std::printf("Top-1 Accuracy: %.2f%%\n", accuracy);
    std::printf("Avg inference : %.3f ms\n", total_inference_ms / total);
    std::cout << "=========================================\n";

    std::cout << "\nPer-class accuracy:\n";
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (per_class_total[c] > 0) {
            double cls_acc = 100.0 * per_class_correct[c] / per_class_total[c];
            std::printf("  [%d] %-12s  %4d / %4d  (%.2f%%)\n",
                        c, CLASS_NAMES[c].c_str(),
                        per_class_correct[c], per_class_total[c], cls_acc);
        }
    }

    return 0;
}
