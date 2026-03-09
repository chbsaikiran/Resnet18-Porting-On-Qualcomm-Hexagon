cmake_minimum_required(VERSION 3.10)
project(calc_accuracy)

set(ORT_DIR "$ENV{HOME}/onnxruntime")
set(OCV_DIR "$ENV{HOME}/opencv/build_android/install/sdk/native/jni")

include_directories(${ORT_DIR}/include)
link_directories(${ORT_DIR}/build/Android/Release)

include_directories(${OCV_DIR}/include)
link_directories($ENV{HOME}/opencv/build_android/lib/${ANDROID_ABI})

add_executable(calc_accuracy arm_calculate_accuracy.cpp)
target_link_libraries(calc_accuracy
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_videoio
    onnxruntime
    log
)
