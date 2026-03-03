import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1,3, 32, 32).to(device)
model = torch.load("resnet18_cifar10_full.pth", map_location=device, weights_only=False)
model.to(device)

torch.onnx.export(
    model,
    dummy_input,
    "resnet18_cifar10.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)