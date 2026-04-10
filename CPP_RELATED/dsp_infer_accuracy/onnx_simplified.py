import onnx
from onnxsim import simplify
m = onnx.load("../../Training/resnet18_cifar10.onnx")
m2, ok = simplify(m, input_shapes={"input":[1,3,32,32]})
assert ok, "onnxsim simplify failed"
onnx.save(m2, "model_simplified.onnx")
print("Wrote model_simplified.onnx")