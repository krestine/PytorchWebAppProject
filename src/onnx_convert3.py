import onnx

onnx_model = onnx.load("./test_onnx.onnx")
print(onnx.checker.check_model(onnx_model))