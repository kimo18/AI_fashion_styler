import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("densepose_full.onnx")

# Check input name and shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input name: {input_name}")
print(f"Expected input shape: {input_shape}")

# Create dummy input with shape [3, 800, 800] (not batched)
dummy_input = np.random.randn(3, 800, 800).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: dummy_input})

# Print output shapes
output_names = [output.name for output in session.get_outputs()]
for name, output in zip(output_names, outputs):
    print(f"{name}: {output.shape}")