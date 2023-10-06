import onnx
from onnx_tf.backend import prepare
import numpy as np
from PIL import Image
import json

# Load the ONNX model

# Use local model
model_path = './vgg19-7.onnx' 
model = onnx.load(model_path)

## Download from hub
# model_path = 'vgg 19'
# model = onnx.hub.load(model_path)

# Prepare the ONNX model for TensorFlow
tf_model = prepare(model)

# Display input and output nodes of the model
print("Input nodes:", tf_model.inputs)
print("Output nodes:", tf_model.outputs)

# Load class labels from a JSON file
with open('labels.json', 'r') as json_file:
    label_data = json.load(json_file)

# Load and preprocess the input image
input_image_path = 'images/frog.jpg'
input_image = Image.open(input_image_path).resize((224, 224))
input_image = np.asarray(input_image, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
input_image = input_image.transpose(0, 1, 4, 2, 3)
input_image = input_image.reshape(1, 3, 224, 224)  # Transform to Input Tensor

# Run inference on the input image
output = tf_model.run(input_image)[0][0]

# Find the top 5 class indices
top_indices = np.argpartition(output, -5)[-5:]

# Display the top 5 class labels and their scores
for i in top_indices:
    class_label = label_data.get(str(i), "Unknown")
    class_score = output[i]
    print(f"Class: {class_label}, Score: {class_score:.4f}")