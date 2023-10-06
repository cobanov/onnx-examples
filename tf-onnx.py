import tensorflow as tf
import tf2onnx

tf_model = tf.keras.models.load_model('tf_model.h5')
onnx_model = tf2onnx.convert.from_keras(tf_model)

onnx_model_path = 'converted_model.onnx'
with open(onnx_model_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Model converted to ONNX and saved as:", onnx_model_path)