import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import onnxruntime as rt
from skl2onnx import to_onnx

# Load Iris dataset and split into training and testing sets
iris = load_iris()

X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train a Random Forest Classifier

clr = RandomForestClassifier()
clr.fit(X_train, y_train)


onx = to_onnx(clr, X[:1])
with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Inference using ONNX Runtime
sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]

# Now you can work with pred_onx for further analysis or evaluation.
