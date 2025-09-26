# prediksi_iris.py
import numpy as np
import joblib
from sklearn.datasets import load_iris

model = joblib.load("model_iris.joblib")
nama_kelas = load_iris().target_names

# Ubah nilai di bawah sesuai input yang ingin kamu tes
data_baru = np.array([[5.8, 2.7, 5.1, 1.9]])
pred = model.predict(data_baru)[0]
print(f"Prediksi: {nama_kelas[pred]}")
