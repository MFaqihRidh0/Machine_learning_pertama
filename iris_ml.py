# iris_ml.py
# ML paling sederhana: klasifikasi bunga Iris dengan Logistic Regression

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib

# 1) Muat data
iris = load_iris()
X = iris.data  # fitur: sepal length/width, petal length/width
y = iris.target  # label: 0=setosa, 1=versicolor, 2=virginica
nama_kelas = iris.target_names

# (opsional) lihat sekilas data sebagai DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

# 2) Bagi data menjadi train/test
X_latih, X_uji, y_latih, y_uji = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Pilih & latih model sederhana
model = LogisticRegression(max_iter=200)
model.fit(X_latih, y_latih)

# 4) Evaluasi
pred_uji = model.predict(X_uji)
akurasi = accuracy_score(y_uji, pred_uji)
print(f"Akurasi: {akurasi:.3f}")
print("\nClassification report:")
print(classification_report(y_uji, pred_uji, target_names=nama_kelas))
print("Confusion matrix:")
print(confusion_matrix(y_uji, pred_uji))

# 5) Simpan model untuk dipakai lagi
joblib.dump(model, "model_iris.joblib")
print("\nModel disimpan ke: model_iris.joblib")

# 6) Contoh prediksi data baru (format: [sepal length, sepal width, petal length, petal width])
contoh_data_baru = np.array([[5.1, 3.5, 1.4, 0.2],
                             [6.0, 2.9, 4.5, 1.5],
                             [6.3, 3.3, 6.0, 2.5]])
pred_baru = model.predict(contoh_data_baru)
for i, p in enumerate(pred_baru):
    print(f"Input {i+1}: diprediksi sebagai '{nama_kelas[p]}'")

# Catatan:
# - Tidak perlu tuning rumit. Tujuan: end-to-end jalan.
# - Nanti kamu bisa ganti model (SVM/RandomForest), fitur, atau dataset.
