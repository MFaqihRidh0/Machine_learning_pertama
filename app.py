# app_iris_streamlit.py
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.datasets import load_iris
from pathlib import Path

st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered",
)

st.title("🌸 Iris Classifier")
st.caption("Demo Streamlit untuk memprediksi spesies bunga Iris dari 4 fitur.")

@st.cache_resource(show_spinner=False)
def load_model(model_path: str = "model_iris.joblib"):
    p = Path(model_path)
    if not p.exists():
        return None, f"File model tidak ditemukan: {p.resolve()}"
    try:
        model = joblib.load(p)
        return model, None
    except Exception as e:
        return None, f"Gagal memuat model: {e}"

@st.cache_data(show_spinner=False)
def iris_meta():
    ds = load_iris()
    feature_names = ds.feature_names  # ["sepal length (cm)", ...]
    target_names = ds.target_names    # ["setosa", "versicolor", "virginica"]
    # Range rekomendasi slider dari data asli
    X = ds.data
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    means = X.mean(axis=0)
    return feature_names, target_names, mins, maxs, means

feature_names, target_names, mins, maxs, means = iris_meta()
model, model_err = load_model("model_iris.joblib")

with st.sidebar:
    st.header("Pengaturan")
    mode = st.radio("Mode Prediksi", ["Single Input", "Batch (CSV)"], horizontal=False)

    st.divider()
    st.subheader("Preset Contoh")
    preset = st.selectbox(
        "Isi cepat contoh fitur:",
        [
            "Tidak ada (kosongkan)",
            "Setosa (rata-rata)",
            "Versicolor (rata-rata)",
            "Virginica (rata-rata)",
        ],
        index=0,
    )
    st.caption("Preset akan mengisi nilai input secara otomatis.")

    st.divider()
    with st.expander("Tentang Model"):
        if model_err:
            st.error(model_err)
            st.info(
                "Pastikan file `model_iris.joblib` ada di folder yang sama.\n"
                "Model sebaiknya dilatih pada dataset Iris (urutan fitur: "
                "`sepal length, sepal width, petal length, petal width`)."
            )
        else:
            st.success("Model berhasil dimuat.")
            st.code(repr(model), language="python")

def class_mean_samples():
    ds = load_iris()
    X, y = ds.data, ds.target
    means_by_class = []
    for c in range(len(ds.target_names)):
        means_by_class.append(np.mean(X[y == c], axis=0))
    return means_by_class  # [setosa_mean, versicolor_mean, virginica_mean]

setosa_m, versicolor_m, virginica_m = class_mean_samples()

def preset_values(name: str):
    if name.startswith("Setosa"):
        return setosa_m
    if name.startswith("Versicolor"):
        return versicolor_m
    if name.startswith("Virginica"):
        return virginica_m
    return None

if mode == "Single Input":
    st.subheader("Input Fitur")

    # Tentukan nilai awal: mean dataset
    defaults = means.copy()

    pv = preset_values(preset)
    if pv is not None:
        defaults = pv

    # Buat 4 input numerik (urutan penting!)
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input(
            label=feature_names[0].title().replace("(Cm)", "(cm)"),
            min_value=float(mins[0]) - 0.5,
            max_value=float(maxs[0]) + 0.5,
            value=float(defaults[0]),
            step=0.1,
            help="Panjang sepal (cm)",
        )
        petal_length = st.number_input(
            label=feature_names[2].title().replace("(Cm)", "(cm)"),
            min_value=float(mins[2]) - 0.5,
            max_value=float(maxs[2]) + 0.5,
            value=float(defaults[2]),
            step=0.1,
            help="Panjang petal (cm)",
        )
    with col2:
        sepal_width = st.number_input(
            label=feature_names[1].title().replace("(Cm)", "(cm)"),
            min_value=float(mins[1]) - 0.5,
            max_value=float(maxs[1]) + 0.5,
            value=float(defaults[1]),
            step=0.1,
            help="Lebar sepal (cm)",
        )
        petal_width = st.number_input(
            label=feature_names[3].title().replace("(Cm)", "(cm)"),
            min_value=float(mins[3]) - 0.5,
            max_value=float(maxs[3]) + 0.5,
            value=float(defaults[3]),
            step=0.1,
            help="Lebar petal (cm)",
        )

    do_predict = st.button("Prediksi")

    if do_predict:
        if model is None:
            st.error("Model belum tersedia, tidak bisa melakukan prediksi.")
        else:
            x = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            pred = model.predict(x)[0]
            label = target_names[pred] if pred < len(target_names) else str(pred)
            st.success(f"Hasil Prediksi: **{label}**")

            # Tampilkan probabilitas jika ada
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)[0]
                prob_df = pd.DataFrame(
                    {"kelas": target_names, "probabilitas": proba}
                ).sort_values("probabilitas", ascending=False)
                st.caption("Probabilitas per kelas:")
                st.bar_chart(prob_df.set_index("kelas"))
                st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
            else:
                st.info("Model tidak menyediakan probabilitas (`predict_proba`).")

else:
    st.subheader("📥 Prediksi Batch via CSV")
    st.caption(
        "Format kolom (wajib, urutan bebas): "
        "`sepal_length, sepal_width, petal_length, petal_width` (satuan cm)."
    )

    st.download_button(
        "📎 Unduh Contoh CSV",
        data=(
            "sepal_length,sepal_width,petal_length,petal_width\n"
            "5.1,3.5,1.4,0.2\n"
            "6.7,3.0,5.2,2.3\n"
            "5.9,3.0,4.2,1.5\n"
        ).encode("utf-8"),
        file_name="contoh_iris.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()

        # Normalisasi nama kolom
        rename_map = {
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "sepal_length": "sepal_length",
            "sepal_width": "sepal_width",
            "petal_length": "petal_length",
            "petal_width": "petal_width",
        }
        df_cols_lower = {c: c.strip().lower() for c in df.columns}
        df.rename(columns=df_cols_lower, inplace=True)
        df.rename(columns=rename_map, inplace=True)

        required_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Kolom berikut belum ada di CSV: {missing}")
            st.stop()

        st.write("Preview data:")
        st.dataframe(df.head(), use_container_width=True)

        if model is None:
            st.error("Model belum tersedia, tidak bisa melakukan prediksi batch.")
        else:
            X = df[required_cols].to_numpy(dtype=float)
            preds = model.predict(X)
            labels = [target_names[p] if p < len(target_names) else str(p) for p in preds]
            out = df.copy()
            out["prediksi"] = labels

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)
                for i, cname in enumerate(target_names):
                    out[f"proba_{cname}"] = probas[:, i]

            st.success("Prediksi selesai.")
            st.dataframe(out.head(20), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Unduh Hasil (CSV)",
                data=csv_bytes,
                file_name="hasil_prediksi_iris.csv",
                mime="text/csv",
            )

st.caption(
    "Catatan: Pastikan urutan/skalering fitur saat melatih model sesuai dengan input di aplikasi ini. "
    "Jika model adalah pipeline (mis. StandardScaler + classifier), langsung gunakan file pipeline."
)
