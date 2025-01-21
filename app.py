import streamlit as st
import numpy as np
import pickle
import os

# Menampilkan nama dan NIM
st.title("Prediksi Pembayaran Premi Asuransi")
st.write("Nama: Pandu Satya Ramadhani")
st.write("NIM: 2021230005")

# Input data
age = st.number_input('Masukkan Umur (Age)', min_value=18, max_value=100, value=27)
sex = st.radio('Jenis Kelamin (Sex)', options=[0, 1], index=0, help="0: Perempuan, 1: Laki-laki")
bmi = st.number_input('Masukkan BMI (Body Mass Index)', min_value=10.0, max_value=60.0, value=30.0)
children = st.number_input('Jumlah Anak (Children)', min_value=0, max_value=5, value=1)
smoker = st.radio('Perokok (Smoker)', options=[0, 1], index=0, help="0: Tidak Merokok, 1: Merokok")

# Membuat tombol submit
submit = st.button('Submit')

if submit:
    # Menyiapkan data untuk prediksi
    X = np.array([age, sex, bmi, children, smoker])
    X = X.reshape(1, -1)

    # Menentukan path absolut untuk file model
    model_path = os.path.join(os.path.dirname(__file__), 'model_uas.pkl')

    try:
        # Memastikan file model ada sebelum dimuat
        if not os.path.exists(model_path):
            raise FileNotFoundError("File model_uas.pkl tidak ditemukan! Pastikan file model ada di direktori yang sama dengan app.py.")

        # Memuat model yang telah disimpan dengan pickle
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Melakukan prediksi
        charge_pred = loaded_model.predict(X)

        # Menampilkan hasil prediksi
        st.success(f"Prediksi Pembayaran Premi Asuransi: ${charge_pred[0]:,.2f}")

    except FileNotFoundError as fnf_error:
        st.error(fnf_error)
    except ModuleNotFoundError as mnf_error:
        st.error("Module scikit-learn tidak ditemukan! Pastikan telah menginstal dependensi dengan `pip install -r requirements.txt`.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
