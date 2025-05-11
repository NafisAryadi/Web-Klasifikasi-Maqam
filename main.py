import streamlit as st
import numpy as np
import librosa as lib
import soundfile as sf
import tensorflow as tf
from pydub import AudioSegment

import h5py
import os
import sys
import time
import io
import gdown

sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

from preprocessing import predict_audio_with_mfcc, predict_audio_with_chroma, predict_audio_with_both

def check_model_ready(filepath, min_size_mb=1.0, max_wait=180):
    waited = 0
    while True:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb > min_size_mb:
                try:
                    with h5py.File(filepath, 'r'):
                        print(f"✅ Model {filepath} sudah valid dan siap dipakai.")
                        break
                except Exception as e:
                    print(f"⏳ Menunggu file valid: {e}")
        time.sleep(5)
        waited += 5
        if waited > max_wait:
            raise TimeoutError(f"❌ Gagal validasi file model: {filepath}")

url_model_mfcc = "https://drive.google.com/file/d/1-Nkzb4PA9PcHXdGnmiGCoL5PWcAudnms/view?usp=drive_link"
output_model_mfcc = "model/mfcc_model.h5"  # Path tempat menyimpan file model yang diunduh
gdown.download(url_model_mfcc, output_model_mfcc, quiet=False)
check_model_ready(output_model_mfcc)

url_model_chroma = "https://drive.google.com/file/d/1SdJH64DPPqHs4NcFENwckoFZNC7ipL-B/view?usp=drive_link"
output_model_chroma = "model/chroma_model.h5"  # Path tempat menyimpan file model yang diunduh
gdown.download(url_model_chroma, output_model_chroma, quiet=False)
check_model_ready(output_model_chroma)

url_model_both = "https://drive.google.com/file/d/1QemH-BV736XEiow6AeHJQJm9XlsuLCvk/view?usp=drive_link"
output_model_both = "model/combined_model.h5"  # Path tempat menyimpan file model yang diunduh
gdown.download(url_model_both, output_model_both, quiet=False)
check_model_ready(output_model_both)


model_mfcc = tf.keras.models.load_model(output_model_mfcc)
model_chroma = tf.keras.models.load_model(output_model_chroma)
model_combined = tf.keras.models.load_model(output_model_both)

st.header(":blue[Klasifikasi] Maqam Bacaan :green[Al-'Quran]")

lowcut= 300
highcut= 3400

audio_file = st.file_uploader("Unggah file bacaan Al-Qur'an", type="wav")

method = st.selectbox("Pilih metode ekstraksi fitur:grey[*]", ("MFCC", "Chroma ⭐**", "MFCC dan Chroma"))

button_placeholder = st.empty()
if method:
    with st.spinner('Memproses metode ekstraksi fitur, harap tunggu...'):
        time.sleep(3)
    if audio_file is not None:
        # Menggunakan timestamp sebagai nama file unik
        file_extension = audio_file.name.split('.')[-1].lower()
        timestamp = str(int(time.time()))  # Menggunakan waktu sebagai identifier unik
        if file_extension == "mp3":
            # Konversi langsung dari memory tanpa simpan file
            mp3_bytes = io.BytesIO(audio_file.read())
            try:
                audio = AudioSegment.from_file(mp3_bytes, format="mp3")
            except Exception as e:
                st.error(f"Gagal membaca file MP3: {e}")
            else:
                # Simpan ke WAV di memory juga
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)  # reset posisi pointer
                file_path = f"converted_audio_{timestamp}.wav"
                with open(file_path, "wb") as f:
                    f.write(wav_io.read())
                st.success("MP3 berhasil dikonversi ke WAV.")
        
        else:
            # Untuk WAV, cukup simpan langsung
            file_path = f"uploaded_audio_{timestamp}.wav"
            with open(file_path, "wb") as f:
                f.write(audio_file.read())
            st.success("WAV berhasil disimpan.")
    #if method:
    #    with st.spinner('Memproses metode ekstraksi fitur, harap tunggu...'):
    #        time.sleep(3)
    # Membuat tombol prediksi
        if st.button('Prediksi'):
            with st.spinner('Proses prediksi sedang berlangsung, harap tunggu...'):
                time.sleep(3)
                if method == "Both":
                    predicted_class, max_probability = predict_audio_with_both(file_path, model_combined)
                else:
                    if method == "MFCC":
                        predicted_class, max_probability = predict_audio_with_mfcc(file_path, model_mfcc)
                    else:
                        predicted_class, max_probability = predict_audio_with_chroma(file_path, model_chroma)

                #st.write(f"Predicted Class: {predicted_class}")

                class_names = ['Ajam', 'Bayat', 'Hijaz', 'Kurd', 'Nahawand', 'Rast', 'Saba', 'Seka']  # Contoh kelas
                st.write(f"Hasil Klasifikasi: {class_names[predicted_class]}")
                st.write(f"Kemungkinan Prediksi: {max_probability:.2f}")

st.markdown("""
---
<div style="font-size: 12px; color: grey;">
Catatan: 
            
*Penjelasan Pilihan Metode Ekstraksi Fitur:
            
- MFCC: Mel Frequency Cepstral Coefficients adalah representasi fitur audio yang menggambarkan karakteristik spektral berdasarkan analisis frekuensi mel, sering digunakan dalam pengenalan suara.
            
- Chroma: Chroma features mengukur distribusi energi pada pitch tertentu dan berguna untuk analisis tonal, seperti mengenali akor atau pola melodi dalam musik.
            
- Kedua Fitur (MFCC & Chroma): Menggabungkan kedua fitur memberikan informasi yang lebih lengkap tentang audio, menggabungkan aspek temporal (MFCC) dan tonal (Chroma).


**Untuk mendapatkan hasil klasifikasi paling akurat, gunakan metode ekstraksi fitur Chroma Feature dikarenakan perbedaan tiap maqam bacaan Al-Qur'an terletak pada aspek tonal-nya.

</div>
""", unsafe_allow_html=True)