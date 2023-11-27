import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="Water Quality Prediction Random Forest",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Water Quality Prediction Random Forest")
st.image("grafik perbandingan metode.png", caption="Grafik Perbandingan Metode")
st.write('Dalam prediksi kali ini saya akan menggunakan model random forest, dimana fitur yang digunakan berjumlah 17 dengan akurasi pelatihan mencapai 98,71%.')
fitur = joblib.load('fiturrandomforest.pkl')
st.write(fitur)

aluminium = st.number_input("Kandungan Aluminium : ")
ammonia = st.number_input("Kandungan Ammonia : ")
arsenic = st.number_input("Kandungan Arsenic : ")
barium = st.number_input("Kandungan Barium : ")
cadmium = st.number_input("Kandungan Cadmium : ")
chloramine = st.number_input("Kandungan Chloramine : ")
chromium = st.number_input("Kandungan Chromium : ")
copper = st.number_input("Kandungan Tembaga : ")
viruses = st.number_input("Kandungan Virus : ")
nitrates = st.number_input("Kandungan Nitrat : ")
nitrites = st.number_input("Kandungan Nitrit : ")
mercury = st.number_input("Kandungan Mercuri : ")
perchlorate = st.number_input("Kandungan Perchlorate : ")
radium = st.number_input("Kandungan Radium : ")
selenium = st.number_input("Kandungan Selenium : ")
silver = st.number_input("Kandungan Perak : ")
uranium = st.number_input("Kandungan Uranium : ")


results = []
data = {'aluminium' : aluminium,
        'ammonia' : ammonia,
        'arsenic' : arsenic,
        'barium' : barium,
        'cadmium' : cadmium,
        'chloramine' : chloramine,
        'chromium' : chromium,
        'copper' : copper,
        'viruses' : viruses,
        'nitrates' : nitrates,
        'nitrites' : nitrites,
        'mercury' : mercury,
        'perchlorate' : perchlorate,
        'radium' : radium,
        'selenium' : selenium,
        'silver' : silver,
        'uranium' : uranium}


results.append(data)
data_implementasi = pd.DataFrame(results)


if st.button("Cek Prediksi"):
    scaler = joblib.load('saclerrandomforest.pkl')
    data_uji_scaled = scaler.transform(data_implementasi)
    st.write('Data Inputan',data_implementasi)
    st.write('Data Normalisasi',data_uji_scaled)

    # Load the model
    rfmodel = joblib.load('modelrandomforest.pkl')

    # Make predictions
    prediksi = rfmodel.predict(data_uji_scaled)
    if (prediksi == 0):
        st.write('Hasil Prediksi : not safe')
    elif (prediksi == 1):
        st.write('Hasil Prediksi : safe')
