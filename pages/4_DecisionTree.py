import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="Water Quality Prediction Decision Tree",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Water Quality Prediction Decision Tree")
st.image("grafik perbandingan metode.png", caption="Grafik Perbandingan Metode")
st.write('Dalam prediksi kali ini saya akan menggunakan model Adaboost, dimana fitur yang digunakan berjumlah 20 dengan akurasi pelatihan mencapai 98,62%.')
fitur = joblib.load('fiturdecisiontree.pkl')
st.write(fitur)

aluminium = st.number_input("Kandungan Aluminium : ")
ammonia = st.number_input("Kandungan Ammonia : ")
arsenic = st.number_input("Kandungan Arsenic : ")
barium = st.number_input("Kandungan Barium : ")
cadmium = st.number_input("Kandungan Cadmium : ")
chloramine = st.number_input("Kandungan Chloramine : ")
chromium = st.number_input("Kandungan Chromium : ")
copper = st.number_input("Kandungan Tembaga : ")
flouride = st.number_input("Kandungan Flourida : ")
bacteria = st.number_input("Kandungan Bakteri : ")
viruses = st.number_input("Kandungan Virus : ")
lead = st.number_input("Kandungan Timbal : ")
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
        'flouride' : flouride,
        'bacteria' : bacteria,
        'viruses' : viruses,
        'lead' : lead,
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
    scaler = joblib.load('saclerdecisiontree.pkl')
    data_uji_scaled = scaler.transform(data_implementasi)
    st.write('Data Inputan',data_implementasi)
    st.write('Data Normalisasi',data_uji_scaled)

    # Load the model
    dtmodel = joblib.load('modeldecisiontree.pkl')

    # Make predictions
    prediksi = dtmodel.predict(data_uji_scaled)
    if (prediksi == 0):
        st.write('Hasil Prediksi : not safe')
    elif (prediksi == 1):
        st.write('Hasil Prediksi : safe')
