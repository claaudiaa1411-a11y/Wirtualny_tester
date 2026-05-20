import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time

st.set_page_config(page_title="Wirtualny Tester Smaku", page_icon="👅", layout="centered")

st.title("👅 WIRTUALNI TESTERZY SMAKU (AI)")
st.markdown("Stwórz własnego cukierka w suwakach, a mój model Sztucznej Inteligencji przewidzi, jaką ocenę smaku (od 1 do 10) dostałby na rynku!")

@st.cache_resource
def trenuj_model_testera():
    df = pd.read_csv('cukrasy_clean.csv')
    X = df[['szerokosc', 'dlugosc', 'twardosc', 'kwas']]
    y = df['smaczek']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = trenuj_model_testera()

st.subheader("⚙️ Zaprojektuj cukierka:")
szerokosc = st.slider("📐 Szerokość [mm]", 1.0, 50.0, 20.0, step=0.5)
dlugosc = st.slider("📏 Długość [mm]", 5.0, 200.0, 50.0, step=1.0)
twardosc = st.slider("🧱 Twardość (-1 miękki do 1 twardy)", -1.0, 1.0, 0.0, step=0.1)
kwas = st.slider("🍋 Kwasowość (-1 słodki do 1 kwaśny)", -1.0, 1.0, 0.0, step=0.1)

if st.button("👅 Daj wirtualnym testerom do spróbowania", use_container_width=True):
    dane_testowe = np.array([[szerokosc, dlugosc, twardosc, kwas]])
    
    with st.spinner('AI degustuje i analizuje dane... ⏳'):
        time.sleep(1.5)
        przewidywana_ocena = model.predict(dane_testowe)[0]
    
    st.divider()
    st.subheader("📊 Wyniki testu konsumenckiego:")
    
    if przewidywana_ocena >= 8.5:
        st.success(f"### 🔥 ABSOLUTNY BESTSELLER! Ocena: {przewidywana_ocena:.1f} / 10.0")
        st.write("Klienci rzucą się na to w sklepach!")
        st.balloons()
    elif przewidywana_ocena >= 6.5:
        st.info(f"### 👍 CAŁKIEM NIEŹLE! Ocena: {przewidywana_ocena:.1f} / 10.0")
        st.write("Solidny produkt, będzie się dobrze sprzedawał.")
    elif przewidywana_ocena >= 4.0:
        st.warning(f"### 🤷 ŚREDNIAK... Ocena: {przewidywana_ocena:.1f} / 10.0")
        st.write("Część osób zje, reszta wyrzuci. Trzeba by coś poprawić.")
    else:
        st.error(f"### 🤮 KATASTROFA! Ocena: {przewidywana_ocena:.1f} / 10.0")
        st.write("Algorytm twierdzi, że nikt nie będzie chciał tego jeść!")
