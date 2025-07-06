import pandas as pd
import streamlit as st

from src.utils import load_object

# ----- ARTIFACTS -----

model_path = 'artifacts/model.pkl'
model = load_object(file_path=model_path)
scaler_path = 'artifacts/scaler.pkl'
scaler = load_object(file_path=scaler_path)

# ----- PAGE -----

st.title("""🏠 Santos Apartment Pricing Prediction""")

st.header('Previsão de Preços de Apartamentos')

iptu = st.number_input('Qual o valor do IPTU?', min_value=0, step=1, value=0)
condominio = st.number_input(
    'Qual o valor do condomínio?', min_value=0, step=1, value=0
)
area = st.number_input(
    'Qual a área do apartamento (m²)?', min_value=0, step=1, value=0
)
banheiro = st.number_input('Quantos banheiros?', min_value=0, step=1, value=0)

input_features = {
    'nrIptu': iptu,
    'nrArea(m2)': area,
    'nrBanheiros': banheiro,
    'nrPrecoFixo': condominio + iptu,
    'nrIptu_m2': iptu / area if area > 0 else 0,
}

input_df = pd.DataFrame(input_features, index=[0])
input_df_scaled = scaler.transform(input_df.values)

with st.container():
    if st.button('Prever'):
        prediction = model.predict(input_df_scaled)
        st.success(
            f'O preço previsto para o apartamento é: R$ {prediction[0]:,.2f}'
        )
    else:
        st.info(
            'Clique no botão "Prever" para obter a previsão do preço do apartamento.'
        )
