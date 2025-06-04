# 🏠 Predição de Preços de Apartamentos em Santos [EM ANDAMENTO]

🖼️🚧*Em Andamento*🚧🖼️


|               |             |
| -----------   | -----------    |
| Autor         | [Matheus Santos](https://www.linkedin.com/in/mathsantos94/) |
| Modelo        | Precificação    |
| Linguagem    | Python    |
| EDA | [Notebook](notebooks/eda.ipynb) |

## 📌 Visão geral 
Esse projeto teve como objetivo analisar o panorama geral dos apartamentos de Santos/SP e criar um modelo preditivo de preços. Foram utilizadas técnicas de análise de dados e Machine Learning para extrair insights, identificar padrões de mercado e construir um modelo capaz de prever o valor de imóveis com base em suas características.

### Objetivos Específicos

- Coletar e tratar dados de apartamentos disponíveis no site da [ZapImoveis](https://www.zapimoveis.com.br/) na cidade de Santos.

- Realizar análise exploratória para entender os principais fatores que influenciam o preço.

- Criar variáveis relevantes atráves de feature engineering  para o modelo.

- Testar diferentes algoritmos de regressão.

- Avaliar o desempenho dos modelos utilizando métricas apropriadas (ex: RMSE, MAPE, R²).

- Implementar um serviço de inferência usando FastAPI para realizar previsões em tempo real.

### 🔍 Tecnologias Utilizadas

- Python (Selenium, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

- Machine Learning: Regressão Linear, Decision Tree, Random Forest, CatBoost

- API: FastAPI

## 🧭 Entendimento do Negócio

O mercado imobiliário da cidade de Santos/SP é um dos mais valorizados do litoral paulista, com grande diversidade de imóveis em termos de metragem, localização e infraestrutura. No entanto, a precificação de apartamentos ainda é altamente subjetiva, sendo influenciada por fatores como bairro, proximidade da praia, quantidade de dormitórios, vagas de garagem e padrão de acabamento.

Nesse contexto, um modelo preditivo pode atuar como uma ferramenta de apoio à decisão, ajudando stakeholders a:

- Avaliar rapidamente o valor estimado de um imóvel.
- Identificar quais características mais impactam o preço.
- Automatizar análises comparativas de imóveis para leads ou clientes.

O objetivo deste projeto é justamente preencher essa lacuna, utilizando dados históricos de anúncios e técnicas de Machine Learning para construir um modelo robusto de avaliação automática de apartamentos em Santos.

## 💡 Análise Exploratória


## ⚙️ Instalação do projeto

**Prerequisitos**
Antes de começar, tenha certeza que você tem instalado em sua maquina:

- Python 3.10
- pip
- Git

Uma vez instalado, abra o terminal na sua maquina local e siga os passos:

1. Clone o repositorio

```
git clone https://github.com/Mathsnts18/SantosApartmentPricing.git
```
2. Vá ao diretório clonado

```
cd SantosApartmentPricing
```

3. Crie um ambiente virtual

```
python -m venv venv
```

4. Ative o ambiente virtual

Ative o ambiente virtual para isolar as dependências do projeto
```
# no Windows
venv\Scripts\activate

# no Linux
source venv/bin/activate
```

<!-- 5. Instale as dependências

Use o pip para instalar as dependências listadas no requirements.txt

```
pip install -r requirements.txt
```

6. Execute a aplicação

```
streamlit run app.py
```
Após a execução, o projeto irá abrir automaticamente. Caso isso não aconteça, digite na barra de endereço do seu navegar o `Local URL` informado no terminal

7. Desligue a aplicação

Para desligar a aplicação, dê o comando `Ctrl+C` no terminal que estiver rodando a aplicação.

8. Desative o ambiente virtual

Quando terminar de ver o projeto, desative o ambiente virtual

```
deactivate
``` -->

## Contato

Portfólio: https://www.matheussantos.com.br/

Linkedin: https://www.linkedin.com/in/mathsantos94/