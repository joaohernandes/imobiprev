import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score

# Configuração da página
st.set_page_config(page_title='Previsão de Preço de Casas', layout='wide')

# Estilos customizados
st.markdown("""
<style>
body {background-color: #FAFAFA;}
.stApp {font-family: 'Arial', sans-serif;}
h1 {color: #e74c3c;}
.stButton>button {background-color: #e74c3c; border-radius: 8px; color: white !important; padding: 0.6em 1.4em;}
.stSelectbox>div, .stNumberInput>div > input, .stSlider>div {border: 2px solid #f39c12 !important; border-radius: 4px;}
.card {background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 1.5em;}
</style>
""", unsafe_allow_html=True)

# Carregamento e pré-processamento dos dados
@st.cache_data
def load_data():
    df = pd.read_csv('T1 RODRIGO.csv', sep=';')
    df['preco'] = pd.to_numeric(df['preco'], errors='coerce')
    return df.dropna(subset=['preco'])

df = load_data()
X = df.drop('preco', axis=1)
y = df['preco']

# Definição de features
numeric = ['area_terreno', 'area_construida', 'quartos', 'banheiros']
ordinal = ['classif_bairro', 'classif_casa']
binary = ['casa_predio', 'energ_solar', 'mov_planejados']

# Pré-processamento e modelos
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('ord', StandardScaler(), ordinal),
    ('bin', 'passthrough', binary)
])
models = [
    ('Linear', LinearRegression(), {}),
    ('RF', RandomForestRegressor(random_state=42), {'model__n_estimators': [100,200], 'model__max_depth':[None,10]}),
    ('GB', GradientBoostingRegressor(random_state=42), {'model__n_estimators':[100,200], 'model__learning_rate':[0.05,0.1]}),
    ('ET', ExtraTreesRegressor(random_state=42), {'model__n_estimators':[100,200], 'model__max_depth':[None,10]}),
    ('SVR', SVR(), {'model__C':[0.1,1,10], 'model__kernel':['rbf','linear']})
]

@st.cache_resource
def train_model(X, y):
    best_score, best_model = np.inf, None
    for name, m, params in models:
        pipe = Pipeline([('prep', preprocessor), ('model', m)])
        if params:
            gs = GridSearchCV(pipe, params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
            gs.fit(X, y)
            score = -gs.best_score_
            candidate = gs.best_estimator_
        else:
            cv = KFold(5, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_root_mean_squared_error')
            pipe.fit(X, y)
            score = -scores.mean()
            candidate = pipe
        if score < best_score:
            best_score, best_model = score, candidate
    return best_model

model = train_model(X, y)

# Avaliação das métricas via CV
cv = KFold(5, shuffle=True, random_state=42)
rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))
mae = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
medae = -cross_val_score(model, X, y, scoring='neg_median_absolute_error', cv=cv)
ev = cross_val_score(model, X, y, scoring='explained_variance', cv=cv)

# Cabeçalho
st.markdown("""
<div style='text-align:center;'>
    <h1>🏠 Previsão de Preço de Casas</h1>
    <p style='color: #555;'>Estimativa de valor baseada em múltiplos modelos de regressão.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('---')

# Layout em colunas
_, mid, _ = st.columns((1,2,1))

# Card de métricas e explicações
with mid:
    st.markdown(f"""
    <div class='card'>
        <h3>📊 Desempenho do Modelo ({model.named_steps['model'].__class__.__name__})</h3>
        <p><b>RMSE:</b> {rmse.mean():.2f} ± {rmse.std():.2f}</p>
        <p><b>MAE:</b> {mae.mean():.2f} ± {mae.std():.2f}</p>
        <p><b>MedAE:</b> {medae.mean():.2f} ± {medae.std():.2f}</p>
        <p><b>Explained Var:</b> {ev.mean():.2f} ± {ev.std():.2f}</p>
        <hr>
        <h4>O que significam essas métricas?</h4>
        <p><b>RMSE</b> penaliza erros grandes, representando o desvio médio das previsões em R$.</p>
        <p><b>MAE</b> é o erro médio absoluto, menos sensível a outliers.</p>
        <p><b>MedAE</b> mostra a mediana dos erros absolutos, robusta a valores extremos.</p>
        <p><b>Explained Var</b> indica a porcentagem de variação nos preços que o modelo explica.</p>
    </div>
    """, unsafe_allow_html=True)

# Seção de inputs
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader('Características da Casa')
inputs = {}
inputs['classif_bairro'] = st.selectbox('Bairro (0-5)', list(range(6)), 3)
inputs['area_terreno'] = st.number_input('Área do Terreno (m²)', min_value=0.0, value=float(df['area_terreno'].mean()), step=1.0)
inputs['area_construida'] = st.number_input('Área Construída (m²)', min_value=0.0, value=float(df['area_construida'].mean()), step=1.0)
inputs['quartos'] = st.number_input('Quartos', min_value=0, max_value=int(df['quartos'].max()), value=int(df['quartos'].median()), step=1)
inputs['banheiros'] = st.number_input('Banheiros', min_value=0, max_value=int(df['banheiros'].max()), value=int(df['banheiros'].median()), step=1)
inputs['classif_casa'] = st.selectbox('Condição (0-5)', list(range(6)), 3)
inputs['casa_predio'] = st.radio('Tipo de Imóvel', ('Casa','Prédio'))
inputs['energ_solar'] = st.checkbox('Energia Solar')
inputs['mov_planejados'] = st.checkbox('Móveis Planejados')

predict = st.button('Calcular Preço')
st.markdown('</div>', unsafe_allow_html=True)

# Predição e resultado
if predict:
    df_inp = pd.DataFrame([{ 
        'classif_bairro': inputs['classif_bairro'],
        'area_terreno': inputs['area_terreno'],
        'area_construida': inputs['area_construida'],
        'quartos': inputs['quartos'],
        'banheiros': inputs['banheiros'],
        'classif_casa': inputs['classif_casa'],
        'casa_predio': 1 if inputs['casa_predio']=='Prédio' else 0,
        'energ_solar': int(inputs['energ_solar']),
        'mov_planejados': int(inputs['mov_planejados'])
    }])
    price = model.predict(df_inp)[0]
    st.markdown(f"<div class='card'><h2>Preço Estimado: R$ {price:,.2f}</h2></div>", unsafe_allow_html=True)
    csv = df_inp.assign(preco_previsto=price).to_csv(index=False, sep=';')
    st.download_button('Exportar Resultado', csv, file_name='previsao.csv', mime='text/csv')
