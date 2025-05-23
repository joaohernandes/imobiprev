import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuração da página
st.set_page_config(page_title='Previsão de Preço de Casas', layout='wide')

# Estilos customizados
st.markdown("""
<style>
body {background-color: #FAFAFA;}
.stApp {font-family: 'Arial', sans-serif;}
h1 {color: #e74c3c;}
.stButton>button {background-color: #e74c3c; border-radius: 8px; color: white !important; padding: 0.6em 1.4em;}
.stSelectbox>div, .stNumberInput>div > input {border: 2px solid #f39c12 !important; border-radius: 4px;}
.stDivider {margin: 2rem 0;}
.card {background-color: white; border: 1px solid #ddd; border-radius: 8px; padding: 1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
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
    ('GB', GradientBoostingRegressor(random_state=42), {'model__n_estimators':[100,200], 'model__learning_rate':[0.05,0.1]})
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

# Avaliar métricas
cv = KFold(5, shuffle=True, random_state=42)
rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv))
r2 = cross_val_score(model, X, y, scoring='r2', cv=cv)

# Cabeçalho
st.markdown("""
<div style='text-align:center;'>
    <h1>🏠 Previsão de Preço de Casas</h1>
    <p style='color: #555;'>Utilize as ferramentas abaixo para estimar o valor de mercado.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('---')

# Layout em três colunas
col1, col2, col3 = st.columns((3,4,3))

# Card de métricas
with col2:
    st.markdown("""
    <div class='card'>
        <h3>📊 Desempenho do Modelo</h3>
        <p><b>Modelo:</b> {}</p>
        <p><b>RMSE (média):</b> {:.2f}</p>
        <p><b>R² (média):</b> {:.2f}</p>
    </div>
    """.format(model.named_steps['model'].__class__.__name__, rmse.mean(), r2.mean()), unsafe_allow_html=True)

# Entrada de dados do usuário
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader('Características da Casa')
inputs = {}
inputs['classif_bairro'] = st.selectbox('Bairro (0-5)', list(range(6)), 3)
inputs['area_terreno'] = st.slider('Área do Terreno (m²)', 0.0, float(df['area_terreno'].max()), float(df['area_terreno'].mean()))
inputs['area_construida'] = st.slider('Área Construída (m²)', 0.0, float(df['area_construida'].max()), float(df['area_construida'].mean()))
inputs['quartos'] = st.slider('Quartos', 0, int(df['quartos'].max()), int(df['quartos'].median()))
inputs['banheiros'] = st.slider('Banheiros', 0, int(df['banheiros'].max()), int(df['banheiros'].median()))
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
