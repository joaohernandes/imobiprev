import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Configuração da página
st.set_page_config(page_title='Previsão de Preço de Casas', layout='centered')

# Carregamento e pré-processamento dos dados (arquivo local)
@st.cache_data
def load_data():
    df = pd.read_csv('T1 RODRIGO.csv', sep=';')
    df['preco'] = pd.to_numeric(df['preco'], errors='coerce')
    df = df.dropna(subset=['preco'])
    return df

# Carregar dados
df = load_data()
X = df.drop('preco', axis=1)
y = df['preco']

# Definição de features
numeric_features = ['area_terreno', 'area_construida', 'quartos', 'banheiros']
ordinal_features = ['classif_bairro', 'classif_casa']
binary_features = ['casa_predio', 'energ_solar', 'mov_planejados']

# Pré-processamento
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('ord', StandardScaler(), ordinal_features),
    ('bin', 'passthrough', binary_features)
])

# Modelos candidatos e seus grids
models = [
    ('LinearRegression', LinearRegression(), {}),
    ('RandomForest', RandomForestRegressor(random_state=42), {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20]
    }),
    ('GradientBoosting', GradientBoostingRegressor(random_state=42), {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5]
    })
]

# Treina e retorna o melhor modelo
@st.cache_resource
def train_best_model(data_X, data_y):
    best_score = np.inf
    best_model = None
    for _, model, params in models:
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        if params:
            gs = GridSearchCV(pipe, param_grid=params,
                              cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
            gs.fit(data_X, data_y)
            score = -gs.best_score_
            candidate = gs.best_estimator_
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, data_X, data_y,
                                     scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
            pipe.fit(data_X, data_y)
            score = -scores.mean()
            candidate = pipe
        if score < best_score:
            best_score = score
            best_model = candidate
    return best_model

model = train_best_model(X, y)

# Estilo customizado (laranja e vermelho)
st.markdown("""
<style>
.stButton>button {background-color: #e74c3c; color: white !important;}
.stSelectbox>div, .stNumberInput input {border-color: #f39c12 !important;}
</style>
""", unsafe_allow_html=True)

# Interface do usuário
st.title('Previsão de Preço de Casas')
st.markdown('Preencha as características abaixo para obter a previsão de preço.')

with st.sidebar:
    st.header('Características')
    classif_bairro = st.selectbox('Classificação do Bairro', options=list(range(0,6)), index=3)
    area_terreno = st.number_input('Área do Terreno (m²)', min_value=0.0,
                                   value=float(df['area_terreno'].mean()), step=1.0)
    area_construida = st.number_input('Área Construída (m²)', min_value=0.0,
                                      value=float(df['area_construida'].mean()), step=1.0)
    quartos = st.number_input('Quartos', min_value=0, max_value=10,
                              value=int(df['quartos'].median()), step=1)
    banheiros = st.number_input('Banheiros', min_value=0, max_value=10,
                                value=int(df['banheiros'].median()), step=1)
    classif_casa = st.selectbox('Classificação da Casa', options=list(range(0,6)), index=3)
    casa_predio = st.selectbox('Tipo de Imóvel', options={0: 'Casa', 1: 'Prédio'})
    energ_solar = st.selectbox('Energia Solar', options={0: 'Não', 1: 'Sim'})
    mov_planejados = st.selectbox('Móveis Planejados', options={0: 'Não', 1: 'Sim'})
    predict_btn = st.button('Prever Preço')

if predict_btn:
    inp = pd.DataFrame([{
        'classif_bairro': classif_bairro,
        'area_terreno': area_terreno,
        'area_construida': area_construida,
        'quartos': quartos,
        'banheiros': banheiros,
        'classif_casa': classif_casa,
        'casa_predio': int(casa_predio),
        'energ_solar': int(energ_solar),
        'mov_planejados': int(mov_planejados)
    }])
    pred = model.predict(inp)[0]
    st.subheader(f'Preço Previsto: R$ {pred:,.2f}')
    # Botão de exportar
    csv = inp.assign(preco_previsto=pred).to_csv(index=False, sep=';')
    st.download_button('Exportar Resultado', data=csv,
                       file_name='previsao_casa.csv', mime='text/csv')
