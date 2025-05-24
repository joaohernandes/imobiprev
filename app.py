import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

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

# Função para carregar dados
@st.cache_data
def load_data():
    df = pd.read_csv('T1 RODRIGO.csv', sep=';')
    df['preco'] = pd.to_numeric(df['preco'], errors='coerce')
    return df.dropna(subset=['preco'])

# Carregar dataset
df = load_data()
X = df.drop('preco', axis=1)
y = df['preco']

# Definir features e pré-processador
numeric = ['area_terreno', 'area_construida', 'quartos', 'banheiros']
ordinal = ['classif_bairro', 'classif_casa']
binary = ['casa_predio', 'energ_solar', 'mov_planejados']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('ord', StandardScaler(), ordinal),
    ('bin', 'passthrough', binary)
])

# Modelos e grids de hiperparâmetros
models = {
    'Linear Regression': (LinearRegression(), {}),
    'Ridge': (Ridge(), {'model__alpha': [0.1, 1, 10]}),
    'Lasso': (Lasso(), {'model__alpha': [0.01, 0.1, 1]}),
    'KNN': (KNeighborsRegressor(), {'model__n_neighbors': [3, 5, 7]}),
    'Random Forest': (RandomForestRegressor(random_state=42), {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10]}),
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42), {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}),
    'Extra Trees': (ExtraTreesRegressor(random_state=42), {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10]}),
    'AdaBoost': (AdaBoostRegressor(random_state=42), {'model__n_estimators': [50, 100]}),
    'SVR': (SVR(), {'model__C': [0.1, 1, 10], 'model__kernel': ['rbf', 'linear']})
}

# Treinar todos os modelos
@st.cache_resource
def train_all_models(X, y):
    trained = {}
    for name, (model, params) in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        if params:
            gs = GridSearchCV(pipe, params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
            gs.fit(X, y)
            trained[name] = gs.best_estimator_
        else:
            pipe.fit(X, y)
            trained[name] = pipe
    return trained

trained_models = train_all_models(X, y)

# Seleção do modelo
st.title('🏠 Previsão de Preço de Casas')
model_name = st.selectbox('Escolha o modelo:', list(trained_models.keys()))
model = trained_models[model_name]

# Avaliar métricas via CV
cv = KFold(5, shuffle=True, random_state=42)
metrics = {
    'R²': cross_val_score(model, X, y, scoring='r2', cv=cv),
    'RMSE': np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)),
    'MAE': -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
}

# Mostrar métricas
st.markdown('---')
_, mid, _ = st.columns((1, 2, 1))
with mid:
    st.markdown(f"""
    <div class='card'>
        <h3>📊 Desempenho: {model_name}</h3>
        <p><b>R²:</b> {metrics['R²'].mean():.2f} ± {metrics['R²'].std():.2f}</p>
        <p><b>RMSE:</b> {metrics['RMSE'].mean():.2f} ± {metrics['RMSE'].std():.2f}</p>
        <p><b>MAE:</b> {metrics['MAE'].mean():.2f} ± {metrics['MAE'].std():.2f}</p>
    </div>
    """, unsafe_allow_html=True)

# Inputs do usuário
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader('Características da Casa')
inputs = {}
inputs['classif_bairro'] = st.selectbox('Classif. Bairro (0-5)', list(range(6)), 3)
inputs['area_terreno'] = st.number_input('Área Terreno (m²)', min_value=0.0, value=float(df['area_terreno'].mean()), step=1.0)
inputs['area_construida'] = st.number_input('Área Construída (m²)', min_value=0.0, value=float(df['area_construida'].mean()), step=1.0)
inputs['quartos'] = st.number_input('Quartos', min_value=0, max_value=int(df['quartos'].max()), value=int(df['quartos'].median()), step=1)
inputs['banheiros'] = st.number_input('Banheiros', min_value=0, max_value=int(df['banheiros'].max()), value=int(df['banheiros'].median()), step=1)
inputs['classif_casa'] = st.selectbox('Classif. Casa (0-5)', list(range(6)), 3)
inputs['casa_predio'] = st.radio('Tipo Imóvel', ('Casa', 'Prédio'))
inputs['energ_solar'] = st.checkbox('Energia Solar')
inputs['mov_planejados'] = st.checkbox('Móveis Planejados')
predict = st.button('Prever Preço')
st.markdown('</div>', unsafe_allow_html=True)

# Predição
if predict:
    df_inp = pd.DataFrame([{ 
        'classif_bairro': inputs['classif_bairro'],
        'area_terreno': inputs['area_terreno'],
        'area_construida': inputs['area_construida'],
        'quartos': inputs['quartos'],
        'banheiros': inputs['banheiros'],
        'classif_casa': inputs['classif_casa'],
        'casa_predio': 1 if inputs['casa_predio'] == 'Prédio' else 0,
        'energ_solar': int(inputs['energ_solar']),
        'mov_planejados': int(inputs['mov_planejados'])
    }])
    price = model.predict(df_inp)[0]
    st.markdown(f"<div class='card'><h2>Preço Estimado: R$ {price:,.2f}</h2></div>", unsafe_allow_html=True)
    csv = df_inp.assign(preco_previsto=price).to_csv(index=False, sep=';')
    st.download_button('Exportar Resultado', data=csv, file_name='previsao.csv', mime='text/csv')
