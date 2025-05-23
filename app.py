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

# Upload do arquivo CSV na sidebar
st.sidebar.title('Upload de Dados')
uploaded_file = st.sidebar.file_uploader('Selecione o CSV de dados', type=['csv'])
if not uploaded_file:
    st.sidebar.warning('Faça upload do arquivo CSV para prosseguir.')
    st.stop()

# Carregamento e pré-processamento dos dados
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=';')
    df['preco'] = pd.to_numeric(df['preco'], errors='coerce')
    df = df.dropna(subset=['preco'])
    return df

# Carregar os dados
df = load_data(uploaded_file)

# Separar features e target
X = df.drop('preco', axis=1)
 y = df['preco']
# Definição das features
numeric_features = ['area_terreno', 'area_construida', 'quartos', 'banheiros']
ordinal_features = ['classif_bairro', 'classif_casa']
binary_features = ['casa_predio', 'energ_solar', 'mov_planejados']

# Pré-processador
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('ord', StandardScaler(), ordinal_features),
    ('bin', 'passthrough', binary_features)
])

# Modelos e parâmetros
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

# Treinamento do melhor modelo (cache para performance)
@st.cache_resource
def train_best_model(data_X, data_y):
    best_score = np.inf
    best_model = None
    for _, model, params in models:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        if params:
            gs = GridSearchCV(
                pipeline,
                param_grid=params,
                cv=5,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            gs.fit(data_X, data_y)
            score = -gs.best_score_
            candidate = gs.best_estimator_
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                pipeline, data_X, data_y,
                scoring='neg_root_mean_squared_error',
                cv=cv,
                n_jobs=-1
            )
            pipeline.fit(data_X, data_y)
            score = -scores.mean()
            candidate = pipeline
        if score < best_score:
            best_score = score
            best_model = candidate
    return best_model

model = train_best_model(X, y)

# Estilo customizado de cores (laranja e vermelho)
st.markdown("""
<style>
.stButton>button {background-color: #e74c3c; color: white !important;}
.stNumberInput input, .stSlider>div {border-color: #f39c12 !important;}
</style>
""", unsafe_allow_html=True)

# Interface do usuário para predição
st.sidebar.title('Características da Casa')
classif_bairro = st.sidebar.slider('Classificação do Bairro', 0, 5, 3)
area_terreno = st.sidebar.number_input('Área do Terreno (m²)', min_value=0.0, value=float(df['area_terreno'].mean()))
area_construida = st.sidebar.number_input('Área Construída (m²)', min_value=0.0, value=float(df['area_construida'].mean()))
quartos = st.sidebar.number_input('Quartos', min_value=0, max_value=10, value=int(df['quartos'].median()))
banheiros = st.sidebar.number_input('Banheiros', min_value=0, max_value=10, value=int(df['banheiros'].median()))
classif_casa = st.sidebar.slider('Classificação da Casa', 0, 5, 3)
casa_predio = st.sidebar.checkbox('Prédio', value=False)
energ_solar = st.sidebar.checkbox('Energia Solar', value=False)
mov_planejados = st.sidebar.checkbox('Móveis Planejados', value=False)

if st.sidebar.button('Prever Preço'):
    input_data = pd.DataFrame([{
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
    prediction = model.predict(input_data)[0]
    st.subheader(f'Preço Previsto: R$ {prediction:,.2f}')
    # Exportar resultado
    result = input_data.copy()
    result['preco_previsto'] = prediction
    csv = result.to_csv(index=False, sep=';')
    st.download_button('Exportar Resultado', data=csv, file_name='previsao_casa.csv', mime='text/csv')
