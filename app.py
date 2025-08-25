# %%
import pandas as pd
import streamlit as st
import mlflow

@st.cache_resource(ttl='1day')
def load_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    models =[i for i in  mlflow.search_registered_models() if i.name=="salario-model"]
    last_version = max([int(i.version) for i in models[0].latest_versions])
    model = mlflow.sklearn.load_model(f"models:///salario-model/{last_version}")
    return model

model = load_model()

# %%
data_template = pd.read_csv("data/template.csv")

st.markdown("# Data Salary")

col1, col2, col3 = st.columns(3)

with col1:
    idade = st.number_input("Idade",
                            min_value=data_template["idade"].min(),
                            max_value=100)

    genero = st.selectbox("Genero",
                        options=data_template["genero"].unique())

    pcd = st.selectbox("PDC", options=data_template["pcd"].unique())


    ufs = data_template["ufOndeMora"].sort_values().unique().tolist()
    uf = st.selectbox("UF", options=ufs)

with col2:
    cargos = data_template["cargoAtual"].sort_values().unique().tolist()
    cargo = st.selectbox("Cargo Atual", options=cargos)

    niveis = data_template["nivel"].sort_values().unique()    
    nivel = st.selectbox("Nível", options=niveis)

with col3:
    temp_dados = data_template["tempoDeExperienciaDados"].sort_values().unique().tolist()
    tempo_exp_dados = st.selectbox("Tempo de Exp. em Dados", options=temp_dados)

    temp_ti = data_template["tempoDeExperienciaEmTi"].sort_values().unique().tolist()
    tempo_exp_ti = st.selectbox("Tempo de Exp. em TI", options=temp_ti)


data = pd.DataFrame([{
    "idade":idade,
    "genero":genero,
    "pcd":pcd,
    "ufOndeMora":uf,
    "cargoAtual":cargo,
    "nivel":nivel,
    "tempoDeExperienciaDados":tempo_exp_dados,
    "tempoDeExperienciaEmTi":tempo_exp_ti,
}])

salario = model.predict(data[model.feature_names_in_])[0]
salario = salario.split("- ")[-1]
st.markdown(f"Sua faixa salarial: `{salario}`")