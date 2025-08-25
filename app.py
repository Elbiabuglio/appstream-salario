# %%
import pandas as pd
import streamlit as st
import mlflow
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

# Configuração da página
st.set_page_config(
    page_title="Previsão Salarial - Área de Dados",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource(ttl='1day')
def load_model():
    """Carrega o modelo MLflow com tratamento de erros aprimorado"""
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        models = [i for i in mlflow.search_registered_models() if i.name == "salario-model"]
        
        if not models:
            st.error("❌ Modelo 'salario-model' não encontrado no MLflow")
            st.stop()
            
        last_version = max([int(i.version) for i in models[0].latest_versions])
        model = mlflow.sklearn.load_model(f"models:/salario-model/{last_version}")
        
        logger.info(f"Modelo carregado com sucesso - versão {last_version}")
        return model, last_version
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")
        st.info("💡 Verifique se o MLflow está rodando em http://127.0.0.1:5000")
        st.stop()

@st.cache_data(ttl='1hour')
def load_template_data():
    """Carrega dados do template com tratamento de erros"""
    try:
        data = pd.read_csv("data/template.csv")
        return data
    except FileNotFoundError:
        st.error("❌ Arquivo 'data/template.csv' não encontrado")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        st.stop()

def create_salary_visualization(predicted_salary):
    """Cria visualização do salário predito"""
    # Faixas salariais típicas para contexto
    salary_ranges = {
        "Júnior": "R$ 3.000 - R$ 6.000",
        "Pleno": "R$ 6.000 - R$ 12.000", 
        "Sênior": "R$ 12.000 - R$ 20.000",
        "Especialista": "R$ 20.000+"
    }
    
    fig = go.Figure()
    
    # Adicionar barras de contexto
    ranges = [6000, 12000, 20000, 30000]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    labels = list(salary_ranges.keys())
    
    for i, (range_val, color, label) in enumerate(zip(ranges, colors, labels)):
        fig.add_trace(go.Bar(
            x=[label],
            y=[range_val],
            name=label,
            marker_color=color,
            text=salary_ranges[label],
            textposition='inside'
        ))
    
    fig.update_layout(
        title="Contexto Salarial por Nível",
        xaxis_title="Nível",
        yaxis_title="Salário (R$)",
        showlegend=False,
        height=400
    )
    
    return fig

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1>💰 Preditor de Salário - Área de Dados</h1>
    <p>Descubra sua faixa salarial baseada no mercado atual de dados</p>
</div>
""", unsafe_allow_html=True)

# Carregar dados e modelo
with st.spinner("🔄 Carregando modelo e dados..."):
    model, model_version = load_model()
    data_template = load_template_data()

# Sidebar com informações do modelo
with st.sidebar:
    st.markdown("### 📊 Informações do Modelo")
    st.info(f"**Versão:** {model_version}")
    st.info(f"**Última atualização:** {datetime.now().strftime('%d/%m/%Y')}")
    
    # Estatísticas dos dados
    st.markdown("### 📈 Estatísticas dos Dados")
    total_registros = len(data_template)
    st.metric("Total de Registros", total_registros)
    
    unique_cargos = data_template["cargoAtual"].nunique()
    st.metric("Cargos Únicos", unique_cargos)
    
    unique_ufs = data_template["ufOndeMora"].nunique()
    st.metric("UFs Representadas", unique_ufs)

# Formulário principal
with st.form("salary_prediction_form"):
    st.markdown("## 👤 Seus Dados Profissionais")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📋 Informações Pessoais")
        
        idade = st.number_input(
            "Idade",
            min_value=int(data_template["idade"].min()),
            max_value=100,
            value=30,
            help="Sua idade atual"
        )
        
        genero = st.selectbox(
            "Gênero",
            options=data_template["genero"].unique(),
            help="Seu gênero"
        )
        
        pcd = st.selectbox(
            "Pessoa com Deficiência (PcD)",
            options=data_template["pcd"].unique(),
            help="Você é uma pessoa com deficiência?"
        )
        
        ufs = data_template["ufOndeMora"].sort_values().unique().tolist()
        uf = st.selectbox(
            "Estado (UF)",
            options=ufs,
            help="Estado onde você mora/trabalha"
        )
    
    with col2:
        st.markdown("### 💼 Informações Profissionais")
        
        cargos = data_template["cargoAtual"].sort_values().unique().tolist()
        cargo = st.selectbox(
            "Cargo Atual",
            options=cargos,
            help="Seu cargo/função atual"
        )
        
        niveis = data_template["nivel"].sort_values().unique()
        nivel = st.selectbox(
            "Nível",
            options=niveis,
            help="Seu nível de senioridade"
        )
    
    with col3:
        st.markdown("### ⏱️ Experiência")
        
        temp_dados = data_template["tempoDeExperienciaDados"].sort_values().unique().tolist()
        tempo_exp_dados = st.selectbox(
            "Tempo de Experiência em Dados",
            options=temp_dados,
            help="Há quanto tempo trabalha com dados?"
        )
        
        temp_ti = data_template["tempoDeExperienciaEmTi"].sort_values().unique().tolist()
        tempo_exp_ti = st.selectbox(
            "Tempo de Experiência em TI",
            options=temp_ti,
            help="Há quanto tempo trabalha com TI em geral?"
        )
    
    # Botão de predição
    submitted = st.form_submit_button(
        "🎯 Previsão Salarial",
        type="primary",
        use_container_width=True
    )

# Processamento da predição
if submitted:
    with st.spinner("🔮 Calculando sua faixa salarial..."):
        # Preparar dados para predição
        user_data = pd.DataFrame([{
            "idade": idade,
            "genero": genero,
            "pcd": pcd,
            "ufOndeMora": uf,
            "cargoAtual": cargo,
            "nivel": nivel,
            "tempoDeExperienciaDados": tempo_exp_dados,
            "tempoDeExperienciaEmTi": tempo_exp_ti,
        }])
        
        try:
            # Fazer predição
            salario_pred = model.predict(user_data[model.feature_names_in_])[0]
            salario_limpo = salario_pred.split("- ")[-1]
            
            # Exibir resultado
            st.markdown("---")
            st.markdown("## 🎯 Estimativa Salarial")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"### 💰 Sua faixa salarial estimada:  `{salario_limpo}`")
                
                # Adicionar contexto
                st.markdown("""
                **📊 Esta predição foi baseada em:**
                - Dados do mercado brasileiro de tecnologia
                - Seu perfil profissional informado
                - Análise de salários por região e senioridade
                """)
                
                # Disclaimer
                st.warning("""
                ⚠️ **Importante:** Esta é uma estimativa baseada em dados históricos. 
                Salários podem variar significativamente baseado em:
                - Empresa e setor
                - Benefícios oferecidos
                - Negociação individual
                - Conjuntura econômica atual
                """)
            
            with col2:
                # Métricas adicionais
                st.markdown("### 📈 Informações Adicionais")
                
                # Contagem de profissionais similares
                similar_profiles = data_template[
                    (data_template["nivel"] == nivel) &
                    (data_template["ufOndeMora"] == uf)
                ]
                
                st.metric(
                    "Perfis Similares na Base", 
                    len(similar_profiles),
                    help="Quantidade de profissionais com perfil similar nos dados"
                )
                
                # Representatividade do cargo
                cargo_count = len(data_template[data_template["cargoAtual"] == cargo])
                cargo_percentage = (cargo_count / len(data_template)) * 100
                
                st.metric(
                    "Representatividade do Cargo",
                    f"{cargo_percentage:.1f}%",
                    help="Percentual deste cargo na base de dados"
                )
            
            # Visualização (opcional)
            if st.checkbox("📊 Mostrar contexto visual", value=False):
                fig = create_salary_visualization(salario_limpo)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erro na predição: {str(e)}")
            logger.error(f"Erro na predição: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>💡 Desenvolvido com Streamlit e MLflow | 
    📊 Dados baseados no mercado brasileiro de tecnologia</p>
</div>
""", unsafe_allow_html=True)