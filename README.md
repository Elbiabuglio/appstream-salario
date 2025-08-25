# Previsão de Salários

Este projeto consiste em uma aplicação de Machine Learning para previsão de salários utilizando MLflow para gerenciamento de experimentos e Streamlit para a interface do usuário.

## 📋 Descrição

A aplicação permite prever salários com base em diferentes características, utilizando modelos de machine learning treinados e gerenciados através do MLflow. A interface web interativa foi desenvolvida com Streamlit para facilitar o uso.

## 🚀 Tecnologias Utilizadas

- Python
- MLflow - Para gerenciamento de experimentos e modelos
- Streamlit - Para a interface web interativa
- Plotly - Para visualizações de dados
- Scikit-learn - Para modelos de machine learning

## 📁 Estrutura do Projeto

```
├── app.py              # Aplicação Streamlit
├── train.py           # Script de treinamento do modelo
├── data/              
│   ├── Dataset.csv    # Dataset principal
│   └── template.csv   # Template para dados
├── mlartifacts/       # Artefatos do MLflow
└── mlruns/            # Registros de execuções do MLflow
```

## 🛠️ Configuração do Ambiente

1. Crie um ambiente conda:
```bash
conda create -n appstream-salario python=3.x
conda activate appstream-salario
```

2. Instale as dependências:
```bash
pip install mlflow streamlit plotly scikit-learn pandas numpy
```

## 🎯 Como Usar

1. Inicie o servidor MLflow:
```bash
mlflow ui
```

2. Em outro terminal, execute a aplicação Streamlit:
```bash
streamlit run app.py
```

3. Acesse a aplicação através do navegador no endereço indicado pelo Streamlit.

## 📊 Experimentação

O projeto utiliza MLflow para rastrear experimentos, permitindo:
- Registro de diferentes modelos e seus parâmetros
- Acompanhamento de métricas de performance
- Visualização de resultados através da interface do MLflow
- Gerenciamento de versões dos modelos

## 📈 Métricas Monitoradas

- Acurácia (accuracy_score)
- F1-Score
- Precisão (precision_score)
- Recall
- ROC AUC
- Log Loss

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## 📝 Licença

Este projeto está sob a licença MIT.
