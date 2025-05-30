# 🎬 TMDB Box Office Predictor

**Predição de Bilheteria de Filmes usando Machine Learning**

Instituto Federal do Maranhão - IFMA Campus Imperatriz  
Disciplina: Introdução à Inteligência Artificial  
Professor: Daniel Duarte Costa

---

## 📋 Sobre o Projeto

Este projeto implementa algoritmos de Machine Learning **do zero** (sem sklearn) para predição de bilheteria de filmes utilizando o dataset TMDB. Oferece duas formas de uso:

1. **🌐 Interface Web Interativa** - Predições em tempo real
2. **📊 Análise Acadêmica Completa** - Gráficos e visualizações para artigo científico

### 🎯 Algoritmos Implementados

- ✅ **Regressão Linear** com gradiente descendente
- ✅ **Regressão Logística** para classificação binária
- ✅ **Métricas completas**: MSE, RMSE, MAE, R², Acurácia, Precisão, Recall, F1-Score
- ✅ **Visualizações**: Curvas de convergência, matriz de confusão, gráficos exploratórios

---

## 🏗️ Estrutura do Projeto

```
tmdb-ml-project/
├── 📁 data/                    # Datasets TMDB
│   ├── train.csv              # Dataset de treinamento (27MB)
│   ├── test.csv               # Dataset de teste (40MB)
│   └── sample_submission.csv   # Exemplo de submissão
├── 📁 web/                     # Interface web
│   ├── 📁 templates/          # Templates HTML
│   └── 📁 static/             # CSS e JavaScript
├── 📁 docs/                    # Documentação
│   └── template_artigo.md     # Template para artigo científico
├── app.py                     # Servidor Flask (interface web)
├── main.py                    # Análise acadêmica completa
├── main_simple.py             # Classes ML implementadas do zero
├── requirements.txt           # Dependências Python
├── start.sh                   # Script de inicialização
└── README.md                  # Este arquivo
```

---

## 🚀 Como Executar

### ⚡ Início Rápido

```bash
# 1. Entrar no diretório do projeto
cd tmdb-ml-project

# 2. Executar o script de start
./start.sh
```

### 🎯 Opções Disponíveis

**1️⃣ Interface Web Interativa**

- 🌐 Demonstração no navegador
- 🔮 Predições em tempo real
- 📱 Design responsivo

**2️⃣ Análise Acadêmica Completa**

- 📊 Gráficos e visualizações profissionais
- 🎓 Ideal para artigo científico
- 📈 Curvas ROC, matrizes de confusão

**3️⃣ Ambos (Recomendado para avaliação)**

- 📋 Análise completa + Interface web
- 🎯 Demonstração completa do projeto

---

## 🌐 Interface Web

### Como Usar

1. Execute `./start.sh` e escolha opção **1** ou **3**
2. Acesse **http://localhost:5000** no navegador
3. Use as 4 abas disponíveis:

#### 🔮 Preditor

- Insira **Orçamento**, **Popularidade** e **Duração**
- Veja predições de **Regressão Linear** e **Logística**

#### 📊 Análise

- Execute análise completa dos algoritmos
- Visualize métricas e gráficos de convergência

#### 🎯 Versões

- Compare versão acadêmica vs web
- Execute análise completa via interface

#### ℹ️ Sobre

- Documentação técnica completa
- Detalhes da implementação

---

## 📊 Análise Acadêmica

### Como Executar

```bash
# Opção 1: Via script
./start.sh
# Escolher opção 2

# Opção 2: Direto
pip install pandas numpy matplotlib seaborn scikit-learn
python3 main.py
```

### Saída Esperada

- 📈 **Gráficos de distribuição** dos dados
- 📊 **Matriz de correlação** visual
- 🎯 **Curvas ROC** com AUC
- 📋 **Matriz de confusão** colorida
- 📉 **Evolução da função de custo**
- 📊 **Métricas completas** para o artigo

---

## 🧠 Detalhes Técnicos

### Regressão Linear

- **Gradiente Descendente**: `θ = θ - α∇J(θ)`
- **Função de Custo**: MSE = `1/2m Σ(h(x) - y)²`
- **Learning Rate**: 0.01
- **Convergência**: Automática

### Regressão Logística

- **Função Sigmoid**: `σ(z) = 1/(1 + e^(-z))`
- **Log-Likelihood**: `J = -1/m Σ[y*log(h) + (1-y)*log(1-h)]`
- **Classificação**: Alta/Baixa bilheteria (threshold = mediana)

### Dataset TMDB

- **Total**: 1.938 registros válidos
- **Features**: Budget, Popularity, Runtime
- **Target**: Revenue (bilheteria)
- **Filtros**: Budget > $1M, Runtime > 60min

---

## 🛠️ Tecnologias

### Backend

- **Python 3.8+** - Linguagem principal
- **Flask** - Framework web
- **CSV nativo** - Manipulação de dados (versão web)
- **Pandas/NumPy** - Análise de dados (versão acadêmica)

### Frontend

- **HTML5 + CSS3** - Interface moderna
- **JavaScript ES6+** - Interatividade
- **Chart.js** - Gráficos interativos
- **Glassmorphism** - Design moderno

### Machine Learning

- **Implementação própria** - Sem sklearn
- **Gradiente descendente** - Otimização manual
- **Métricas calculadas** - Do zero

---

## 📊 Resultados

### Performance dos Modelos

- **Regressão Linear**: R² ≈ 0.40, RMSE ≈ $84M
- **Regressão Logística**: Acurácia ≈ 72%, Precisão ≈ 75%
- **Convergência**: ~230 iterações

### Métricas Calculadas

- **Regressão**: MSE, RMSE, MAE, R² Score
- **Classificação**: Acurácia, Precisão, Recall, F1-Score
- **Matriz de Confusão**: TP, FP, TN, FN

---

## 👨‍🎓 Informações Acadêmicas

**Instituição**: Instituto Federal do Maranhão - IFMA Campus Imperatriz  
**Curso**: Ciência da Computação  
**Disciplina**: Introdução à Inteligência Artificial  
**Professor**: Daniel Duarte Costa  
**Semestre**: 2024.2

### 📝 Requisitos Atendidos

- ✅ Implementação própria dos algoritmos (sem sklearn)
- ✅ Regressão Linear com gradiente descendente
- ✅ Regressão Logística para classificação
- ✅ Todas as métricas solicitadas
- ✅ Interface web moderna e interativa
- ✅ Visualizações e gráficos profissionais
- ✅ Documentação completa

---

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos no IFMA Campus Imperatriz.

---

**🎬 Prediga o sucesso do próximo blockbuster!** 🚀
