# 🎬 TMDB Box Office Predictor

## Predição de Bilheteria usando Machine Learning

**Instituto Federal do Maranhão - IFMA Campus Imperatriz**  
**Disciplina:** Introdução à Inteligência Artificial  
**Professor:** Daniel Duarte Costa

---

## 📋 Agenda da Apresentação

1. **Visão Geral do Projeto**
2. **Objetivos e Metodologia**
3. **Arquitetura e Estrutura**
4. **Algoritmos Implementados**
5. **Interface Web Interativa**
6. **Resultados e Métricas**
7. **Demonstração Prática**
8. **Conclusões e Aprendizados**

---

## 🎯 Visão Geral do Projeto

### **O que é?**

- Sistema de **predição de bilheteria de filmes** usando Machine Learning
- Implementação **do zero** (sem sklearn) de algoritmos fundamentais
- Dataset real do **TMDB** com 1.938 filmes válidos

### **Diferenciais**

- ✅ **Implementação própria** de todos os algoritmos
- ✅ **Interface web moderna** para demonstrações
- ✅ **Análise acadêmica completa** com visualizações
- ✅ **Três versões** diferentes do mesmo projeto

---

## 🎯 Objetivos do Projeto

### **Objetivo Principal**

Desenvolver um sistema completo de predição de bilheteria implementando algoritmos de ML do zero

### **Objetivos Específicos**

- 📊 Implementar **Regressão Linear** com gradiente descendente
- 🎯 Implementar **Regressão Logística** para classificação
- 📈 Calcular **métricas completas** manualmente
- 🌐 Criar **interface web** para demonstrações
- 📋 Gerar **análise acadêmica** com visualizações

---

## 🏗️ Arquitetura do Sistema

```
📁 tmdb-ml-project/
├── 🌐 app.py              # Interface Web Flask (323 linhas)
├── 📊 main.py             # Análise Acadêmica (527 linhas)
├── 🔧 main_simple.py      # Classes ML do Zero (614 linhas)
├── 🚀 start.sh            # Script de Inicialização
├── 📁 data/              # Datasets TMDB (67MB)
├── 📁 web/               # Interface Moderna
│   ├── templates/        # HTML (701 linhas)
│   └── static/          # CSS + JavaScript (1635 linhas)
└── 📁 docs/             # Documentação
```

### **Três Versões Complementares**

1. **Web** - Demonstração interativa
2. **Acadêmica** - Gráficos para artigo científico
3. **Pura** - Implementação educacional

---

## 🧠 Algoritmos Implementados

### **1. Regressão Linear**

```python
# Gradiente Descendente Manual
θ = θ - α∇J(θ)

# Função de Custo MSE Normalizada
J(θ) = 1/2m Σ(h(x) - y)² / σ_y
```

**Características:**

- Learning Rate: 0.01
- Convergência automática
- Normalização para estabilidade

---

### **2. Regressão Logística**

```python
# Função Sigmoid
σ(z) = 1/(1 + e^(-z))

# Log-Likelihood
J = -1/m Σ[y*log(h) + (1-y)*log(1-h)]
```

**Características:**

- Classificação binária (Alta/Baixa bilheteria)
- Threshold baseado na mediana
- Proteção contra overflow

---

### **3. Métricas Implementadas**

#### **Regressão**

- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R²** - Coeficiente de Determinação

#### **Classificação**

- **Acurácia** - Predições corretas
- **Precisão** - Verdadeiros positivos
- **Recall** - Sensibilidade
- **F1-Score** - Média harmônica

---

## 📊 Dataset TMDB

### **Características dos Dados**

- **Total:** 1.938 registros válidos
- **Features:** Budget, Popularity, Runtime
- **Target:** Revenue (bilheteria)
- **Filtros:** Budget > $1M, Runtime > 60min

### **Estatísticas**

- **Bilheteria Média:** $95.8 milhões
- **Bilheteria Mediana:** $42.3 milhões
- **Maior Bilheteria:** $2.8 bilhões
- **Orçamento Médio:** $45.2 milhões

---

## 🌐 Interface Web Interativa

### **4 Abas Principais**

#### **🔮 Preditor**

- Entrada: Orçamento, Popularidade, Duração
- Saída: Predição de bilheteria + classificação
- Tempo real com validação

#### **📊 Análise**

- Execução completa dos algoritmos
- Gráficos de convergência
- Métricas detalhadas

#### **🎯 Versões**

- Comparação entre versões
- Execução da análise acadêmica
- Integração completa

#### **ℹ️ Sobre**

- Documentação técnica
- Detalhes da implementação

---

## 📈 Resultados Obtidos

### **Performance dos Modelos**

#### **Regressão Linear**

- **R² Score:** ~0.41 (41% da variância explicada)
- **RMSE:** ~$84 milhões
- **MAE:** ~$58 milhões
- **Convergência:** ~433 iterações

#### **Regressão Logística**

- **Acurácia:** ~72%
- **Precisão:** ~75%
- **Recall:** ~68%
- **F1-Score:** ~71%

---

## 🔧 Implementação Técnica

### **Tecnologias Utilizadas**

#### **Backend**

- **Python 3.8+** - Linguagem principal
- **Flask** - Framework web
- **CSV nativo** - Manipulação de dados

#### **Frontend**

- **HTML5 + CSS3** - Interface moderna
- **JavaScript ES6+** - Interatividade
- **Chart.js** - Gráficos dinâmicos
- **Glassmorphism** - Design moderno

#### **Machine Learning**

- **Implementação própria** - Sem bibliotecas prontas
- **Gradiente descendente** - Otimização manual
- **Métricas calculadas** - Do zero

---

## 🚀 Como Executar

### **Início Rápido**

```bash
# 1. Entrar no diretório
cd tmdb-ml-project

# 2. Executar script
./start.sh
```

### **3 Opções Disponíveis**

1. **Interface Web** - Demonstração interativa
2. **Análise Acadêmica** - Gráficos profissionais
3. **Ambos** - Experiência completa

### **Acesso**

- **URL:** http://localhost:5000
- **API:** Endpoints REST disponíveis
- **Documentação:** Integrada na interface

---

## 🎯 Demonstração Prática

### **Exemplo de Predição**

```json
{
  "input": {
    "budget": 50000000,
    "popularity": 25.5,
    "runtime": 120
  },
  "output": {
    "predicted_revenue": 95993959,
    "predicted_class": "Alta",
    "probability": 0.73
  }
}
```

### **Interpretação**

- Filme com orçamento de $50M
- Popularidade 25.5, duração 120min
- **Predição:** $96M de bilheteria (73% chance de alta bilheteria)

---

## 📊 Visualizações Geradas

### **Gráficos Disponíveis**

- 📈 **Evolução da Função de Custo**
- 📊 **Matriz de Confusão**
- 🎯 **Curva ROC com AUC**
- 📋 **Distribuição dos Dados**
- 🔗 **Matriz de Correlação**
- 📉 **Predições vs Valores Reais**

### **Análise Visual**

- Convergência estável dos algoritmos
- Boa separação entre classes
- Correlações identificadas nos dados

---

## 🔍 Principais Desafios

### **Problemas Enfrentados**

1. **Instabilidade numérica** - Valores de bilheteria muito grandes
2. **Convergência** - Função de custo oscilando
3. **Normalização** - Inconsistência entre versões

### **Soluções Implementadas**

1. **MSE normalizado** - Divisão pelo desvio padrão
2. **Proteção contra overflow** - Clipping de valores
3. **Padronização** - Mesma lógica em todas as versões

---

## 🎓 Aprendizados Obtidos

### **Técnicos**

- ✅ Implementação de algoritmos ML do zero
- ✅ Gradiente descendente na prática
- ✅ Tratamento de instabilidade numérica
- ✅ Desenvolvimento web com Flask

### **Acadêmicos**

- ✅ Metodologia científica aplicada
- ✅ Análise estatística de dados
- ✅ Visualização de resultados
- ✅ Documentação técnica completa

---

## 🔮 Possíveis Melhorias

### **Algoritmos**

- 🔄 **Regularização** (L1/L2)
- 🔄 **Outros algoritmos** (SVM, Random Forest)
- 🔄 **Feature Engineering** avançado
- 🔄 **Cross-validation**

### **Interface**

- 🔄 **Mais visualizações** interativas
- 🔄 **Comparação de modelos** lado a lado
- 🔄 **Upload de dados** personalizados
- 🔄 **Exportação de resultados**

---

## 📋 Conclusões

### **Objetivos Alcançados**

- ✅ **Implementação completa** de algoritmos ML do zero
- ✅ **Interface web funcional** para demonstrações
- ✅ **Análise acadêmica** com visualizações profissionais
- ✅ **Resultados satisfatórios** para um projeto educacional

### **Contribuições**

- 📚 **Material didático** completo para IA
- 🎯 **Exemplo prático** de implementação ML
- 🌐 **Ferramenta interativa** para aprendizado
- 📊 **Base para artigo científico**

---

## 🙏 Agradecimentos

### **Instituto Federal do Maranhão**

- Campus Imperatriz
- Curso de Ciência da Computação
- Disciplina: Introdução à IA

### **Professor Daniel Duarte Costa**

- Orientação técnica
- Metodologia aplicada
- Suporte acadêmico

### **Dataset TMDB**

- Dados reais de qualidade
- Base para aprendizado prático

---

## ❓ Perguntas e Discussão

### **Tópicos para Debate**

- 🤔 **Implementação vs Bibliotecas Prontas**
- 🤔 **Escolha de Algoritmos** para o problema
- 🤔 **Tratamento de Dados** na prática
- 🤔 **Avaliação de Performance** em ML

### **Demonstração ao Vivo**

- 🌐 Interface web funcionando
- 📊 Execução dos algoritmos
- 🎯 Predições em tempo real

---

## 📞 Contato e Recursos

### **Repositório**

- 📁 Código completo disponível
- 📋 Documentação detalhada
- 🚀 Scripts de execução

### **Recursos Adicionais**

- 📊 Template para artigo científico
- 🎯 Exemplos de uso da API
- 📈 Dados de exemplo para testes

**Obrigado pela atenção!** 🎬✨
