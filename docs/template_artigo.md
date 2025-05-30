# Predição de Bilheteria de Filmes utilizando Regressão Linear e Logística com Gradiente Descendente: Uma Análise do Dataset TMDB

**Autores:** [Nome do(s) Aluno(s)]  
**Instituição:** Instituto Federal de Ciência e Tecnologia do Maranhão – IFMA – Campus Imperatriz  
**Curso:** Ciência da Computação  
**Disciplina:** Introdução à Inteligência Artificial  
**Professor:** Daniel Duarte Costa  

---

## Resumo

Este trabalho apresenta a implementação e avaliação de algoritmos de regressão linear e logística com gradiente descendente para predição de bilheteria de filmes utilizando o dataset TMDB Box Office Prediction. O objetivo principal foi desenvolver modelos preditivos capazes de estimar o sucesso comercial de filmes através de suas características. A metodologia envolveu análise exploratória dos dados, pré-processamento com tratamento de valores ausentes e normalização, implementação dos algoritmos sem uso de bibliotecas prontas como sklearn, e avaliação através de métricas específicas para regressão (MSE, RMSE, MAE, R²) e classificação (acurácia, precisão, recall, F1-score, AUC-ROC). Os resultados obtidos demonstraram [inserir resultados principais] indicando [inserir conclusão principal]. O trabalho evidencia a eficácia dos algoritmos implementados para o problema proposto e fornece insights valiosos sobre os fatores que influenciam o sucesso comercial de filmes.

**Palavras-chave:** Machine Learning, Regressão Linear, Regressão Logística, Gradiente Descendente, Predição de Bilheteria.

---

## 1. Introdução

A indústria cinematográfica movimenta bilhões de dólares anualmente, tornando a predição de bilheteria uma questão fundamental para estúdios, distribuidores e investidores. O sucesso comercial de um filme é influenciado por diversos fatores, incluindo orçamento de produção, gênero, elenco, diretor, época de lançamento, entre outros. Neste contexto, técnicas de aprendizado de máquina emergem como ferramentas valiosas para análise e predição desses resultados comerciais.

Este trabalho aborda o desenvolvimento e implementação de algoritmos de regressão linear e logística utilizando gradiente descendente para predição de bilheteria de filmes. O dataset utilizado é o TMDB (The Movie Database) Box Office Prediction, que contém informações detalhadas sobre filmes e suas respectivas bilheterias. A base de dados escolhida oferece uma rica variedade de atributos que podem influenciar o sucesso comercial de um filme, proporcionando um ambiente adequado para aplicação e avaliação dos algoritmos propostos.

O objetivo principal deste trabalho é implementar, do zero, algoritmos de regressão linear e logística com gradiente descendente, sem utilizar bibliotecas prontas como scikit-learn, demonstrando a compreensão dos fundamentos matemáticos e algorítmicos subjacentes. Adicionalmente, busca-se avaliar a eficácia desses modelos na predição de bilheteria e identificar os principais fatores que influenciam o sucesso comercial de filmes.

A metodologia adotada segue uma abordagem sistemática que inclui análise exploratória dos dados, pré-processamento adequado, implementação dos algoritmos, treinamento dos modelos e avaliação através de métricas apropriadas. O trabalho está estruturado nas seguintes seções: Fundamentação Teórica, que apresenta os conceitos essenciais; Materiais e Métodos, que detalha a base de dados e metodologia empregada; Resultados, que apresenta os achados obtidos; e Conclusão, que sintetiza os principais insights e limitações encontradas.

---

## 2. Fundamentação Teórica

### 2.1 Regressão Linear

A regressão linear é um dos algoritmos fundamentais de aprendizado de máquina supervisionado, utilizada para modelar a relação entre uma variável dependente contínua e uma ou mais variáveis independentes. O modelo assume uma relação linear entre as variáveis de entrada (features) e a variável alvo, sendo matematicamente expressa pela equação:

```
h_θ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θᵀx
```

Onde θ representa o vetor de parâmetros (pesos) do modelo, x é o vetor de features, e h_θ(x) é a predição do modelo. O objetivo do treinamento é encontrar os valores ótimos de θ que minimizem a função de custo, tipicamente definida como o Erro Quadrático Médio (MSE):

```
J(θ) = (1/2m) Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

A regressão linear é amplamente aplicada em problemas de predição de valores contínuos, como preços, temperaturas, e no contexto deste trabalho, bilheteria de filmes. Suas principais vantagens incluem simplicidade de implementação, interpretabilidade dos coeficientes, e eficiência computacional.

### 2.2 Regressão Logística

A regressão logística é um algoritmo de classificação que utiliza a função logística (sigmoid) para mapear qualquer valor real para um valor entre 0 e 1, tornando-a adequada para problemas de classificação binária. A função sigmoid é definida como:

```
σ(z) = 1 / (1 + e^(-z))
```

Onde z = θᵀx. A hipótese da regressão logística é expressa como:

```
h_θ(x) = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

A função de custo para regressão logística é baseada na log-verossimilhança:

```
J(θ) = -(1/m) Σ[y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]
```

A regressão logística é particularmente útil quando o problema pode ser formulado como uma questão de classificação binária, como determinar se um filme terá alta ou baixa bilheteria com base em um limiar estabelecido.

### 2.3 Gradiente Descendente

O gradiente descendente é um algoritmo de otimização iterativo utilizado para encontrar o mínimo de uma função. No contexto de aprendizado de máquina, é empregado para minimizar a função de custo e encontrar os parâmetros ótimos do modelo. O algoritmo atualiza os parâmetros na direção oposta ao gradiente da função de custo:

```
θⱼ := θⱼ - α ∂/∂θⱼ J(θ)
```

Onde α é a taxa de aprendizado (learning rate) que controla o tamanho dos passos tomados em direção ao mínimo. Para regressão linear, o gradiente é:

```
∂/∂θⱼ J(θ) = (1/m) Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
```

Para regressão logística, o gradiente tem a mesma forma, mas com h_θ(x) sendo a função sigmoid. A escolha adequada da taxa de aprendizado é crucial: valores muito altos podem causar oscilações ou divergência, enquanto valores muito baixos resultam em convergência lenta.

### 2.4 Métricas de Avaliação

Para regressão, as principais métricas incluem:
- **MSE (Mean Squared Error):** Mede a média dos quadrados dos erros
- **RMSE (Root Mean Squared Error):** Raiz quadrada do MSE, na mesma unidade da variável alvo
- **MAE (Mean Absolute Error):** Média dos valores absolutos dos erros
- **R² (Coeficiente de Determinação):** Proporção da variância explicada pelo modelo

Para classificação, as métricas fundamentais são:
- **Acurácia:** Proporção de predições corretas
- **Precisão:** Proporção de verdadeiros positivos entre as predições positivas
- **Recall (Sensibilidade):** Proporção de verdadeiros positivos identificados
- **F1-Score:** Média harmônica entre precisão e recall
- **AUC-ROC:** Área sob a curva ROC, medindo a capacidade discriminativa do modelo

---

## 3. Materiais e Métodos

### 3.1 Base de Dados

O dataset utilizado neste trabalho é o TMDB Box Office Prediction, disponibilizado na plataforma Kaggle. A base de dados contém informações sobre [inserir número] filmes, distribuídos em conjuntos de treinamento ([inserir número] amostras) e teste ([inserir número] amostras). As principais características do dataset incluem:

- **Variável alvo:** revenue (bilheteria total)
- **Atributos:** [listar principais atributos encontrados]
- **Tipos de dados:** Numéricos (budget, popularity, runtime) e categóricos (genres, production_companies)
- **Período temporal:** [inserir se disponível]

A análise exploratória inicial revelou [inserir principais achados da análise exploratória, como distribuição da variável alvo, valores ausentes, outliers].

### 3.2 Pré-processamento

O pré-processamento dos dados seguiu uma metodologia sistemática:

#### 3.2.1 Tratamento de Valores Ausentes
Os valores ausentes foram identificados e tratados através da substituição pela mediana para variáveis numéricas, escolhida por sua robustez a outliers em comparação com a média. Colunas com alta proporção de valores ausentes (>50%) foram consideradas para remoção.

#### 3.2.2 Tratamento de Outliers
Outliers extremos foram identificados utilizando o método do Intervalo Interquartílico (IQR) e removidos quando considerados atípicos (valores fora do intervalo [Q1 - 1.5×IQR, Q3 + 1.5×IQR]).

#### 3.2.3 Normalização
Devido à sensibilidade dos algoritmos à escala dos dados, foi aplicada normalização Z-score (média=0, variância=1) para garantir que todas as features contribuam igualmente para o processo de aprendizado.

#### 3.2.4 Codificação de Variáveis Categóricas
[Descrever se foi necessário e como foi implementado]

#### 3.2.5 Divisão dos Dados
Os dados foram divididos aleatoriamente em conjuntos de treinamento (70%) e teste (30%), mantendo a proporção original da variável alvo.

### 3.3 Implementação dos Algoritmos

#### 3.3.1 Regressão Linear
A implementação da regressão linear incluiu:
- Adição automática do termo de intercepto (bias)
- Função de custo baseada no MSE
- Gradiente descendente com critério de convergência
- Monitoramento da evolução do custo durante o treinamento

#### 3.3.2 Regressão Logística
Para a regressão logística, o problema de regressão foi transformado em classificação binária estabelecendo um limiar na mediana da variável alvo. A implementação incluiu:
- Função sigmoid com prevenção de overflow numérico
- Função de custo baseada na log-verossimilhança
- Gradiente descendente adaptado para classificação
- Geração de probabilidades e predições binárias

### 3.4 Ferramentas Utilizadas

O projeto foi desenvolvido em Python utilizando as seguintes bibliotecas:
- **Pandas:** Manipulação e análise de dados
- **NumPy:** Operações matemáticas e arrays
- **Matplotlib/Seaborn:** Visualização de dados
- **Implementação própria:** Algoritmos de ML (sem sklearn)

---

## 4. Resultados

### 4.1 Análise Exploratória

[Inserir principais achados da análise exploratória baseados na execução do código]

A análise exploratória revelou que o dataset possui [inserir forma], com [inserir informações sobre valores ausentes]. A distribuição da variável alvo [descrever distribuição - normal, assimétrica, etc.]. A matriz de correlação indicou [inserir principais correlações encontradas].

### 4.2 Resultados da Regressão Linear

Os resultados obtidos com o modelo de regressão linear foram:

**Métricas de Treinamento:**
- MSE: [inserir valor]
- RMSE: [inserir valor]
- MAE: [inserir valor]
- R²: [inserir valor]

**Métricas de Teste:**
- MSE: [inserir valor]
- RMSE: [inserir valor]
- MAE: [inserir valor]
- R²: [inserir valor]

[Analisar os resultados: O modelo apresentou convergência em X iterações. O valor de R² indica que o modelo explica Y% da variância dos dados. A diferença entre treino e teste sugere/não sugere overfitting.]

### 4.3 Resultados da Regressão Logística

Para a regressão logística, transformando o problema em classificação binária com limiar na mediana ([inserir valor]):

**Métricas de Treinamento:**
- Acurácia: [inserir valor]

**Métricas de Teste:**
- Acurácia: [inserir valor]
- Precisão: [inserir valor]
- Recall: [inserir valor]
- F1-Score: [inserir valor]
- AUC-ROC: [inserir valor]

[Analisar os resultados: A matriz de confusão mostrou... A curva ROC indica... O modelo apresenta boa/regular/baixa capacidade discriminativa...]

### 4.4 Análise Comparativa

[Comparar os dois modelos em termos de performance, adequação ao problema, convergência, etc.]

### 4.5 Visualizações

As visualizações geradas incluíram:
- Distribuição da variável alvo
- Boxplots para identificação de outliers
- Matriz de correlação
- Evolução da função de custo durante o treinamento
- Matriz de confusão
- Curva ROC

---

## 5. Conclusão

Este trabalho apresentou a implementação e avaliação de algoritmos de regressão linear e logística com gradiente descendente para predição de bilheteria de filmes. Os principais achados incluem:

1. **Implementação bem-sucedida:** Os algoritmos foram implementados sem uso de bibliotecas prontas, demonstrando compreensão dos fundamentos matemáticos.

2. **Performance dos modelos:** [Resumir performance obtida e adequação ao problema]

3. **Insights sobre fatores influentes:** [Mencionar se foi possível identificar quais features são mais importantes]

4. **Eficácia do gradiente descendente:** O algoritmo demonstrou convergência adequada em ambos os casos.

### 5.1 Limitações

As principais limitações identificadas foram:
- [Listar limitações encontradas, como qualidade dos dados, features disponíveis, etc.]
- Simplificação do problema para classificação binária na regressão logística
- Uso apenas de features numéricas por simplificação

### 5.2 Trabalhos Futuros

Para trabalhos futuros, sugere-se:
- Implementação de regularização (Ridge, Lasso)
- Teste com outros algoritmos (SVM, Random Forest)
- Engenharia de features mais sofisticada
- Análise de features categóricas
- Otimização de hiperparâmetros
- Validação cruzada

O trabalho demonstrou que técnicas de machine learning podem ser eficazes para predição de bilheteria, fornecendo insights valiosos para a indústria cinematográfica. A implementação própria dos algoritmos evidenciou a importância de compreender os fundamentos teóricos para aplicação eficaz de técnicas de aprendizado de máquina.

---

## Referências

[Inserir referências utilizadas, incluindo:]

1. HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. The Elements of Statistical Learning. 2ª ed. Stanford: Springer, 2009.

2. BISHOP, C. M. Pattern Recognition and Machine Learning. Cambridge: Cambridge University Press, 2006.

3. MURPHY, K. P. Machine Learning: A Probabilistic Perspective. Cambridge: MIT Press, 2012.

4. [Outras referências utilizadas para fundamentação teórica]

5. TMDB Box Office Prediction Dataset. Kaggle. Disponível em: [URL]. Acesso em: [data].

---

**Nota:** Este template deve ser preenchido com os resultados reais obtidos pela execução do código implementado. Os valores em colchetes [inserir valor] devem ser substituídos pelos dados concretos gerados pelo algoritmo. 