"""
Instituto Federal de Ciência e Tecnologia do Maranhão – IFMA – Campus Imperatriz
Curso de Ciência da Computação
Disciplina: Introdução a Inteligência Artificial
Professor: Daniel Duarte Costa

Projeto: Predição de Bilheteria de Filmes TMDB
Implementação de Regressão Linear e Logística com Gradiente Descendente
Versão Simplificada - usando apenas bibliotecas padrão Python
"""

import csv
import random
import math
import json

class RegressaoLinear:
    """
    Implementação de Regressão Linear com Gradiente Descendente
    Sem uso de bibliotecas prontas como sklearn
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.cost_history = []
        
    def _add_intercept(self, X):
        """Adiciona coluna de intercepto (bias)"""
        return [[1.0] + row for row in X]
    
    def _matrix_multiply(self, A, B):
        """Multiplicação de matrizes"""
        if isinstance(B[0], list):  # B é matriz
            result = []
            for i in range(len(A)):
                row = []
                for j in range(len(B[0])):
                    sum_val = 0
                    for k in range(len(B)):
                        sum_val += A[i][k] * B[k][j]
                    row.append(sum_val)
                result.append(row)
            return result
        else:  # B é vetor
            result = []
            for i in range(len(A)):
                sum_val = 0
                for j in range(len(A[i])):
                    sum_val += A[i][j] * B[j]
                result.append(sum_val)
            return result
    
    def _dot_product(self, A, B):
        """Produto escalar entre vetores"""
        return sum(a * b for a, b in zip(A, B))
    
    def _transpose(self, matrix):
        """Transposta de uma matriz"""
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    def _cost_function(self, predictions, y):
        """Calcula função de custo (MSE) com normalização para estabilidade"""
        n = len(y)
        errors = [(pred - actual) for pred, actual in zip(predictions, y)]
        
        # Normalizar erros pela magnitude dos valores para evitar overflow
        y_mean = sum(y) / len(y) if y else 1
        y_std = (sum((val - y_mean) ** 2 for val in y) / len(y)) ** 0.5 if y else 1
        
        # Evitar divisão por zero
        if y_std == 0:
            y_std = 1
        
        # MSE normalizado
        normalized_errors = [error / (y_std + 1e-8) for error in errors]
        error_sum = sum(err ** 2 for err in normalized_errors)
        return error_sum / (2 * n)
    
    def fit(self, X, y):
        """Treina o modelo usando gradiente descendente"""
        # Adiciona intercepto
        X_with_bias = self._add_intercept(X)
        n_samples = len(X_with_bias)
        n_features = len(X_with_bias[0])
        
        # Inicializa pesos aleatoriamente
        random.seed(42)
        self.weights = [random.gauss(0, 0.01) for _ in range(n_features)]
        
        # Gradiente descendente
        for iteration in range(self.max_iterations):
            # Predição
            predictions = []
            for row in X_with_bias:
                pred = self._dot_product(row, self.weights)
                predictions.append(pred)
            
            # Custo
            cost = self._cost_function(predictions, y)
            self.cost_history.append(cost)
            
            # Calcula gradiente
            gradients = [0.0] * n_features
            for i in range(n_samples):
                error = predictions[i] - y[i]
                for j in range(n_features):
                    gradients[j] += error * X_with_bias[i][j]
            
            # Normaliza gradiente
            gradients = [g / n_samples for g in gradients]
            
            # Atualiza pesos
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * gradients[j]
            
            # Verifica convergência
            if len(self.cost_history) > 1:
                if abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                    print(f"Convergência alcançada na iteração {iteration}")
                    break
    
    def predict(self, X):
        """Faz predições"""
        X_with_bias = self._add_intercept(X)
        predictions = []
        for row in X_with_bias:
            pred = self._dot_product(row, self.weights)
            predictions.append(pred)
        return predictions

class RegressaoLogistica:
    """
    Implementação de Regressão Logística com Gradiente Descendente
    Sem uso de bibliotecas prontas como sklearn
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.cost_history = []
    
    def _add_intercept(self, X):
        """Adiciona coluna de intercepto (bias)"""
        return [[1.0] + row for row in X]
    
    def _dot_product(self, A, B):
        """Produto escalar entre vetores"""
        return sum(a * b for a, b in zip(A, B))
    
    def _sigmoid(self, z):
        """Função sigmoid"""
        # Evita overflow
        if z > 250:
            z = 250
        elif z < -250:
            z = -250
        return 1 / (1 + math.exp(-z))
    
    def _cost_function(self, probabilities, y):
        """Calcula função de custo (log-likelihood)"""
        n = len(y)
        cost = 0
        for i in range(n):
            p = probabilities[i]
            # Evita log(0)
            if p <= 1e-15:
                p = 1e-15
            elif p >= 1 - 1e-15:
                p = 1 - 1e-15
            
            cost -= y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return cost / n
    
    def fit(self, X, y):
        """Treina o modelo usando gradiente descendente"""
        # Adiciona intercepto
        X_with_bias = self._add_intercept(X)
        n_samples = len(X_with_bias)
        n_features = len(X_with_bias[0])
        
        # Inicializa pesos aleatoriamente
        random.seed(42)
        self.weights = [random.gauss(0, 0.01) for _ in range(n_features)]
        
        # Gradiente descendente
        for iteration in range(self.max_iterations):
            # Predição
            probabilities = []
            for row in X_with_bias:
                z = self._dot_product(row, self.weights)
                p = self._sigmoid(z)
                probabilities.append(p)
            
            # Custo
            cost = self._cost_function(probabilities, y)
            self.cost_history.append(cost)
            
            # Calcula gradiente
            gradients = [0.0] * n_features
            for i in range(n_samples):
                error = probabilities[i] - y[i]
                for j in range(n_features):
                    gradients[j] += error * X_with_bias[i][j]
            
            # Normaliza gradiente
            gradients = [g / n_samples for g in gradients]
            
            # Atualiza pesos
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * gradients[j]
            
            # Verifica convergência
            if len(self.cost_history) > 1:
                if abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                    print(f"Convergência alcançada na iteração {iteration}")
                    break
    
    def predict_proba(self, X):
        """Retorna probabilidades"""
        X_with_bias = self._add_intercept(X)
        probabilities = []
        for row in X_with_bias:
            z = self._dot_product(row, self.weights)
            p = self._sigmoid(z)
            probabilities.append(p)
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """Faz predições binárias"""
        probabilities = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probabilities]

class Metricas:
    """Classe para cálculo das métricas de avaliação"""
    
    @staticmethod
    def mse(y_true, y_pred):
        """Erro Quadrático Médio"""
        n = len(y_true)
        return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / n
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Raiz do Erro Quadrático Médio"""
        return math.sqrt(Metricas.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Erro Absoluto Médio"""
        n = len(y_true)
        return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / n
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """Coeficiente de Determinação R²"""
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
        ss_tot = sum((true - mean_y) ** 2 for true in y_true)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Acurácia"""
        n = len(y_true)
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / n
    
    @staticmethod
    def precision(y_true, y_pred):
        """Precisão"""
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def recall(y_true, y_pred):
        """Revocação/Sensibilidade"""
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    @staticmethod
    def f1_score(y_true, y_pred):
        """F1-Score"""
        prec = Metricas.precision(y_true, y_pred)
        rec = Metricas.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

class PreProcessamento:
    """Classe para pré-processamento dos dados"""
    
    @staticmethod
    def normalizar(X):
        """Normalização (média=0, variância=1)"""
        n_samples = len(X)
        n_features = len(X[0])
        
        # Calcula média e desvio padrão para cada feature
        means = []
        stds = []
        
        for j in range(n_features):
            # Média
            col_values = [X[i][j] for i in range(n_samples)]
            mean = sum(col_values) / len(col_values)
            means.append(mean)
            
            # Desvio padrão
            variance = sum((x - mean) ** 2 for x in col_values) / len(col_values)
            std = math.sqrt(variance)
            if std == 0:
                std = 1  # Evita divisão por zero
            stds.append(std)
        
        # Normaliza
        X_normalized = []
        for i in range(n_samples):
            row = []
            for j in range(n_features):
                normalized_val = (X[i][j] - means[j]) / stds[j]
                row.append(normalized_val)
            X_normalized.append(row)
        
        return X_normalized
    
    @staticmethod
    def dividir_dados(X, y, test_size=0.3, random_state=42):
        """Divide dados em treino e teste (70% treino, 30% teste)"""
        random.seed(random_state)
        n_samples = len(X)
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        test_size = int(n_samples * test_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        
        return X_train, X_test, y_train, y_test

def carregar_csv(filename, max_rows=1000):
    """Carrega dados de um arquivo CSV (limitado para demonstração)"""
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                if i >= max_rows:  # Limita para demonstração
                    break
                data.append(row)
        return data
    except FileNotFoundError:
        print(f"Arquivo {filename} não encontrado!")
        return None

def analise_basica(data, target_col):
    """Análise exploratória básica"""
    print("=" * 80)
    print("ANÁLISE EXPLORATÓRIA BÁSICA")
    print("=" * 80)
    
    print(f"\n1. Número de amostras: {len(data)}")
    
    # Colunas disponíveis
    if data:
        print(f"2. Colunas disponíveis: {list(data[0].keys())}")
    
    # Valores ausentes para target
    missing_target = sum(1 for row in data if not row.get(target_col) or row[target_col] == '')
    print(f"3. Valores ausentes na variável alvo '{target_col}': {missing_target}")
    
    # Estatísticas básicas da variável alvo
    target_values = []
    for row in data:
        try:
            val = float(row.get(target_col, 0))
            if val > 0:  # Remove valores inválidos
                target_values.append(val)
        except (ValueError, TypeError):
            continue
    
    if target_values:
        target_values.sort()
        n = len(target_values)
        mean_val = sum(target_values) / n
        median_val = target_values[n // 2]
        min_val = min(target_values)
        max_val = max(target_values)
        
        print(f"4. Estatísticas da variável alvo '{target_col}':")
        print(f"   Média: {mean_val:,.2f}")
        print(f"   Mediana: {median_val:,.2f}")
        print(f"   Mínimo: {min_val:,.2f}")
        print(f"   Máximo: {max_val:,.2f}")
        print(f"   Amostras válidas: {n}")
        
        return target_values, median_val
    else:
        print("4. Não foi possível calcular estatísticas da variável alvo")
        return [], 0

def preparar_dados(data, features_numericas, target_col):
    """Prepara dados para modelagem"""
    X = []
    y = []
    
    for row in data:
        # Extrai features numéricas
        feature_row = []
        valid_row = True
        
        for feature in features_numericas:
            try:
                val = float(row.get(feature, 0))
                feature_row.append(val)
            except (ValueError, TypeError):
                valid_row = False
                break
        
        # Extrai target
        try:
            target_val = float(row.get(target_col, 0))
            if target_val <= 0:
                valid_row = False
        except (ValueError, TypeError):
            valid_row = False
        
        if valid_row and len(feature_row) == len(features_numericas):
            X.append(feature_row)
            y.append(target_val)
    
    return X, y

def main():
    """Função principal do projeto"""
    print("PROJETO DE PREDIÇÃO DE BILHETERIA - TMDB")
    print("Instituto Federal do Maranhão - IFMA")
    print("Disciplina: Introdução à Inteligência Artificial")
    print("=" * 80)
    
    # 1. Carregar dados
    print("\nCarregando dados...")
    train_data = carregar_csv('data/train.csv', max_rows=500)  # Limitado para demonstração
    
    if not train_data:
        print("Não foi possível carregar os dados. Criando dados sintéticos para demonstração...")
        # Dados sintéticos para demonstração
        train_data = []
        random.seed(42)
        for i in range(100):
            budget = random.uniform(1000000, 100000000)
            popularity = random.uniform(1, 100)
            runtime = random.uniform(80, 180)
            revenue = budget * random.uniform(0.5, 5.0) + random.uniform(-10000000, 10000000)
            
            train_data.append({
                'budget': str(budget),
                'popularity': str(popularity),
                'runtime': str(runtime),
                'revenue': str(max(0, revenue))
            })
        print("Dados sintéticos criados para demonstração.")
    
    print(f"Dados carregados: {len(train_data)} amostras")
    
    # 2. Análise exploratória
    target_col = 'revenue'
    target_values, median_revenue = analise_basica(train_data, target_col)
    
    # 3. Preparar dados
    features_numericas = ['budget', 'popularity', 'runtime']
    X, y = preparar_dados(train_data, features_numericas, target_col)
    
    if len(X) == 0:
        print("Erro: Nenhum dado válido encontrado!")
        return
    
    print(f"\nDados preparados: {len(X)} amostras com {len(X[0])} features")
    print(f"Features utilizadas: {features_numericas}")
    
    # 4. Pré-processamento
    print("\n" + "="*80)
    print("PRÉ-PROCESSAMENTO DOS DADOS")
    print("="*80)
    
    # Normalização
    X_normalized = PreProcessamento.normalizar(X)
    
    # Dividir dados (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = PreProcessamento.dividir_dados(
        X_normalized, y, test_size=0.3, random_state=42
    )
    
    print(f"Dados de treino: {len(X_train)} amostras")
    print(f"Dados de teste: {len(X_test)} amostras")
    
    # 5. REGRESSÃO LINEAR
    print("\n" + "="*80)
    print("REGRESSÃO LINEAR")
    print("="*80)
    
    # Treinar modelo
    reg_linear = RegressaoLinear(learning_rate=0.01, max_iterations=1000)  # Taxa padrão já que a função de custo foi corrigida
    print("Treinando modelo de regressão linear...")
    reg_linear.fit(X_train, y_train)
    
    # Predições
    y_pred_train = reg_linear.predict(X_train)
    y_pred_test = reg_linear.predict(X_test)
    
    # Métricas
    print("\nMÉTRICAS DE REGRESSÃO LINEAR:")
    print(f"MSE (Treino): {Metricas.mse(y_train, y_pred_train):.2e}")
    print(f"MSE (Teste): {Metricas.mse(y_test, y_pred_test):.2e}")
    print(f"RMSE (Treino): {Metricas.rmse(y_train, y_pred_train):.2e}")
    print(f"RMSE (Teste): {Metricas.rmse(y_test, y_pred_test):.2e}")
    print(f"MAE (Treino): {Metricas.mae(y_train, y_pred_train):.2e}")
    print(f"MAE (Teste): {Metricas.mae(y_test, y_pred_test):.2e}")
    print(f"R² (Treino): {Metricas.r2_score(y_train, y_pred_train):.4f}")
    print(f"R² (Teste): {Metricas.r2_score(y_test, y_pred_test):.4f}")
    
    print(f"\nIterações de treinamento: {len(reg_linear.cost_history)}")
    if len(reg_linear.cost_history) >= 10:
        print(f"Custo inicial: {reg_linear.cost_history[0]:.2e}")
        print(f"Custo final: {reg_linear.cost_history[-1]:.2e}")
    
    # Análise de Dispersão: Receita Real vs Prevista
    print("\n" + "-"*60)
    print("ANÁLISE DE DISPERSÃO: RECEITA REAL vs PREVISTA")
    print("-"*60)
    
    # Calcular correlação manual
    n = len(y_test)
    mean_real = sum(y_test) / n
    mean_pred = sum(y_pred_test) / n
    
    numerator = sum((y_test[i] - mean_real) * (y_pred_test[i] - mean_pred) for i in range(n))
    sum_sq_real = sum((y_test[i] - mean_real) ** 2 for i in range(n))
    sum_sq_pred = sum((y_pred_test[i] - mean_pred) ** 2 for i in range(n))
    
    correlation = numerator / (sum_sq_real * sum_sq_pred) ** 0.5 if sum_sq_real > 0 and sum_sq_pred > 0 else 0
    
    print(f"Correlação entre valores reais e preditos: {correlation:.4f}")
    print(f"Média dos valores reais: ${mean_real:,.0f}")
    print(f"Média dos valores preditos: ${mean_pred:,.0f}")
    
    # Mostrar algumas comparações diretas
    print(f"\nComparações diretas (primeiras 10 amostras):")
    print(f"{'Real':>15} {'Predito':>15} {'Erro':>15} {'Erro %':>10}")
    print("-" * 60)
    
    for i in range(min(10, len(y_test))):
        real = y_test[i]
        pred = y_pred_test[i]
        erro = abs(real - pred)
        erro_pct = (erro / real * 100) if real > 0 else 0
        print(f"${real:>14,.0f} ${pred:>14,.0f} ${erro:>14,.0f} {erro_pct:>9.1f}%")
    
    # Análise de qualidade das predições
    erros_absolutos = [abs(y_test[i] - y_pred_test[i]) for i in range(len(y_test))]
    erro_medio = sum(erros_absolutos) / len(erros_absolutos)
    
    # Contar predições dentro de diferentes margens de erro
    margem_10 = sum(1 for i in range(len(y_test)) if abs(y_test[i] - y_pred_test[i]) / y_test[i] <= 0.1 if y_test[i] > 0)
    margem_25 = sum(1 for i in range(len(y_test)) if abs(y_test[i] - y_pred_test[i]) / y_test[i] <= 0.25 if y_test[i] > 0)
    margem_50 = sum(1 for i in range(len(y_test)) if abs(y_test[i] - y_pred_test[i]) / y_test[i] <= 0.5 if y_test[i] > 0)
    
    print(f"\nQualidade das Predições:")
    print(f"Erro médio absoluto: ${erro_medio:,.0f}")
    print(f"Predições dentro de 10% do valor real: {margem_10}/{len(y_test)} ({margem_10/len(y_test)*100:.1f}%)")
    print(f"Predições dentro de 25% do valor real: {margem_25}/{len(y_test)} ({margem_25/len(y_test)*100:.1f}%)")
    print(f"Predições dentro de 50% do valor real: {margem_50}/{len(y_test)} ({margem_50/len(y_test)*100:.1f}%)")
    print("-"*60)
    
    # 6. REGRESSÃO LOGÍSTICA
    print("\n" + "="*80)
    print("REGRESSÃO LOGÍSTICA")
    print("="*80)
    
    # Transformar em problema de classificação binária
    threshold_revenue = median_revenue
    y_binary = [1 if val > threshold_revenue else 0 for val in y]
    y_train_binary = [1 if val > threshold_revenue else 0 for val in y_train]
    y_test_binary = [1 if val > threshold_revenue else 0 for val in y_test]
    
    print(f"Limiar para classificação: ${threshold_revenue:,.2f}")
    print(f"Distribuição das classes:")
    print(f"  Classe 0 (baixa bilheteria): {sum(1 for x in y_binary if x == 0)}")
    print(f"  Classe 1 (alta bilheteria): {sum(1 for x in y_binary if x == 1)}")
    
    # Treinar modelo
    reg_logistica = RegressaoLogistica(learning_rate=0.1, max_iterations=1000)
    print("\nTreinando modelo de regressão logística...")
    reg_logistica.fit(X_train, y_train_binary)
    
    # Predições
    y_pred_train_binary = reg_logistica.predict(X_train)
    y_pred_test_binary = reg_logistica.predict(X_test)
    y_proba_test = reg_logistica.predict_proba(X_test)
    
    # Métricas
    print("\nMÉTRICAS DE REGRESSÃO LOGÍSTICA:")
    print(f"Acurácia (Treino): {Metricas.accuracy(y_train_binary, y_pred_train_binary):.4f}")
    print(f"Acurácia (Teste): {Metricas.accuracy(y_test_binary, y_pred_test_binary):.4f}")
    print(f"Precisão (Teste): {Metricas.precision(y_test_binary, y_pred_test_binary):.4f}")
    print(f"Recall (Teste): {Metricas.recall(y_test_binary, y_pred_test_binary):.4f}")
    print(f"F1-Score (Teste): {Metricas.f1_score(y_test_binary, y_pred_test_binary):.4f}")
    
    # Matriz de confusão manual
    tp = sum(1 for true, pred in zip(y_test_binary, y_pred_test_binary) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_test_binary, y_pred_test_binary) if true == 0 and pred == 1)
    tn = sum(1 for true, pred in zip(y_test_binary, y_pred_test_binary) if true == 0 and pred == 0)
    fn = sum(1 for true, pred in zip(y_test_binary, y_pred_test_binary) if true == 1 and pred == 0)
    
    print(f"\nMatriz de Confusão:")
    print(f"                Predito")
    print(f"Actual    0     1")
    print(f"     0   {tn:3d}   {fp:3d}")
    print(f"     1   {fn:3d}   {tp:3d}")
    
    print(f"\nIterações de treinamento: {len(reg_logistica.cost_history)}")
    if len(reg_logistica.cost_history) >= 10:
        print(f"Custo inicial: {reg_logistica.cost_history[0]:.4f}")
        print(f"Custo final: {reg_logistica.cost_history[-1]:.4f}")
    
    # Exemplo de predições
    print("\n" + "="*80)
    print("EXEMPLOS DE PREDIÇÕES")
    print("="*80)
    
    print("\nRegressão Linear (primeiras 5 predições):")
    for i in range(min(5, len(y_test))):
        print(f"Real: ${y_test[i]:,.0f} | Predito: ${y_pred_test[i]:,.0f}")
    
    print("\nRegressão Logística (primeiras 5 predições):")
    for i in range(min(5, len(y_test_binary))):
        prob = y_proba_test[i]
        real_class = "Alta" if y_test_binary[i] == 1 else "Baixa"
        pred_class = "Alta" if y_pred_test_binary[i] == 1 else "Baixa"
        print(f"Real: {real_class} | Predito: {pred_class} (Prob: {prob:.3f})")
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA!")
    print("="*80)
    print("Relatório completo gerado com todas as métricas solicitadas.")
    print("Algoritmos implementados sem uso de bibliotecas sklearn.")
    print("\nObservações:")
    print("- Os algoritmos foram implementados do zero usando apenas bibliotecas padrão Python")
    print("- As funções de custo e gradiente descendente foram implementadas manualmente")
    print("- Todas as métricas foram calculadas sem bibliotecas externas")

if __name__ == "__main__":
    main() 