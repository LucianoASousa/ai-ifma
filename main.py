"""
Instituto Federal de Ciência e Tecnologia do Maranhão – IFMA – Campus Imperatriz
Curso de Ciência da Computação
Disciplina: Introdução a Inteligência Artificial
Professor: Daniel Duarte Costa

Projeto: Predição de Bilheteria de Filmes TMDB
Implementação de Regressão Linear e Logística com Gradiente Descendente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Configuração para melhor visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
        self.bias = None
        self.cost_history = []
        
    def _add_intercept(self, X):
        """Adiciona coluna de intercepto (bias)"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _cost_function(self, h, y):
        """Calcula função de custo (MSE) com normalização para estabilidade"""
        errors = h - y
        
        # Normalizar erros pela magnitude dos valores para evitar overflow
        y_std = np.std(y) if len(y) > 1 else 1
        if y_std == 0:
            y_std = 1
        
        # MSE normalizado
        normalized_errors = errors / (y_std + 1e-8)
        return (1 / (2 * len(y))) * np.sum(normalized_errors ** 2)
    
    def fit(self, X, y):
        """Treina o modelo usando gradiente descendente"""
        # Adiciona intercepto
        X = self._add_intercept(X)
        
        # Inicializa pesos aleatoriamente
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        # Gradiente descendente
        for i in range(self.max_iterations):
            # Predição
            h = X.dot(self.weights)
            
            # Custo
            cost = self._cost_function(h, y)
            self.cost_history.append(cost)
            
            # Gradiente
            gradient = X.T.dot(h - y) / len(y)
            
            # Atualiza pesos
            self.weights -= self.learning_rate * gradient
            
            # Verifica convergência
            if len(self.cost_history) > 1:
                if abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                    print(f"Convergência alcançada na iteração {i}")
                    break
    
    def predict(self, X):
        """Faz predições"""
        X = self._add_intercept(X)
        return X.dot(self.weights)
    
    def plot_cost_history(self):
        """Plota histórico da função de custo"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Evolução da Função de Custo - Regressão Linear')
        plt.xlabel('Iterações')
        plt.ylabel('Custo (MSE)')
        plt.grid(True)
        plt.show()

class RegressaoLogistica:
  
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.cost_history = []
    
    def _add_intercept(self, X):
        """Adiciona coluna de intercepto (bias)"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _sigmoid(self, z):
        """Função sigmoid"""
        # Evita overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _cost_function(self, h, y):
        """Calcula função de custo (log-likelihood)"""
        # Evita log(0)
        h = np.clip(h, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    def fit(self, X, y):
        """Treina o modelo usando gradiente descendente"""
        # Adiciona intercepto
        X = self._add_intercept(X)
        
        # Inicializa pesos aleatoriamente
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        # Gradiente descendente
        for i in range(self.max_iterations):
            # Predição
            z = X.dot(self.weights)
            h = self._sigmoid(z)
            
            # Custo
            cost = self._cost_function(h, y)
            self.cost_history.append(cost)
            
            # Gradiente
            gradient = X.T.dot(h - y) / len(y)
            
            # Atualiza pesos
            self.weights -= self.learning_rate * gradient
            
            # Verifica convergência
            if len(self.cost_history) > 1:
                if abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                    print(f"Convergência alcançada na iteração {i}")
                    break
    
    def predict_proba(self, X):
        """Retorna probabilidades"""
        X = self._add_intercept(X)
        z = X.dot(self.weights)
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Faz predições binárias"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def plot_cost_history(self):
        """Plota histórico da função de custo"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Evolução da Função de Custo - Regressão Logística')
        plt.xlabel('Iterações')
        plt.ylabel('Custo (Log-Likelihood)')
        plt.grid(True)
        plt.show()

class Metricas:
    """Classe para cálculo das métricas de avaliação"""
    
    @staticmethod
    def mse(y_true, y_pred):
        """Erro Quadrático Médio"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Raiz do Erro Quadrático Médio"""
        return np.sqrt(Metricas.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Erro Absoluto Médio"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """Coeficiente de Determinação R²"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Acurácia"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        """Precisão"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def recall(y_true, y_pred):
        """Revocação/Sensibilidade"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
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
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    @staticmethod
    def min_max_scale(X):
        """Normalização Min-Max (valores entre 0 e 1)"""
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    
    @staticmethod
    def dividir_dados(X, y, test_size=0.3, random_state=42):
        """Divide dados em treino e teste (70% treino, 30% teste)"""
        np.random.seed(random_state)
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        test_size = int(n_samples * test_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def analise_exploratoria(df):
    """Realiza análise exploratória inicial dos dados"""
    print("=" * 80)
    print("ANÁLISE EXPLORATÓRIA INICIAL")
    print("=" * 80)
    
    # Informações básicas
    print("\n1. INFORMAÇÕES BÁSICAS:")
    print(f"Forma do dataset: {df.shape}")
    print(f"\nTipos de dados:")
    print(df.dtypes)
    
    # Primeiras linhas
    print("\n2. PRIMEIRAS 5 LINHAS:")
    print(df.head())
    
    # Estatísticas descritivas
    print("\n3. ESTATÍSTICAS DESCRITIVAS:")
    print(df.describe())
    
    # Valores ausentes
    print("\n4. VALORES AUSENTES:")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valores Ausentes': missing_values,
        'Porcentagem': missing_percent
    })
    print(missing_df[missing_df['Valores Ausentes'] > 0])
    
    # Valores únicos
    print("\n5. VALORES ÚNICOS POR COLUNA:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} valores únicos")
    
    return missing_df

def visualizar_dados(df, target_column):
    """Cria visualizações dos dados"""
    print("\n6. VISUALIZAÇÕES:")
    
    # Configurar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribuição da variável alvo
    axes[0, 0].hist(df[target_column], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title(f'Distribuição de {target_column}')
    axes[0, 0].set_xlabel(target_column)
    axes[0, 0].set_ylabel('Frequência')
    
    # Boxplot da variável alvo
    axes[0, 1].boxplot(df[target_column].dropna())
    axes[0, 1].set_title(f'Boxplot de {target_column}')
    axes[0, 1].set_ylabel(target_column)
    
    # Matriz de correlação (apenas variáveis numéricas)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 0].set_title('Matriz de Correlação')
    axes[1, 0].set_xticks(range(len(numeric_cols)))
    axes[1, 0].set_yticks(range(len(numeric_cols)))
    axes[1, 0].set_xticklabels(numeric_cols, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(numeric_cols)
    plt.colorbar(im, ax=axes[1, 0])
    
    # Gráfico de dispersão (se houver variáveis numéricas suficientes)
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        axes[1, 1].scatter(df[col1], df[col2], alpha=0.5)
        axes[1, 1].set_xlabel(col1)
        axes[1, 1].set_ylabel(col2)
        axes[1, 1].set_title(f'Dispersão: {col1} vs {col2}')
    
    plt.tight_layout()
    plt.show()

def main():
    """Função principal do projeto"""
    print("PROJETO DE PREDIÇÃO DE BILHETERIA - TMDB")
    print("Instituto Federal do Maranhão - IFMA")
    print("Disciplina: Introdução à Inteligência Artificial")
    print("=" * 80)
    
    # 1. Carregar dados
    print("\nCarregando dados...")
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
        print(f"Dados de treino carregados: {train_df.shape}")
        print(f"Dados de teste carregados: {test_df.shape}")
    except FileNotFoundError:
        print("Erro: Arquivos de dados não encontrados!")
        return
    
    # 2. Análise exploratória
    missing_info = analise_exploratoria(train_df)
    
    # Identificar variável alvo (assumindo que é 'revenue' baseado no contexto TMDB)
    target_columns = ['revenue', 'box_office', 'gross', 'target']
    target_col = None
    
    for col in target_columns:
        if col in train_df.columns:
            target_col = col
            break
    
    if target_col is None:
        print("Variável alvo não identificada automaticamente.")
        print("Colunas disponíveis:", list(train_df.columns))
        target_col = input("Digite o nome da variável alvo: ")
    
    print(f"\nVariável alvo identificada: {target_col}")
    
    # 3. Visualizações
    visualizar_dados(train_df, target_col)
    
    # 4. Pré-processamento
    print("\n" + "="*80)
    print("PRÉ-PROCESSAMENTO DOS DADOS")
    print("="*80)
    
    # Selecionar apenas colunas numéricas para simplificar
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    print(f"Colunas numéricas selecionadas: {numeric_cols}")
    
    # Tratar valores ausentes (substituir pela mediana)
    train_processed = train_df[numeric_cols + [target_col]].copy()
    for col in numeric_cols:
        median_val = train_processed[col].median()
        train_processed[col].fillna(median_val, inplace=True)
    
    # Remover outliers extremos (usando IQR)
    for col in numeric_cols:
        Q1 = train_processed[col].quantile(0.25)
        Q3 = train_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        train_processed = train_processed[
            (train_processed[col] >= lower_bound) & 
            (train_processed[col] <= upper_bound)
        ]
    
    print(f"Dados após remoção de outliers: {train_processed.shape}")
    
    # Preparar dados para modelagem
    X = train_processed[numeric_cols].values
    y = train_processed[target_col].values
    
    # Remover linhas com NaN na variável alvo
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    # Normalização
    X_normalized = PreProcessamento.normalizar(X)
    
    # Dividir dados (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = PreProcessamento.dividir_dados(
        X_normalized, y, test_size=0.3, random_state=42
    )
    
    print(f"Dados de treino: {X_train.shape}")
    print(f"Dados de teste: {X_test.shape}")
    
    # 5. REGRESSÃO LINEAR
    print("\n" + "="*80)
    print("REGRESSÃO LINEAR")
    print("="*80)
    
    # Treinar modelo
    reg_linear = RegressaoLinear(learning_rate=0.01, max_iterations=1000)
    reg_linear.fit(X_train, y_train)
    
    # Predições
    y_pred_train = reg_linear.predict(X_train)
    y_pred_test = reg_linear.predict(X_test)
    
    # Métricas
    print("\nMÉTRICAS DE REGRESSÃO LINEAR:")
    print(f"MSE (Treino): {Metricas.mse(y_train, y_pred_train):.2f}")
    print(f"MSE (Teste): {Metricas.mse(y_test, y_pred_test):.2f}")
    print(f"RMSE (Treino): {Metricas.rmse(y_train, y_pred_train):.2f}")
    print(f"RMSE (Teste): {Metricas.rmse(y_test, y_pred_test):.2f}")
    print(f"MAE (Treino): {Metricas.mae(y_train, y_pred_train):.2f}")
    print(f"MAE (Teste): {Metricas.mae(y_test, y_pred_test):.2f}")
    print(f"R² (Treino): {Metricas.r2_score(y_train, y_pred_train):.4f}")
    print(f"R² (Teste): {Metricas.r2_score(y_test, y_pred_test):.4f}")
    
    # Plotar evolução do custo
    reg_linear.plot_cost_history()
    
    # Gráfico de dispersão: Receita Real vs Prevista
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue', s=50, edgecolors='black', linewidth=0.5)
    
    # Linha de referência perfeita (y = x)
    min_val = min(min(y_test), min(y_pred_test))
    max_val = max(max(y_test), max(y_pred_test))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predição Perfeita (y=x)')
    
    # Formatação do gráfico
    plt.xlabel('Receita Real', fontsize=12)
    plt.ylabel('Receita Prevista', fontsize=12)
    plt.title('Gráfico de Dispersão: Receita Real vs Receita Prevista\nRegressão Linear', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar informações estatísticas no gráfico
    r2_test = Metricas.r2_score(y_test, y_pred_test)
    rmse_test = Metricas.rmse(y_test, y_pred_test)
    plt.text(0.05, 0.95, f'R² = {r2_test:.4f}\nRMSE = {rmse_test:.2e}', 
             transform=plt.gca().transAxes, fontsize=11, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 6. REGRESSÃO LOGÍSTICA (transformar em problema de classificação)
    print("\n" + "="*80)
    print("REGRESSÃO LOGÍSTICA")
    print("="*80)
    
    # Transformar em problema de classificação binária
    # (ex: alta bilheteria vs baixa bilheteria)
    threshold_revenue = np.median(y)
    y_binary = (y > threshold_revenue).astype(int)
    y_train_binary = (y_train > threshold_revenue).astype(int)
    y_test_binary = (y_test > threshold_revenue).astype(int)
    
    print(f"Limiar para classificação: {threshold_revenue:.2f}")
    print(f"Distribuição das classes - Classe 0: {np.sum(y_binary == 0)}, Classe 1: {np.sum(y_binary == 1)}")
    
    # Treinar modelo
    reg_logistica = RegressaoLogistica(learning_rate=0.1, max_iterations=1000)
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
    
    # Matriz de confusão
    cm = confusion_matrix(y_test_binary, y_pred_test_binary)
    print(f"\nMatriz de Confusão:")
    print(cm)
    
    # Plotar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Regressão Logística')
    plt.ylabel('Valores Reais')
    plt.xlabel('Predições')
    plt.show()
    
    # Curva ROC e AUC
    fpr, tpr, _ = roc_curve(y_test_binary, y_proba_test)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC - Regressão Logística')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # Plotar evolução do custo
    reg_logistica.plot_cost_history()
    
    print("\n" + "="*80)
    print("ANÁLISE CONCLUÍDA!")
    print("="*80)
    print("Relatório completo gerado com todas as métricas solicitadas.")
    print("Algoritmos implementados sem uso de bibliotecas sklearn.")

if __name__ == "__main__":
    main() 