"""
TMDB Box Office Predictor - Projeto Consolidado
Instituto Federal do Maranhão - IFMA Campus Imperatriz
Disciplina: Introdução à Inteligência Artificial
Professor: Daniel Duarte Costa

🎬 Predição de Bilheteria usando Machine Learning
"""

from flask import Flask, render_template, request, jsonify
import json
import csv
import random
import math
import os
import subprocess
from io import StringIO

# Importar as classes do projeto principal
from main_simple import RegressaoLinear, RegressaoLogistica, Metricas, PreProcessamento

# Configurar Flask
app = Flask(__name__, 
           template_folder='web/templates',
           static_folder='web/static')

# Dados globais para cache
cached_data = None
cached_results = None

def carregar_dados_completos(max_rows=None):
    """Carrega dados para o frontend"""
    global cached_data
    
    if cached_data is not None:
        return cached_data
    
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'train.csv')
        with open(data_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            dados = []
            
            for i, row in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                
                try:
                    budget = float(row['budget']) if row['budget'] else 0
                    popularity = float(row['popularity']) if row['popularity'] else 0
                    runtime = float(row['runtime']) if row['runtime'] else 0
                    revenue = float(row['revenue']) if row['revenue'] else 0
                    
                    # Filtrar dados válidos
                    if budget > 1000000 and popularity > 1 and runtime > 60 and revenue > 0:
                        dados.append({
                            'budget': budget,
                            'popularity': popularity,
                            'runtime': runtime,
                            'revenue': revenue
                        })
                        
                except (ValueError, KeyError):
                    continue
            
            print(f"✅ Carregados {len(dados)} registros válidos do dataset TMDB")
            cached_data = dados
            return dados
            
    except FileNotFoundError:
        print("❌ Arquivo train.csv não encontrado na pasta data/")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return []

def treinar_modelos():
    """Treina os modelos de ML"""
    global cached_results
    
    if cached_results is not None:
        return cached_results
    
    dados = carregar_dados_completos()
    if not dados:
        return None
    
    print("🧠 Treinando modelos de Machine Learning...")
    
    # Preparar dados
    preprocessamento = PreProcessamento()
    
    # Extrair features e targets
    X = [[d['budget'], d['popularity'], d['runtime']] for d in dados]
    y_regression = [d['revenue'] for d in dados]
    
    # Criar classes para regressão logística
    mediana_revenue = sorted(y_regression)[len(y_regression)//2]
    y_classification = [1 if revenue > mediana_revenue else 0 for revenue in y_regression]
    
    # Normalizar apenas as features (X)
    X_norm = preprocessamento.normalizar(X)
    
    # Dividir dados (usando y_regression sem normalização)
    X_train, X_test, y_reg_train, y_reg_test = preprocessamento.dividir_dados(X_norm, y_regression, test_size=0.3)
    _, _, y_class_train, y_class_test = preprocessamento.dividir_dados(X_norm, y_classification, test_size=0.3)
    
    # Treinar Regressão Linear
    print("📈 Treinando Regressão Linear...")
    reg_linear = RegressaoLinear(learning_rate=0.01, max_iterations=1000)
    reg_linear.fit(X_train, y_reg_train)
    pred_reg_test = reg_linear.predict(X_test)
    
    # Treinar Regressão Logística
    print("📊 Treinando Regressão Logística...")
    reg_logistica = RegressaoLogistica(learning_rate=0.01, max_iterations=1000)
    reg_logistica.fit(X_train, y_class_train)
    pred_class_test = reg_logistica.predict(X_test)
    prob_test = reg_logistica.predict_proba(X_test)
    
    # Calcular métricas
    metricas = Metricas()
    
    # Métricas de Regressão Linear
    mse_test = metricas.mse(y_reg_test, pred_reg_test)
    rmse_test = metricas.rmse(y_reg_test, pred_reg_test)
    mae_test = metricas.mae(y_reg_test, pred_reg_test)
    r2_test = metricas.r2_score(y_reg_test, pred_reg_test)
    
    # Métricas de Regressão Logística
    accuracy_test = metricas.accuracy(y_class_test, pred_class_test)
    precision_test = metricas.precision(y_class_test, pred_class_test)
    recall_test = metricas.recall(y_class_test, pred_class_test)
    f1_test = metricas.f1_score(y_class_test, pred_class_test)
    
    # Matriz de confusão
    tp = sum(1 for true, pred in zip(y_class_test, pred_class_test) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_class_test, pred_class_test) if true == 0 and pred == 1)
    tn = sum(1 for true, pred in zip(y_class_test, pred_class_test) if true == 0 and pred == 0)
    fn = sum(1 for true, pred in zip(y_class_test, pred_class_test) if true == 1 and pred == 0)
    cm = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    
    # Amostras de predições
    predictions_sample = []
    for i in range(min(10, len(X_test))):
        predictions_sample.append({
            'real_revenue': y_reg_test[i],
            'pred_revenue': pred_reg_test[i],
            'real_class': 'Alta' if y_class_test[i] == 1 else 'Baixa',
            'pred_class': 'Alta' if pred_class_test[i] == 1 else 'Baixa',
            'probability': prob_test[i]
        })
    
    cached_results = {
        'models': {
            'linear': reg_linear,
            'logistic': reg_logistica,
            'preprocessor': preprocessamento,
            'median_revenue': mediana_revenue
        },
        'linear_regression': {
            'metrics': {
                'mse_test': mse_test,
                'rmse_test': rmse_test,
                'mae_test': mae_test,
                'r2_test': r2_test
            },
            'cost_history': reg_linear.cost_history
        },
        'logistic_regression': {
            'metrics': {
                'accuracy_test': accuracy_test,
                'precision_test': precision_test,
                'recall_test': recall_test,
                'f1_test': f1_test
            },
            'cost_history': reg_logistica.cost_history,
            'confusion_matrix': cm
        },
        'predictions_sample': predictions_sample,
        'dataset_info': {
            'total_samples': len(dados),
            'mean_revenue': sum(y_regression) / len(y_regression),
            'median_revenue': mediana_revenue
        }
    }
    
    print("✅ Treinamento concluído!")
    return cached_results

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API para fazer predições individuais"""
    try:
        data = request.get_json()
        
        budget = float(data.get('budget', 0))
        popularity = float(data.get('popularity', 0))
        runtime = float(data.get('runtime', 0))
        
        if budget <= 0 or popularity <= 0 or runtime <= 0:
            return jsonify({'error': 'Valores inválidos fornecidos'}), 400
        
        results = treinar_modelos()
        if not results:
            return jsonify({'error': 'Erro ao treinar modelos'}), 500
        
        models = results['models']
        
        # Preparar dados para predição
        X_pred = [[budget, popularity, runtime]]
        X_pred_norm = models['preprocessor'].normalizar(X_pred)
        
        # Fazer predições
        pred_revenue = models['linear'].predict(X_pred_norm)[0]
        pred_class = models['logistic'].predict(X_pred_norm)[0]
        pred_prob = models['logistic'].predict_proba(X_pred_norm)[0]
        
        return jsonify({
            'predicted_revenue': pred_revenue,
            'predicted_class': 'Alta' if pred_class == 1 else 'Baixa',
            'probability': pred_prob,
            'input_data': {
                'budget': budget,
                'popularity': popularity,
                'runtime': runtime
            }
        })
        
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/api/analyze')
def analyze():
    """API para análise completa dos dados"""
    try:
        results = treinar_modelos()
        if not results:
            return jsonify({'error': 'Erro ao treinar modelos'}), 500
        
        response_data = {
            'linear_regression': results['linear_regression'],
            'logistic_regression': results['logistic_regression'],
            'predictions_sample': results['predictions_sample'],
            'dataset_info': results['dataset_info']
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Erro na análise: {e}")
        return jsonify({'error': f'Erro interno: {str(e)}'}), 500

@app.route('/api/status')
def status():
    """Status da aplicação"""
    dados = carregar_dados_completos()
    return jsonify({
        'status': 'online',
        'dados_carregados': len(dados) if dados else 0,
        'modelo_treinado': cached_results is not None
    })

@app.route('/executar_main_original')
def executar_main_original():
    """Executa o main.py original com visualizações completas"""
    try:
        main_path = os.path.join(os.path.dirname(__file__), 'main.py')
        if not os.path.exists(main_path):
            return jsonify({
                'success': False,
                'error': 'Arquivo main.py não encontrado'
            })
        
        result = subprocess.run([
            'python3', main_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'Análise acadêmica executada com sucesso!',
                'output': result.stdout
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Erro na execução',
                'message': result.stderr
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Erro interno',
            'message': str(e)
        })

if __name__ == '__main__':
    print("🎬 TMDB Box Office Predictor")
    print("=" * 40)
    print("Instituto Federal do Maranhão - IFMA")
    print("Disciplina: Introdução à IA")
    print("=" * 40)
    print()
    print("Iniciando aplicação Flask...")
    print("Carregando dados...")
    carregar_dados_completos()
    print()
    print("🌐 Acesse: http://localhost:5000")
    print("🎯 Interface web com predições interativas")
    print("📊 Análise completa dos algoritmos ML")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000) 