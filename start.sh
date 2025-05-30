#!/bin/bash

# TMDB Box Office Predictor - Script de Inicialização
# Instituto Federal do Maranhão - IFMA Campus Imperatriz

echo "🎬 TMDB Box Office Predictor"
echo "============================"
echo "IFMA Campus Imperatriz - Introdução à IA"
echo ""

# Verificar se está no diretório correto
if [ ! -f "app.py" ]; then
    echo "❌ Execute este script no diretório do projeto"
    exit 1
fi

echo "🎯 Escolha o modo de execução:"
echo ""
echo "1️⃣  Interface Web Interativa"
echo "   🌐 Demonstração no navegador"
echo "   🔮 Predições em tempo real"
echo ""
echo "2️⃣  Análise Acadêmica Completa"
echo "   📊 Gráficos e visualizações"
echo "   🎓 Para artigo científico"
echo ""
echo "3️⃣  Ambos (Recomendado)"
echo "   📋 Análise + Interface web"
echo ""

read -p "🤔 Escolha uma opção (1/2/3): " opcao

case $opcao in
    1)
        echo ""
        echo "🌐 Iniciando Interface Web..."
        echo "=========================="
        
        # Instalar dependências web
        echo "📦 Instalando Flask..."
        pip install Flask --break-system-packages 2>/dev/null || pip install Flask
        
        # Verificar datasets
        if [ ! -f "data/train.csv" ]; then
            echo "❌ Dataset train.csv não encontrado em data/"
            exit 1
        fi
        
        echo "🚀 Iniciando servidor..."
        echo "🌐 Acesse: http://localhost:5000"
        python3 app.py
        ;;
        
    2)
        echo ""
        echo "📊 Executando Análise Acadêmica..."
        echo "================================="
        
        # Instalar dependências completas
        echo "📦 Instalando dependências para visualizações..."
        pip install pandas numpy matplotlib seaborn scikit-learn --break-system-packages 2>/dev/null || pip install pandas numpy matplotlib seaborn scikit-learn
        
        echo "🚀 Executando main.py..."
        python3 main.py
        ;;
        
    3)
        echo ""
        echo "📋 Executando Análise + Interface Web..."
        echo "======================================="
        
        # Instalar todas as dependências
        echo "📦 Instalando todas as dependências..."
        pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt
        
        # Executar análise acadêmica primeiro
        echo ""
        echo "1️⃣ Executando análise acadêmica..."
        python3 main.py
        
        # Executar interface web
        echo ""
        echo "2️⃣ Iniciando interface web..."
        echo "🌐 Acesse: http://localhost:5000"
        python3 app.py
        ;;
        
    *)
        echo ""
        echo "❌ Opção inválida. Execute novamente e escolha 1, 2 ou 3."
        exit 1
        ;;
esac 