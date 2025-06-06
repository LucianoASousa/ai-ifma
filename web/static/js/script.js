// Funcionalidade das abas
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetTab = button.getAttribute("data-tab");

      // Remove active class from all tabs and contents
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      tabContents.forEach((content) => content.classList.remove("active"));

      // Add active class to clicked tab and corresponding content
      button.classList.add("active");
      document.getElementById(targetTab).classList.add("active");
    });
  });
}

// Função para formatar números
function formatCurrency(value) {
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatNumber(value, decimals = 2) {
  return new Intl.NumberFormat("pt-BR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

function formatScientific(value) {
  return value.toExponential(2);
}

// Função para prever bilheteria
async function predictRevenue() {
  const budget = document.getElementById("budget").value;
  const popularity = document.getElementById("popularity").value;
  const runtime = document.getElementById("runtime").value;

  // Validação básica
  if (!budget || !popularity || !runtime) {
    alert("Por favor, preencha todos os campos!");
    return;
  }

  if (budget < 1000000 || budget > 500000000) {
    alert("Orçamento deve estar entre $1M e $500M!");
    return;
  }

  if (popularity < 1 || popularity > 100) {
    alert("Popularidade deve estar entre 1 e 100!");
    return;
  }

  if (runtime < 60 || runtime > 300) {
    alert("Duração deve estar entre 60 e 300 minutos!");
    return;
  }

  // Mostrar loading
  const loadingSpinner = document.getElementById("loadingSpinner");
  const predictionResults = document.getElementById("predictionResults");

  loadingSpinner.style.display = "block";
  predictionResults.style.display = "none";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        budget: parseFloat(budget),
        popularity: parseFloat(popularity),
        runtime: parseFloat(runtime),
      }),
    });

    if (!response.ok) {
      throw new Error("Erro na requisição");
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error);
    }

    // Atualizar resultados
    document.getElementById("linearResult").textContent = formatCurrency(
      data.predicted_revenue
    );
    document.getElementById("logisticResult").textContent =
      data.predicted_class;

    // Atualizar barra de probabilidade
    const probabilityPercent = (data.probability * 100).toFixed(1);
    document.getElementById(
      "probabilityText"
    ).textContent = `${probabilityPercent}%`;
    document.getElementById(
      "probabilityFill"
    ).style.width = `${probabilityPercent}%`;

    // Esconder loading e mostrar resultados
    loadingSpinner.style.display = "none";
    predictionResults.style.display = "block";
  } catch (error) {
    console.error("Erro:", error);
    alert("Erro ao fazer predição: " + error.message);
    loadingSpinner.style.display = "none";
  }
}

// Função para executar análise completa
async function runFullAnalysis() {
  // Mostrar loading
  const analysisLoading = document.getElementById("analysisLoading");
  const analysisResults = document.getElementById("analysisResults");

  analysisLoading.style.display = "block";
  analysisResults.style.display = "none";

  try {
    const response = await fetch("/api/analyze");

    if (!response.ok) {
      throw new Error("Erro na requisição");
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error);
    }

    // Atualizar estatísticas do dataset
    document.getElementById("totalSamples").textContent =
      data.dataset_info.total_samples;
    document.getElementById("meanRevenue").textContent = formatCurrency(
      data.dataset_info.mean_revenue
    );
    document.getElementById("medianRevenue").textContent = formatCurrency(
      data.dataset_info.median_revenue
    );
    document.getElementById("maxRevenue").textContent = formatCurrency(
      data.dataset_info.max_revenue
    );

    // Atualizar métricas
    document.getElementById("mse").textContent = formatScientific(
      data.linear_regression.metrics.mse_test
    );
    document.getElementById("rmse").textContent = formatCurrency(
      data.linear_regression.metrics.rmse_test
    );
    document.getElementById("mae").textContent = formatCurrency(
      data.linear_regression.metrics.mae_test
    );
    document.getElementById("r2").textContent = formatNumber(
      data.linear_regression.metrics.r2_test,
      4
    );

    document.getElementById("accuracy").textContent =
      formatNumber(data.logistic_regression.metrics.accuracy_test * 100, 1) +
      "%";
    document.getElementById("precision").textContent =
      formatNumber(data.logistic_regression.metrics.precision_test * 100, 1) +
      "%";
    document.getElementById("recall").textContent =
      formatNumber(data.logistic_regression.metrics.recall_test * 100, 1) + "%";
    document.getElementById("f1").textContent = formatNumber(
      data.logistic_regression.metrics.f1_test,
      4
    );

    // Criar gráficos
    createCostChart(
      "linearCostChart",
      data.linear_regression.cost_history,
      "Regressão Linear"
    );
    createCostChart(
      "logisticCostChart",
      data.logistic_regression.cost_history,
      "Regressão Logística"
    );
    createScatterChart(data.scatter_data);

    // Atualizar matriz de confusão
    createConfusionMatrix(data.logistic_regression.confusion_matrix);

    // Atualizar tabela de predições
    createPredictionsTable(data.predictions_sample);

    // Esconder loading e mostrar resultados
    analysisLoading.style.display = "none";
    analysisResults.style.display = "block";
  } catch (error) {
    console.error("Erro:", error);
    alert("Erro ao executar análise: " + error.message);
    analysisLoading.style.display = "none";
  }
}

function createCostChart(canvasId, costHistory, title) {
  console.log(`Criando gráfico ${title}:`, {
    canvasId,
    costHistoryLength: costHistory.length,
    firstValues: costHistory.slice(0, 5),
    lastValues: costHistory.slice(-5),
  });

  const ctx = document.getElementById(canvasId).getContext("2d");

  if (window[canvasId + "Chart"]) {
    window[canvasId + "Chart"].destroy();
  }

  // Filtrar valores inválidos (NaN, Infinity, etc.)
  const validCostHistory = costHistory.filter(
    (cost) =>
      typeof cost === "number" && isFinite(cost) && !isNaN(cost) && cost >= 0
  );

  console.log(`Valores válidos filtrados para ${title}:`, {
    originalLength: costHistory.length,
    validLength: validCostHistory.length,
    minValue: Math.min(...validCostHistory),
    maxValue: Math.max(...validCostHistory),
  });

  if (validCostHistory.length === 0) {
    console.error(
      `Nenhum valor válido encontrado no cost_history para ${title}`
    );
    return;
  }

  // Calcular limites apropriados para o eixo Y
  const minCost = Math.min(...validCostHistory);
  const maxCost = Math.max(...validCostHistory);
  const costRange = maxCost - minCost;

  // Se o range for muito pequeno, usar valores padrão
  let yMin, yMax;
  if (costRange < 1e-10) {
    yMin = minCost - 0.1;
    yMax = maxCost + 0.1;
  } else {
    // Adicionar margem de 10% para melhor visualização
    const margin = costRange * 0.1;
    yMin = Math.max(0, minCost - margin);
    yMax = maxCost + margin;
  }

  console.log(`Configuração do eixo Y para ${title}:`, {
    minCost,
    maxCost,
    costRange,
    yMin,
    yMax,
  });

  window[canvasId + "Chart"] = new Chart(ctx, {
    type: "line",
    data: {
      labels: validCostHistory.map((_, index) => index + 1),
      datasets: [
        {
          label: "Custo",
          data: validCostHistory,
          borderColor: "#667eea",
          backgroundColor: "rgba(102, 126, 234, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: "index",
      },
      plugins: {
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: title,
          font: {
            size: 14,
            weight: "600",
          },
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              return `Iteração ${context.label}: ${context.parsed.y.toFixed(
                6
              )}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: {
            display: true,
            text: "Iterações",
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          ticks: {
            maxTicksLimit: 10,
          },
        },
        y: {
          type: "linear",
          min: yMin,
          max: yMax,
          title: {
            display: true,
            text: "Custo",
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          ticks: {
            callback: function (value) {
              return value.toFixed(4);
            },
            maxTicksLimit: 8,
          },
        },
      },
    },
  });

  console.log(`Gráfico ${title} criado com sucesso!`);
}

function createScatterChart(scatterData) {
  const ctx = document.getElementById("scatterChart").getContext("2d");

  if (window.scatterChartInstance) {
    window.scatterChartInstance.destroy();
  }

  // Preparar dados para o gráfico de dispersão
  const chartData = scatterData.real_values.map((real, index) => ({
    x: real,
    y: scatterData.predicted_values[index],
  }));

  // Calcular limites mais apropriados para os eixos
  const minVal = Math.max(0, scatterData.min_value); // Garantir que não seja negativo
  const maxVal = scatterData.max_value;

  // Adicionar margem de 10% nos eixos
  const margin = (maxVal - minVal) * 0.1;
  const axisMin = Math.max(0, minVal - margin);
  const axisMax = maxVal + margin;

  // Linha de referência perfeita (y = x)
  const perfectLine = [
    { x: axisMin, y: axisMin },
    { x: axisMax, y: axisMax },
  ];

  window.scatterChartInstance = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Predições",
          data: chartData,
          backgroundColor: "rgba(102, 126, 234, 0.6)",
          borderColor: "rgba(102, 126, 234, 1)",
          borderWidth: 1,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
        {
          label: "Predição Perfeita (y=x)",
          data: perfectLine,
          type: "line",
          borderColor: "rgba(239, 68, 68, 1)",
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
        title: {
          display: true,
          text: "Receita Real vs Receita Prevista",
          font: {
            size: 16,
            weight: "600",
          },
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              if (context.datasetIndex === 0) {
                return `Real: ${formatCurrency(
                  context.parsed.x
                )}, Previsto: ${formatCurrency(context.parsed.y)}`;
              }
              return context.dataset.label;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          min: axisMin,
          max: axisMax,
          title: {
            display: true,
            text: "Receita Real (USD)",
            font: {
              size: 12,
              weight: "600",
            },
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          ticks: {
            callback: function (value) {
              return formatCurrency(value);
            },
            maxTicksLimit: 8,
          },
        },
        y: {
          type: "linear",
          min: axisMin,
          max: axisMax,
          title: {
            display: true,
            text: "Receita Prevista (USD)",
            font: {
              size: 12,
              weight: "600",
            },
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          ticks: {
            callback: function (value) {
              return formatCurrency(value);
            },
            maxTicksLimit: 8,
          },
        },
      },
    },
  });
}

function createConfusionMatrix(cm) {
  const matrixContainer = document.getElementById("confusionMatrix");

  matrixContainer.innerHTML = `
    <div class="header"></div>
    <div class="header">Predito: Baixa</div>
    <div class="header">Predito: Alta</div>
    <div class="header">Real: Baixa</div>
    <div class="cell tn">${cm.tn}</div>
    <div class="cell fp">${cm.fp}</div>
    <div class="header">Real: Alta</div>
    <div class="cell fn">${cm.fn}</div>
    <div class="cell tp">${cm.tp}</div>
  `;
}

function createPredictionsTable(predictions) {
  const tableContainer = document.getElementById("predictionsTable");

  let tableHTML = `
    <table>
      <thead>
        <tr>
          <th>Bilheteria Real</th>
          <th>Bilheteria Predita</th>
          <th>Classe Real</th>
          <th>Classe Predita</th>
          <th>Probabilidade</th>
        </tr>
      </thead>
      <tbody>
  `;

  predictions.forEach((pred) => {
    tableHTML += `
      <tr>
        <td>${formatCurrency(pred.real_revenue)}</td>
        <td>${formatCurrency(pred.pred_revenue)}</td>
        <td>${pred.real_class}</td>
        <td>${pred.pred_class}</td>
        <td>${formatNumber(pred.probability * 100, 1)}%</td>
      </tr>
    `;
  });

  tableHTML += `
      </tbody>
    </table>
  `;

  tableContainer.innerHTML = tableHTML;
}

// Função para adicionar exemplos de filmes famosos
function addMovieExamples() {
  const examples = [
    {
      name: "Avengers: Endgame",
      budget: 356000000,
      popularity: 73.5,
      runtime: 181,
    },
    { name: "Avatar", budget: 237000000, popularity: 67.3, runtime: 162 },
    { name: "Titanic", budget: 200000000, popularity: 62.1, runtime: 194 },
    {
      name: "Star Wars: The Force Awakens",
      budget: 245000000,
      popularity: 58.9,
      runtime: 138,
    },
    { name: "Filme Indie", budget: 5000000, popularity: 15.2, runtime: 95 },
  ];

  const selectElement = document.createElement("select");
  selectElement.id = "movieExamples";
  selectElement.innerHTML = '<option value="">Selecione um exemplo...</option>';

  examples.forEach((movie) => {
    const option = document.createElement("option");
    option.value = JSON.stringify(movie);
    option.textContent = movie.name;
    selectElement.appendChild(option);
  });

  selectElement.addEventListener("change", (e) => {
    if (e.target.value) {
      const movie = JSON.parse(e.target.value);
      document.getElementById("budget").value = movie.budget;
      document.getElementById("popularity").value = movie.popularity;
      document.getElementById("runtime").value = movie.runtime;
    }
  });

  // Adicionar o select antes do primeiro form-group
  const firstFormGroup = document.querySelector(".form-group");
  const exampleContainer = document.createElement("div");
  exampleContainer.className = "form-group";
  exampleContainer.innerHTML = `
        <label for="movieExamples">
            <i class="fas fa-lightbulb"></i> Exemplos de Filmes
        </label>
    `;
  exampleContainer.appendChild(selectElement);

  firstFormGroup.parentNode.insertBefore(exampleContainer, firstFormGroup);
}

// Inicializar aplicação
document.addEventListener("DOMContentLoaded", function () {
  initTabs();
  addMovieExamples();

  // Adicionar eventos de teclado
  document.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      const activeTab = document.querySelector(".tab-content.active");
      if (activeTab && activeTab.id === "predictor") {
        predictRevenue();
      } else if (activeTab && activeTab.id === "analysis") {
        runFullAnalysis();
      }
    }
  });

  // Adicionar tooltips para inputs
  const inputs = document.querySelectorAll('input[type="number"]');
  inputs.forEach((input) => {
    input.addEventListener("focus", function () {
      this.select();
    });
  });

  // Event listeners para a aba Versões
  document.addEventListener("DOMContentLoaded", function () {
    // Executar main.py original
    const executarMainBtn = document.getElementById("executar-main-original");
    if (executarMainBtn) {
      executarMainBtn.addEventListener("click", function () {
        executarAnaliseAcademica();
      });
    }

    // Botões de informação
    const infoAcademicaBtn = document.getElementById("info-academica");
    if (infoAcademicaBtn) {
      infoAcademicaBtn.addEventListener("click", function () {
        mostrarInfoVersoes("academica");
      });
    }

    const infoWebBtn = document.getElementById("info-web");
    if (infoWebBtn) {
      infoWebBtn.addEventListener("click", function () {
        mostrarInfoVersoes("web");
      });
    }

    // Usar interface web
    const usarWebBtn = document.getElementById("usar-web");
    if (usarWebBtn) {
      usarWebBtn.addEventListener("click", function () {
        // Mudar para a aba do preditor
        showTab("predictor");
      });
    }
  });
});

// Função para mostrar notificações
function showNotification(message, type = "info") {
  const notification = document.createElement("div");
  notification.className = `notification ${type}`;
  notification.textContent = message;
  notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === "error" ? "#f56565" : "#48bb78"};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;

  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.animation = "slideOut 0.3s ease";
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 300);
  }, 3000);
}

// Adicionar animações CSS
const style = document.createElement("style");
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

function executarAnaliseAcademica() {
  const btn = document.getElementById("executar-main-original");
  const resultContainer = document.getElementById("execution-result");
  const outputDiv = document.getElementById("execution-output");

  // Mostrar loading
  btn.innerHTML = "⏳ Executando...";
  btn.disabled = true;

  resultContainer.style.display = "block";
  outputDiv.innerHTML =
    "🚀 Iniciando análise acadêmica completa...\n⏳ Isso pode levar alguns minutos...";

  fetch("/executar_main_original")
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        outputDiv.innerHTML = `✅ ${data.message}\n\n📊 Saída da execução:\n${data.output}\n\n💡 ${data.info}`;
      } else {
        outputDiv.innerHTML = `❌ Erro: ${data.error}\n\n📝 Detalhes:\n${data.message}`;
      }
    })
    .catch((error) => {
      outputDiv.innerHTML = `❌ Erro de conexão: ${error.message}\n\n💡 Tente executar manualmente:\ncd ../tmdb-box-office-prediction\npython3 main.py`;
    })
    .finally(() => {
      btn.innerHTML = "🚀 Executar Análise Acadêmica";
      btn.disabled = false;
    });
}

function mostrarInfoVersoes(tipo) {
  fetch("/info_versoes")
    .then((response) => response.json())
    .then((data) => {
      const info =
        tipo === "academica" ? data.versao_academica : data.versao_web;

      const modal = document.createElement("div");
      modal.className = "modal-overlay";
      modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>${
                          tipo === "academica"
                            ? "📊 Versão Acadêmica"
                            : "🌐 Versão Web"
                        }</h3>
                        <button class="modal-close">&times;</button>
                    </div>
                    <div class="modal-body">
                        <p><strong>Arquivo:</strong> <code>${
                          info.arquivo
                        }</code></p>
                        <p><strong>Descrição:</strong> ${info.descricao}</p>
                        
                        <h4>✨ Recursos:</h4>
                        <ul>
                            ${info.recursos
                              .map((recurso) => `<li>${recurso}</li>`)
                              .join("")}
                        </ul>
                        
                        <h4>🎯 Ideal para:</h4>
                        <ul>
                            ${info.ideal_para
                              .map((uso) => `<li>${uso}</li>`)
                              .join("")}
                        </ul>
                    </div>
                </div>
            `;

      document.body.appendChild(modal);

      // Fechar modal
      modal.querySelector(".modal-close").addEventListener("click", () => {
        document.body.removeChild(modal);
      });

      modal.addEventListener("click", (e) => {
        if (e.target === modal) {
          document.body.removeChild(modal);
        }
      });
    })
    .catch((error) => {
      alert("Erro ao carregar informações: " + error.message);
    });
}
