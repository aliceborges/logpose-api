# Log Pose API 🧭 - Predição de Ativos Financeiros com LSTM

Este repositório contém a solução completa para o **Tech Challenge da Fase 4** da Pós-Tech em **Machine Learning Engineering**. O projeto implementa uma pipeline *end-to-end* que utiliza redes neurais profundas do tipo **Long Short-Term Memory (LSTM)** para prever o preço de fechamento de ativos na bolsa de valores, disponibilizando a inferência através de uma API RESTful robusta, monitorada e containerizada.

O nome **Log Pose API** é uma referência à bússola capaz de navegar em mares imprevisíveis, refletindo o objetivo do modelo: encontrar a direção correta no oceano caótico do mercado financeiro.

---

## Visão Geral e Arquitetura

O ecossistema foi projetado com fortes princípios de Engenharia de Software, separando o ciclo de treinamento analítico da camada de inferência em produção.

* **Camada de Dados (`src/pipeline.py`)**: Extração de dados financeiros em tempo real via `yfinance`, aplicação de normalização matemática (`MinMaxScaler`) e engenharia de *features* através de janelamento temporal (look-back window de 60 dias).
* **Camada de Modelagem (`src/train.py`)**: Construção de uma rede neural profunda com múltiplas camadas LSTM e regularização por *Dropout* (20%). O script foi transformado em uma ferramenta de linha de comando (CLI), permitindo treinos dinâmicos com passagem de parâmetros (ticker e datas) direto pelo terminal. Inclui divisão estrita de dados (90% treino / 10% teste) para evitar vazamento de dados do futuro e gera métricas financeiras reais (MAE, RMSE, MAPE).
* **Camada de Produção / API (`src/main.py`)**: API de alta performance construída em **FastAPI**. Os artefatos do modelo são carregados na memória RAM durante a inicialização (via gerenciadores de contexto *Lifespan* modernos). A inferência é feita sob demanda de forma assíncrona.
* **Monitoramento de Performance**: Implementação de *Middleware* customizado que intercepta cada requisição HTTP, calcula o tempo exato de processamento em milissegundos e o injeta no cabeçalho de resposta (`X-Process-Time`), além de gerar logs de auditoria automáticos no terminal do servidor.
* **Infraestrutura (`Dockerfile`)**: Configuração baseada em ambiente *slim* otimizada para implantação em nuvem, garantindo a paridade entre ambientes e o isolamento completo de dependências.

---

## 📂 Estrutura do Projeto

    logpose-api/
    ├── models/
    │   ├── modelo_lstm.keras       # Modelo preditivo treinado e persistido pelo Keras
    │   └── scaler.pkl              # Objeto de normalização ajustado aos dados de treino
    ├── src/
    │   ├── __init__.py             
    │   ├── pipeline.py             # Pipeline de extração, janelamento e transformação
    │   ├── train.py                # CLI para treinamento, avaliação e exportação do modelo
    │   ├── main.py                 # Ponto de entrada da API REST FastAPI e monitoramento
    │   └── model.py                # Definição estrutural alternativa (PyTorch/Keras)
    ├── Dockerfile                  # Manifesto para containerização completa da API
    ├── requirements.txt            # Dependências com versões estritas fixadas
    └── README.md                   # Documentação técnica do projeto

---

## Tecnologias Utilizadas

* **Linguagem Principal:** Python 3.10
* **Deep Learning Framework:** TensorFlow 2.13.0 / Keras
* **Processamento & Modelagem Estatística:** Pandas, NumPy, Scikit-Learn
* **Framework de API:** FastAPI + Uvicorn + Pydantic (validação de dados e tipagem)
* **Fontes de Dados:** Yahoo Finance API (`yfinance`)
* **DevOps & Infraestrutura:** Docker

---

## Como Executar o Projeto Localmente

### 1. Preparar o Ambiente Virtual
Instale as dependências fixadas isolando o escopo do seu interpretador Python:
    
    python -m venv .venv
    
    # No Windows (PowerShell):
    .venv\Scripts\activate
    
    # No Linux / macOS:
    source .venv/bin/activate
    
    # Instalar pré-requisitos:
    pip install -r requirements.txt


### 2. Treinar o Modelo via CLI Dinâmica
O script de treinamento aceita argumentos dinâmicos diretamente pelo terminal. O modelo será avaliado automaticamente no final do ciclo e os arquivos serão persistidos na pasta `models/`.

    # Treinamento padrão (Ativo da Disney - DIS, de 2018 até 2024):
    python -m src.train
    
    # Treinamento parametrizado (Exemplo: Apple - AAPL, período de 2020 a 2023):
    python -m src.train --ticker AAPL --start 2020-01-01 --end 2023-12-31


### 3. Iniciar a API de Produção
Suba o servidor Uvicorn com hot-reload ativo para escutar requisições locais:
    
    uvicorn src.main:app --reload

Acesse a documentação interativa completa (Swagger UI) gerada automaticamente através do endereço: `http://127.0.0.1:8000/docs`.

---

## Como Executar via Docker (Pronto para Nuvem)

A imagem está totalmente configurada para ser enviada para plataformas de nuvem (como Render, Heroku ou instâncias AWS).

    # 1. Construir a imagem Docker localmente
    docker build -t logpose-api:latest .
    
    # 2. Executar o contêiner mapeando a porta de produção 8000
    docker run -d -p 8000:8000 --name logpose-app logpose-api:latest


---

## Utilização da API

### **POST** `/predict`
Endpoint dedicado à predição do valor de fechamento do próximo dia útil para o ativo especificado.

* **Payload de Entrada (JSON):**

    {
      "ticker": "DIS"
    }


* **Resposta de Sucesso (JSON - HTTP 200):**

    {
      "ticker": "DIS",
      "next_day_prediction_close": 106.77076721191406
    }


* **Headers de Resposta (Monitoramento Ativo):**
Ao analisar os metadados da resposta, verifique a presença do cabeçalho customizado incluído pelo nosso middleware:
`X-Process-Time: 345.12 ms`

---

## Métricas de Avaliação Obtidas
O modelo utiliza uma rigorosa divisão temporal (90% treino / 10% teste) para avaliar a assertividade em dados históricos que a rede neural nunca viu durante o ajuste de pesos. As métricas reais consolidadas na escala monetária foram:

* **MAE (Mean Absolute Error):** Margem de erro absoluto médio diário de apenas **$ 1.99**.
* **RMSE (Root Mean Square Error):** Medida de variância que penaliza desvios drásticos, consolidada em **$ 2.64**.
* **MAPE (Mean Absolute Percentage Error):** Demonstra a eficácia do modelo em termos percentuais, atingindo **1.94%** (significando uma precisão superior a 98% nas previsões de curto prazo).

---

## Autoria e Entrega
Este projeto foi desenvolvido como critério de avaliação obrigatório da **Fase 4 - Deep Learning e IA** do curso de Pós-Graduação em **Machine Learning Engineering**.

* **Autora:** Alice Borges dos Santos