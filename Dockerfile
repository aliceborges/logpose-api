# ==============================================================================
# Dockerfile - Codalices Forecaster
# ==============================================================================
# Imagem base oficial do Python otimizada para produção
FROM python:3.10-slim

# Evita que o Python escreva ficheiros .pyc no disco
ENV PYTHONDONTWRITEBYTECODE=1

# Garante que os logs do ecossistema (FastAPI/Uvicorn) sejam transmitidos imediatamente
ENV PYTHONUNBUFFERED=1

# Define o diretório de trabalho interno do contêiner
WORKDIR /app

# Instala dependências essenciais do sistema operacional para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia o ficheiro de dependências primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Atualiza o gerenciador de pacotes pip e instala os pré-requisitos de ML e API
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Cria a estrutura de pastas necessária para a inferência estável
RUN mkdir -p models src

# Copia os módulos de código e os artefactos pré-treinados do modelo LSTM
COPY src/ ./src/
COPY models/ ./models/

# Expõe a porta interna do FastAPI para mapeamento de rede
EXPOSE 8000

# Executa o servidor Uvicorn ligando a API a todas as interfaces de rede do contêiner
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]