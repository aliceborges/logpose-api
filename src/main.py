import datetime
import time
from contextlib import asynccontextmanager

import joblib
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from tensorflow.keras.models import load_model

from src.pipeline import preparar_dados_predicao

model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de contexto para carregar e liberar recursos."""
    global model, scaler
    try:
        model = load_model('models/modelo_lstm.keras')
        scaler = joblib.load('models/scaler.pkl')
        print("Artefatos da Log Pose carregados com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar artefatos: {e}")

    yield
    print("Encerrando a API...")


app = FastAPI(
    title="Log Pose API",
    description="API para predição de ações com LSTM com monitoramento de tempo",
    lifespan=lifespan
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware que calcula o tempo de resposta e injeta no Header e no Terminal."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    process_time_ms = f"{process_time * 1000:.2f} ms"
    response.headers["X-Process-Time"] = process_time_ms

    print(f"⏱️ Tempo de Resposta ({request.url.path}): {process_time_ms}")
    return response


class PredictRequest(BaseModel):
    ticker: str = 'DIS'


@app.post("/predict")
async def predict_stock(request: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor.")

    try:
        hoje = datetime.date.today().strftime('%Y-%m-%d')
        df = yf.download(request.ticker, period="100d", end=hoje)

        if df.empty or len(df) < 60:
            raise HTTPException(status_code=400, detail="Dados insuficientes no Yahoo Finance.")

        df_close = df[['Close']]
        X_infer = preparar_dados_predicao(df_close, scaler, window_size=60)

        predicao_escalada = model.predict(X_infer, verbose=0)  # verbose=0 para manter o terminal limpo
        preco_projetado = scaler.inverse_transform(predicao_escalada)

        return {
            "ticker": request.ticker,
            "next_day_prediction_close": float(preco_projetado[0][0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
