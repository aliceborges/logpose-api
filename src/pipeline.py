from typing import Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


def baixar_dados(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Baixa os dados históricos do Yahoo Finance."""
    print(f"Baixando dados para {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    return df[['Close']]


def preparar_dados(df: pd.DataFrame, window_size: int = 60) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Normaliza e cria as janelas de tempo para a LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    dados_escalados = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(dados_escalados)):
        X.append(dados_escalados[i - window_size:i, 0])
        y.append(dados_escalados[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def preparar_dados_predicao(df: pd.DataFrame, scaler: Any, window_size: int = 60) -> np.ndarray:
    """Prepara apenas os últimos N dias para a inferência na API."""
    ultimos_dias = df[-window_size:].values
    dados_escalados = scaler.transform(ultimos_dias)
    X_test = np.array([dados_escalados[:, 0]])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test
