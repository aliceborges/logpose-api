import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.pipeline import baixar_dados, preparar_dados


def treinar_modelo(ticker: str = 'DIS', start_date: str = '2018-01-01', end_date: str = '2024-07-20'):
    """
    Treina o modelo LSTM baseado em um ticker e intervalo de datas parametrizáveis.
    """
    window_size = 60

    print(f"Iniciando pipeline de dados para {ticker} no período de {start_date} a {end_date}...")
    df = baixar_dados(ticker, start_date, end_date)

    if df.empty or len(df) <= window_size:
        print("Erro: Dados insuficientes para o período selecionado.")
        return

    X, y, scaler = preparar_dados(df, window_size)

    split_index = int(len(X) * 0.9)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print("Construindo o modelo LSTM...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    print("\nIniciando o treino...")
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

    print("\n" + "=" * 30)
    print("📊 AVALIAÇÃO DO MODELO (TESTE)")
    print("=" * 30)

    predicoes = model.predict(X_test)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    predicoes_reais = scaler.inverse_transform(predicoes)

    mae = mean_absolute_error(y_test_real, predicoes_reais)
    rmse = np.sqrt(mean_squared_error(y_test_real, predicoes_reais))
    mape = mean_absolute_percentage_error(y_test_real, predicoes_reais)

    print(f"MAE  (Erro Absoluto Médio):          $ {mae:.2f}")
    print(f"RMSE (Raiz do Erro Quadrático Médio): $ {rmse:.2f}")
    print(f"MAPE (Erro Percentual Absoluto):      {mape:.2%}")
    print("=" * 30 + "\n")

    os.makedirs('models', exist_ok=True)
    model.save('models/modelo_lstm.keras')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Modelo e Scaler guardados com sucesso na pasta 'models/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento do modelo LSTM da Log Pose")
    parser.add_argument("--ticker", type=str, default="DIS", help="Símbolo da ação (ex: DIS, AAPL, PETR4.SA)")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Data de início no formato YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-07-20", help="Data de fim no formato YYYY-MM-DD")

    args = parser.parse_args()

    treinar_modelo(ticker=args.ticker, start_date=args.start, end_date=args.end)
