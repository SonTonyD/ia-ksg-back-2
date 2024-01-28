import os
import sys
import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import ai_script.agent as agt
from keras.models import load_model


def fetch_data(stock_symbol, start_date, end_date, interval="1d"):
    stock_data = yf.download(
        stock_symbol, start=start_date, end=end_date, interval=interval
    )
    full_data = stock_data
    return stock_data["Close"].values.reshape(-1, 1), full_data

def reliability_score_margin(prediction):
    prediction = np.squeeze(prediction)
    sorted_prediction = np.sort(prediction)
    return sorted_prediction[-1] - sorted_prediction[-2]

def run_simulation(stock_symbol, model_path):
    model = agt.DQN_LSTM_Agent(1, 1, 1)
    model.load_agent(model_path)
    window_size = 30
    stock_symbol = stock_symbol
    scaler = MinMaxScaler()
    # Définir la date de fin comme la date actuelle
    end_date = datetime.datetime.now().date()

    # Calculer la date de début (6 mois en arrière à partir de la date actuelle)
    start_date = end_date - datetime.timedelta(days=180)

    data, _ = fetch_data(stock_symbol, start_date=start_date, end_date=end_date)

    data = scaler.fit_transform(data)
    diffs = np.vstack(([0], np.diff(data, axis=0)))

    # Création du résultat final
    modified_data = np.hstack((data, diffs))
    observation = modified_data[-window_size:]
    predicted_action = model.select_action(observation)

    if predicted_action == 1:
        action = "buy"
    elif predicted_action == 0:
        action = "sell"
    else:
        action = "hold"

    fiability = random.randint(0, 100)
    position = "short"
    return action, fiability, position


def run_prediction_LSTM(stock_symbol, model_path):
    model = load_model(model_path)

    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=60)
    interval = "1d"
    
    _, stock_data = fetch_data(
        stock_symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    close_data = stock_data["Close"].tail(30).tolist()
    close_data_array = np.array(close_data)
    new_sequence = close_data_array.reshape(1, 30, 1)


    prediction = model.predict(new_sequence)
    predicted_class = np.argmax(prediction, axis=1)
    reliability = reliability_score_margin(prediction)
    #return prediction, predicted_class, reliability

    predicted_action = predicted_class.item()

    if predicted_action == 0:
        action = "buy"
    elif predicted_action == 1:
        action = "sell"
    else:
        action = "hold"

    fiability = int(reliability * 100)
    position = "short"
    return action, fiability, position