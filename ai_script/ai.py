import os
import sys
import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import argparse

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import random
import ai_script.agent as agt


def fetch_data(stock_symbol, start_date, end_date, interval="1d"):
    stock_data = yf.download(
        stock_symbol, start=start_date, end=end_date, interval=interval
    )
    full_data = stock_data
    return stock_data["Close"].values.reshape(-1, 1), full_data


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

    data, full_data = fetch_data(stock_symbol, start_date=start_date, end_date=end_date)
    full_data = full_data.reset_index(drop=True)

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
