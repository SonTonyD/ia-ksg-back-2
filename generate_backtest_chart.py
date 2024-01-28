import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import ai_script.agent as agt
import datetime
from keras.models import load_model
import ai_script.ai as ai

models_map = {
    "1": "./ai_script/model_1.pth",
    "2": "./ai_script/lstm_v3.h5"
}

def run_prediction_RL(new_sequence):
    model = agt.DQN_LSTM_Agent(1, 1, 1)
    model.load_agent("./ai_script/model_1.pth")

    window_size = 30
    scaler = MinMaxScaler()

    new_sequence = new_sequence.reshape(-1, 1)
    data = scaler.fit_transform(new_sequence)
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

    return action


def run_prediction(model, new_sequence):
    prediction = model.predict(new_sequence)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_action = predicted_class.item()

    if predicted_action == 0:
        action = "sell"
    elif predicted_action == 1:
        action = "buy"
    else:
        action = "hold"

    return action

def run_prediction_bot(new_sequence):
    rsi = ai.calculate_rsi(new_sequence)
    # Décider de l'action à prendre
    action = ""
    fiabilite = 0

    if rsi < 40:
        action = "buy"
        # Fiabilité de 100% si RSI ≤ 35, sinon calculée en fonction de la distance au seuil
        fiabilite = 100 if rsi <= 35 else min(100, (40 - rsi) * 100 / 5)
    elif rsi > 60:
        action = "sell"
        # Fiabilité de 100% si RSI ≥ 65, sinon calculée en fonction de la distance au seuil
        fiabilite = 100 if rsi >= 65 else min(100, (rsi - 60) * 100 / 5)
    else:
        action = "hold"
        # Fiabilité basée sur la proximité du RSI à 50
        fiabilite = 100 - abs(50 - rsi) * 100 / 10

    return action



def generate_backtest_chart(stock_symbol, ia_index, images):

    if "plot"+ia_index+"_"+stock_symbol in images and images["plot"+ia_index+"_"+stock_symbol] >= datetime.datetime.now().date():
        print(images["plot"+ia_index+"_"+stock_symbol] >= datetime.datetime.now().date())
        return "./backtest_images/plot"+ia_index+"_"+stock_symbol+".png"
    else:
        print("bonjour")
        images["plot"+ia_index+"_"+stock_symbol] = datetime.datetime.now().date()

        model = load_model(models_map["2"])

        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=360)
        
        index = 1
        prices = []
        prices = np.array(prices)

        date_for_plot = []
        close_price_for_plot = []
        predicted_actions_for_plot = []

        stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval="1d")
        print("Looop")
        for date, row in stock_data.iterrows():
            print(f"Date: {date}, Close Price: {row['Close']}")
            prices = np.append(prices, row['Close'])
            if index>30:
                new_sequence = prices[-30:].reshape(1, 30, 1)
                bot_data = stock_data["Close"].tail(30).tolist()
                scaler = MinMaxScaler(feature_range=(0, 1))
                normalized = scaler.fit_transform(new_sequence.reshape(-1, 1)).reshape(1, 30, 1)

                if ia_index == "2":
                    predicted_action = run_prediction(model, normalized)
                if ia_index == "1":
                    print("RL prediction")
                    predicted_action = run_prediction_RL(normalized)
                if ia_index == "3":
                    predicted_action = run_prediction_bot(bot_data)


                #save data for plot
                date_for_plot.append(date)
                close_price_for_plot.append(row['Close'])
                predicted_actions_for_plot.append(predicted_action)
            
            index += 1

        # Créez un dictionnaire pour mapper les actions aux marqueurs et couleurs correspondants
        actions_markers = {
            "buy": ("g^", "green", "Buy"),
            "sell": ("rv", "red", "Sell"),
            "hold": ("", "blue", "Hold")
        }

        # Créez une liste pour stocker les marqueurs, les couleurs et les étiquettes correspondants à chaque point de données
        markers, colors, labels = zip(*[actions_markers[action] for action in predicted_actions_for_plot])

        # Créez le graphique avec les marqueurs personnalisés
        plt.figure(figsize=(14, 7))
        plt.plot(date_for_plot, close_price_for_plot, 'b-', label="Close Price")
        for i in range(len(date_for_plot)):
            plt.plot(date_for_plot[i], close_price_for_plot[i], markers[i], markersize=10, color=colors[i])

        # Réglez les légendes et les étiquettes
        plt.legend()
        plt.title('Décisions d\'achat et de vente sur la courbe des prix')
        plt.xlabel('Date')
        plt.ylabel('Prix de clôture')
        plt.grid(True)

        # Affichez le graphique
        print(predicted_actions_for_plot)
        from io import BytesIO
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        decodeit = open(f"./backtest_images/plot{ia_index}_{stock_symbol}.png", 'wb')
        decodeit.write(figfile.getvalue())
        
        #plt.savefig("./backtest_images/plot"+ia_index+"_"+stock_symbol+".png")
        return "Image generated !"
    

#generate_backtest_chart("AAPL", "./ai_script/lstm_v3.h5")






