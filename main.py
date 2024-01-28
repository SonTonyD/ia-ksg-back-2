from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import datetime
import yfinance as yf

import ai_script.ai as ai
import generate_backtest_chart as gbc

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

images = {
    
}


@app.get("/")
async def root():
    return {"message": "IA KSG"}


@app.get("/runAi/{stock_symbol}")
async def run_ai(stock_symbol: str):
    stock_symbol = stock_symbol
    action_1, fiability_1, position_1 = ai.run_prediction_LSTM(
        stock_symbol, "./ai_script/lstm_v3.h5"
    )
    action_2, fiability_2, position_2 = ai.run_prediction_LSTM(
        stock_symbol, "./ai_script/lstm_v2.h5"
    )
    action_3, fiability_3, position_3 = ai.trading_bot(
        stock_symbol
    )

    result = {
        "ai_1": {"action": action_1, "fiability": fiability_1, "position": position_1},
        "ai_2": {"action": action_2, "fiability": fiability_2, "position": position_2},
        "ai_3": {"action": action_3, "fiability": fiability_3, "position": position_3},
    }

    return {"result": result, "stock_symbol": stock_symbol}


@app.get("/getChart/{stock_symbol}")
async def get_chart(stock_symbol: str):
    # Définir la date de fin comme la date actuelle
    end_date = datetime.datetime.now().date()

    # Calculer la date de début (6 mois en arrière à partir de la date actuelle)
    start_date = end_date - datetime.timedelta(days=180)

    # Définir l'intervalle de temps (1 jour)
    interval = "1wk"

    # Obtenir les données de Yahoo Finance
    _, stock_data = ai.fetch_data(
        stock_symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )

    return {"data": stock_data["Close"]}

@app.get("/plot/{ia_index}/{stock_symbol}")
async def get_plot(ia_index: str, stock_symbol: str):
    gbc.generate_backtest_chart(stock_symbol, ia_index, images)
    return FileResponse("./backtest_images/plot"+ia_index+"_"+stock_symbol+".png")