# import json
# import random
# from datetime import datetime

# # Third-party imports
# from colorama import Fore, Style, init
# from lumibot.backtesting import CcxtBacktesting
# from lumibot.entities import Asset
# from lumibot.strategies.strategy import Strategy
# from ollama import chat
# from pydantic import BaseModel
# # from timedelta import Timedelta
# from datetime import timedelta

# # Initialize colorama
# init(autoreset=True)

# # --- MOCK HELPERS (Replace these with your actual llmprompts.py logic) ---
# def get_web_deets(start_date, end_date):
#     """
#     Placeholder: Normally this scrapes news. 
#     Returning dummy headlines for the backtest to function.
#     """
#     headlines = [
#         "Bitcoin shows strong resilience above support levels.",
#         "Market analysts predict a bullish trend for crypto this week.",
#         "Regulatory concerns cause slight dip in altcoin markets.",
#         "Institutional inflows into Bitcoin ETFs hit record highs."
#     ]
#     return f"News from {start_date} to {end_date}: {random.choice(headlines)}"

# def prompt_template(news_text):
#     return f"""
#     Analyze the sentiment of the following news regarding Bitcoin/Crypto:
#     "{news_text}"
    
#     Return a JSON object with:
#     - sentiment: "positive", "negative", or "neutral"
#     - score: a float between 0.0 and 1.0 representing confidence (1.0 is high confidence)
#     """
# # -------------------------------------------------------------------------

# class Response(BaseModel):
#     sentiment: str
#     score: float

# class CryptoTrader(Strategy):
#     def initialize(self, cash_at_risk: float = 0.2, coin: str = "BTC"):
#         self.set_market("24/7")
#         self.sleeptime = "1D"
#         self.last_trade = None
#         self.cash_at_risk = cash_at_risk
#         self.coin = coin

#     def position_sizing(self):
#         cash = self.get_cash()
#         last_price = self.get_last_price(
#             Asset(symbol=self.coin, asset_type=Asset.AssetType.CRYPTO),
#             quote=Asset(symbol="USD", asset_type="crypto"),
#         )
        
#         if last_price is None:
#             quantity = 0
#         else:
#             quantity = cash * self.cash_at_risk / last_price
            
#         return cash, last_price, quantity

#     def get_dates(self):
#         today = self.get_datetime()
#         day_prior = today - Timedelta(days=1)
#         return today.strftime("%Y-%m-%d"), day_prior.strftime("%Y-%m-%d")

#     def get_sentiment(self):
#         today, day_prior = self.get_dates()
        
#         # 1. Get News
#         news = get_web_deets(day_prior, today)
#         print(Fore.YELLOW + f"[NEWS] {news}" + Fore.RESET)

#         # 2. Call Ollama
#         # Note: 'stream=False' ensures we wait for the full response
#         try:
#             response = chat(
#                 model="deepseek-r1:14b",
#                 messages=[{"role": "user", "content": prompt_template(news)}],
#                 format=Response.model_json_schema(), # Enforce JSON structure
#                 options={"temperature": 0} # Deterministic output
#             )
            
#             # 3. Parse Response
#             content = response["message"]["content"]
#             result = json.loads(content)
            
#             print(Fore.LIGHTBLUE_EX + f"[AI] {result}" + Fore.RESET)
#             return result
            
#         except Exception as e:
#             print(Fore.RED + f"Error getting sentiment: {e}" + Fore.RESET)
#             # Fallback in case of AI failure
#             return {"sentiment": "neutral", "score": 0.0}

#     def on_trading_iteration(self):
#         cash, last_price, quantity = self.position_sizing()
        
#         # Check if we have price data
#         if last_price is None:
#             return

#         # Get AI Analysis
#         news_data = self.get_sentiment()
#         sentiment = news_data.get("sentiment", "neutral")
#         probability = news_data.get("score", 0.0)

#         # Logic
#         if cash > (quantity * last_price):
            
#             # BUY SIGNAL
#             if sentiment == "positive" and probability >= 0.7:
#                 if self.last_trade == "sell":
#                     self.sell_all()
                
#                 order = self.create_order(
#                     Asset(symbol=self.coin, asset_type=Asset.AssetType.CRYPTO),
#                     quantity,
#                     "buy",
#                     type="market",
#                     quote=Asset(symbol="USD", asset_type="crypto"),
#                 )
#                 print(Fore.LIGHTMAGENTA_EX + f"[BUY] {order}" + Fore.RESET)
#                 self.submit_order(order)
#                 self.last_trade = "buy"

#             # SELL SIGNAL
#             elif sentiment == "negative" and probability >= 0.7:
#                 if self.last_trade == "buy":
#                     self.sell_all()
                
#                 order = self.create_order(
#                     Asset(symbol=self.coin, asset_type=Asset.AssetType.CRYPTO),
#                     quantity,
#                     "sell",
#                     type="market",
#                     quote=Asset(symbol="USD", asset_type="crypto"),
#                 )
#                 print(Fore.LIGHTMAGENTA_EX + f"[SELL] {order}" + Fore.RESET)
#                 self.submit_order(order)
#                 self.last_trade = "sell"

# if __name__ == "__main__":
#     # Shortened timeframe for testing (Ollama is slower than standard backtests)
#     start_date = datetime(2023, 12, 1) 
#     end_date = datetime(2023, 12, 5)
    
#     exchange_id = "kraken"
#     kwargs = {
#         "exchange_id": exchange_id,
#     }
    
#     # Initialize Backtest
#     CcxtBacktesting.MIN_TIMESTEP = "day" # Important for daily data

#     results, strat_obj = CryptoTrader.run_backtest(
#         CcxtBacktesting,
#         start_date,
#         end_date,
#         benchmark_asset="BTC/USD",
#         quote_asset=Asset(symbol="USD", asset_type="crypto"),
#         parameters={"cash_at_risk": 0.25, "coin": "BTC"},
#         **kwargs,
#     )


import json
import random
from datetime import datetime, timedelta  # FIXED: Using standard timedelta

# Third-party imports
from colorama import Fore, Style, init
from lumibot.backtesting import YahooDataBacktesting  # CHANGED: More reliable for backtests
from lumibot.entities import Asset
from lumibot.strategies.strategy import Strategy
from ollama import chat
from pydantic import BaseModel

# Initialize colorama
init(autoreset=True)

# # --- MOCK HELPERS (Replace these with your actual llmprompts.py logic) ---
# def get_web_deets(start_date, end_date):
#     headlines = [
#         "Bitcoin shows strong resilience above support levels.",
#         "Market analysts predict a bullish trend for crypto this week.",
#         "Regulatory concerns cause slight dip in altcoin markets.",
#         "Institutional inflows into Bitcoin ETFs hit record highs."
#     ]
#     return f"News from {start_date} to {end_date}: {random.choice(headlines)}"

# def prompt_template(news_text):
#     return f"""
#     Analyze the sentiment of the following news regarding Bitcoin/Crypto:
#     "{news_text}"
    
#     Return a JSON object with:
#     - sentiment: "positive", "negative", or "neutral"
#     - score: a float between 0.0 and 1.0 representing confidence (1.0 is high confidence)
#     """

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Initialize the search tool
search = DuckDuckGoSearchAPIWrapper(max_results=5)

def get_web_deets(start_date, end_date):
    # This searches the web for news in that specific timeframe
    query = f"bitcoin crypto market news sentiment {start_date}"
    try:
        results = search.run(query)
        if not results:
            return "Market is quiet. No significant news found."
        return results
    except Exception as e:
        print(f"Search Error: {e}")
        return "Market is quiet."

def prompt_template(news_text):
    # Improved prompt to force diverse scores (not just 0.5 or 0.8)
    return f"""
    Analyze the sentiment of this crypto news summary:
    "{news_text}"
    
    Determine the market sentiment score from -1.0 (Extreme Fear) to 1.0 (Extreme Greed).
    
    Return ONLY a JSON object:
    {{
        "sentiment": "positive" or "negative" or "neutral",
        "score": <float between 0.0 and 1.0 representing strictly the STRENGTH of the signal>
    }}
    """
# -------------------------------------------------------------------------

class Response(BaseModel):
    sentiment: str
    score: float

class CryptoTrader(Strategy):
    def initialize(self, cash_at_risk: float = 0.2, coin: str = "BTC-USD"):
        self.set_market("24/7")
        self.sleeptime = "1D"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.coin = coin

    def position_sizing(self):
        cash = self.get_cash()
        
        # Yahoo Finance uses "BTC-USD" for symbol, and asset_type is usually ignored or set to generic
        last_price = self.get_last_price(self.coin)
        
        if last_price is None:
            quantity = 0
        else:
            quantity = cash * self.cash_at_risk / last_price
            
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        day_prior = today - timedelta(days=1)
        return today.strftime("%Y-%m-%d"), day_prior.strftime("%Y-%m-%d")

    def get_sentiment(self):
        today, day_prior = self.get_dates()
        
        # 1. Get News
        news = get_web_deets(day_prior, today)
        print(Fore.YELLOW + f"[NEWS] {news}" + Fore.RESET)

        # 2. Call Ollama
        try:
            response = chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt_template(news)}],
                format=Response.model_json_schema(),
                options={"temperature": 0}
            )
            
            # 3. Parse Response
            content = response["message"]["content"]
            result = json.loads(content)
            
            print(Fore.LIGHTBLUE_EX + f"[AI] {result}" + Fore.RESET)
            return result
            
        except Exception as e:
            print(Fore.RED + f"Error getting sentiment: {e}" + Fore.RESET)
            return {"sentiment": "neutral", "score": 0.0}

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        
        if last_price is None:
            return

        news_data = self.get_sentiment()
        sentiment = news_data.get("sentiment", "neutral")
        probability = news_data.get("score", 0.0)

        if cash > (quantity * last_price):
            
            # BUY SIGNAL
            if sentiment == "positive" and probability >= 0.7:
                if self.last_trade == "sell":
                    self.sell_all()
                
                # For Yahoo, we simplify the order creation slightly
                order = self.create_order(
                    self.coin,
                    quantity,
                    "buy",
                    type="market"
                )
                print(Fore.LIGHTMAGENTA_EX + f"[BUY] {order}" + Fore.RESET)
                self.submit_order(order)
                self.last_trade = "buy"

            # SELL SIGNAL
            elif sentiment == "negative" and probability >= 0.7:
                if self.last_trade == "buy":
                    self.sell_all()
                
                order = self.create_order(
                    self.coin,
                    quantity,
                    "sell",
                    type="market"
                )
                print(Fore.LIGHTMAGENTA_EX + f"[SELL] {order}" + Fore.RESET)
                self.submit_order(order)
                self.last_trade = "sell"

if __name__ == "__main__":
    start_date = datetime(2023, 12, 1) 
    end_date = datetime(2024, 12, 1)
    
    # CHANGED: Using YahooDataBacktesting
    results, strat_obj = CryptoTrader.run_backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="BTC-USD",
        parameters={"cash_at_risk": 0.25, "coin": "BTC-USD"},
    )