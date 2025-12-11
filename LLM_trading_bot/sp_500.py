
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

# --- MOCK HELPERS (Replace these with your actual llmprompts.py logic) ---

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Initialize the search tool
search = DuckDuckGoSearchAPIWrapper(max_results=5)

def get_web_deets(start_date, end_date):
    # This searches the web for news in that specific timeframe
    # query = f"bitcoin crypto market news sentiment {start_date}"
    query = f"S&P 500 market news sentiment {start_date} to {end_date}"
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
    Analyze the sentiment of this S&P 500 market news summary:
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
    
class StockTrader(Strategy):
    def initialize(self, cash_at_risk: float = 0.5, symbol: str = "SPY"):
        # CHANGED: Set to NYSE (New York Stock Exchange) hours
        self.set_market("NYSE") 
        self.sleeptime = "1D"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.symbol = symbol

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        
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
        
        # CHANGED: Search for S&P 500 news instead of Bitcoin
        # (Make sure your get_web_deets function uses this new query!)
        
        
        # ... (Rest of the sentiment logic is the same) ...
        # # For this example, I'll return a mock result
        # return {"sentiment": "positive", "score": 0.9} 
        
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
            if sentiment == "positive" and probability >= 0.7:
                if self.last_trade == "sell":
                    self.sell_all()
                
                # CHANGED: Buy SPY
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="market"
                )
                self.submit_order(order)
                self.last_trade = "buy"

            elif sentiment == "negative" and probability >= 0.7:
                if self.last_trade == "buy":
                    self.sell_all()
                
                # CHANGED: Sell SPY
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="market"
                )
                self.submit_order(order)
                self.last_trade = "sell"

if __name__ == "__main__":
    start_date = datetime(2023, 1, 1) 
    end_date = datetime(2024, 1, 1)
    
    # CHANGED: Run backtest on SPY
    results, strat_obj = StockTrader.run_backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="SPY",
        parameters={"cash_at_risk": 0.5, "symbol": "SPY"},
    )