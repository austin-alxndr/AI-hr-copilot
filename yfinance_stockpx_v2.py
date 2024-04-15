################# Define custom functions w/in Python ################

from datetime import datetime, timedelta
import yfinance as yf

def get_current_stock_price(ticker):
    """How to get Current Stock Price"""

    ticker_data = yf.Ticker(ticker)
    recent = ticker_data.history(period="1d")
    return {"price": recent.iloc[0]["Close"], "currency": ticker_data.info["currency"]}

def get_stock_performance(ticker, days):
    """Get Percentage Change (Delta) in Stock Price"""

    past_date = datetime.today() - timedelta(days=days)
    ticker_data = yf.Ticker(ticker)
    history = ticker_data.history(start=past_date)
    old_price = history.iloc[0]["Close"]
    current_price = history.iloc[-1]["Close"]
    return {"percent_change": ((current_price - old_price) / old_price) * 100}

def get_stock_news(ticker):
    ticker_data = yf.Ticker(ticker)
    news_data = ticker_data.news
    return {
        "title" : news_data[0]['title'], 
        "publisher" : news_data[0]['publisher'], 
        "link" : news_data[0]['link']
    }

# print(get_current_stock_price("AAPL")) --> Test get_current_stock_price Function
 
# print(get_stock_performance("AAPL", 30)) --> Test get_stock_performance Function


##################### Make custom tool ############################

from typing import Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class CurrentStockPriceInput(BaseModel):
    """get_current_stock_price"""

    ticker: str = Field(description="Stock's ticker symbol")

class CurrentStockPriceTool(BaseTool):
    name = "get_current_stock_price"
    description = """
        Useful when you want to get the current stock price. You should enter the stock ticker symbol recognized by Yahoo Finance.
        """
    args_schema: Type[BaseModel] = CurrentStockPriceInput

    def _run(self, ticker: str):
        price_response = get_current_stock_price(ticker)
        return price_response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_current_stock_price Does not support asynchronous")
    
class StockPercentChangeInput(BaseModel):
    """input of get_stock_performance"""
    ticker: str = Field(description="Stock's ticker symbol")
    days: int = Field(description="Difference of number of days from today")

class StockPerformanceTool(BaseTool):
    name = "get_stock_performance"
    description = """
        Used to check and calculate a stock's performance. You should enter the stock symbol RECOGNIZED by Yahoo Finance. You should enter the difference in number of days from today. The output will be the percentage change in the stock price.
        """
    args_schema: Type[BaseModel] = StockPercentChangeInput

    def _run(self, ticker: str, days: int):
        response = get_stock_performance(ticker, days)
        return response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_stock_performance does not support asynchronous")

class CurrentStockNewsInput(BaseModel):
    """get_stock_news"""
    ticker: str = Field(description="Stock's ticker symbol")

class CurrentStockNewsTool(BaseTool):
    name = "get_stock_news"
    description = """
        Useful when you want to get the current stock news. You should enter the stock ticker symbol RECOGNIZED by Yahoo Finance. For companies in Indonesia, you have to add a .JK at the end. For example, Bank Central Asia is BBCA.JK and Bank Mandiri is BMRI.JK.
        """
    args_schema: Type[BaseModel] = CurrentStockNewsInput

    def _run(self, ticker: str):
        news_response = get_stock_news(ticker)
        return news_response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_stock_news Does not support asynchronous")

########################### Create Agent ##############################

import os
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = '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'

llm = ChatOpenAI(
    model='gpt-3.5-turbo-0613',
    temperature=0
)

tools = [CurrentStockPriceTool(), StockPerformanceTool(), CurrentStockNewsTool()]

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.OPENAI_MULTI_FUNCTIONS, 
    verbose=True
)

agent.invoke(
    "what is the current price for Microsoft stock? And what is its performance in the past 6 months?"
)