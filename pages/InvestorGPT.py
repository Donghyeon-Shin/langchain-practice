import streamlit as st
import os
import requests
from typing import Type
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain.utilities import DuckDuckGoSearchAPIWrapper

llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo-0125",)

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


class CompanyStockSearchArgsSchema(BaseModel):
    symbol: str = Field(description="Stock symbol of the company.Example: AAPL, TSLA")


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class CompanyStackPerformanceTool(BaseTool):
    name: Type[str] = "CompanyStackPerformance"
    description: Type[
        str
    ] = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyStockSearchArgsSchema] = CompanyStockSearchArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        response = r.json()
        return list(response["Weekly Time Series"].items())[:200]


class CompanyIncomeStatementTool(BaseTool):
    name: Type[str] = "CompanyIncomeStatement"
    description: Type[
        str
    ] = """
    Use this to get an income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyStockSearchArgsSchema] = CompanyStockSearchArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()["annualReports"]


class CompanyOverviewTool(BaseTool):
    name: Type[str] = "CompanyOverview"
    description: Type[
        str
    ] = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyStockSearchArgsSchema] = CompanyStockSearchArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
        )
        return r.json()


class StockMarketSymbolSearchTool(BaseTool):
    name: Type[str] = "StockMarketSymbolSearch"
    description: Type[
        str
    ] = """
    Use this tool to find the stock market symbool for a company.
    It takes a query as an argument.
    Example query: Stock Market Symbol for Apple Company.
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = (
        StockMarketSymbolSearchToolArgsSchema
    )

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStackPerformanceTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a hedge fund manager.

            You evaluate a company and provide your opinion and reasons why the stock is a boy or not.

            Consider the performance of a stock, the company overview and the inocome statement
            
            Be assertive in your judgement and recommend the stock or advise teh user against it.
            """
        )
    },
)

st.set_page_config(
    page_title="Investor GPT",
    page_icon="🤣",
)

st.markdown(
    """
    # Investor GPT

    Welcome to Investor GPT.

    Write down the name of a company and our Agent will do the resaerch for you.
    """
)

company = st.text_input("Write the name of company you are interested on.")

if company:
    result = agent.invoke(company)

    st.write(result["output"].replace("$","\$"))
