import streamlit as st
import json
import openai as client
import yfinance
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from typing_extensions import override
from openai import AssistantEventHandler
from openai.types.beta.threads.runs import ToolCall, RunStep


def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Ticker symbol of company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company.",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company.",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company.",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
]


def create_assistant():
    assistant = client.beta.assistants.create(
        name="Investor Assistant",
        instructions="You help user do research on publicly traded companies and you help them decide if they should buy the stock or not.",
        model="gpt-3.5-turbo-0125",
        tools=functions,
    )
    return assistant.id


def create_thread():
    thread = client.beta.threads.create()
    return thread.id


def send_thread_message(thread_id, context):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=context,
    )


class EventHandler(AssistantEventHandler):
    def __init__(self, thread_id, assistant_id):
        super().__init__()
        self.output = None
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.run_id = None
        self.run_step = None
        self.response = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        self.response += delta.value.replace("$", "\$")
        self.message_box.markdown(self.response)

    @override
    def on_text_done(self, text):
        save_message(self.response, "ai")

    @override
    def on_run_step_created(self, run_step: RunStep) -> None:
        print(f"on_run_step_created")
        self.run_id = run_step.run_id
        self.run_step = run_step

    @override
    def on_tool_call_done(self, tool_call: ToolCall) -> None:
        # tool_call이 끝났을 때
        current_retrieving_run = client.beta.threads.runs.retrieve(
            thread_id=self.thread_id, run_id=self.run_id
        )
        # 현재 retrieving_run 정보
        print(f"\nDONE STATUS: {current_retrieving_run.status}")
        if current_retrieving_run.status == "completed":
            return
        elif current_retrieving_run.status == "requires_action":
            # keep_retrieving_run.status가 actoin을 요구하면
            outputs = []
            
            for action in current_retrieving_run.required_action.submit_tool_outputs.tool_calls:
                function = action.function
                tool_call_id = action.id
                if function.name in functions_map:
                    outputs.append(
                        {
                            "output": functions_map[function.name](
                                json.loads(function.arguments)
                            ),
                            "tool_call_id": tool_call_id,
                        }
                    )
                else:
                    print("unknown function")
            print(f"The number of outputs : {len(outputs)}")

            # 해당 요구하는 action이 내가 정한 함수 안에 있을 때
            with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=self.thread_id,
                run_id=self.run_id,
                tool_outputs=outputs,
                event_handler=EventHandler(self.thread_id, self.assistant_id),
            ) as stream:
                stream.until_done()
            # 해당 action에 대한 stream을 만듬

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == "function":
            # self.arguments += delta.function.arguments
            pass
        elif delta.type == "code_interpreter":
            print(f"on_tool_call_delta > code_interpreter")
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)
        else:
            print("ELSE")
            print(delta, end="", flush=True)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for dic_message in st.session_state["messages"]:
        send_message(dic_message["message"], dic_message["role"], save=False)


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(message, role)


st.set_page_config(
    page_title="InvestorGPTVer2",
    page_icon="🤣",
)

st.markdown(
    """
    # Investor GPT

    Welcome to Investor GPT.

    Write down the name of a company and our Agent will do the resaerch for you.

    Using OpenAI Assistant Not agent
    """
)


company = st.chat_input("Write the name of company you are interested on.")

if "assistant_id" not in st.session_state:
    st.session_state["assistant_id"] = create_assistant()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if company:
    save_message(company, "human")
    paint_history()
    thread_id = create_thread()
    send_thread_message(thread_id, f"I want to know if {company} is a good buy.")
    with st.chat_message("ai"):
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=st.session_state["assistant_id"],
            instructions="You help user do research on publicly traded companies and you help them decide if they should buy the stock or not.",
            event_handler=EventHandler(thread_id, st.session_state["assistant_id"]),
        ) as stream:
            stream.until_done()
