import streamlit as st
import json
import openai as client
import yfinance
import re
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


def create_investor_assistant():
    assistant = client.beta.assistants.create(
        name="Investor Assistant",
        instructions="You help user do research on publicly traded companies and you help them decide if they should buy the stock or not.",
        model="gpt-3.5-turbo-0125",
        tools=functions,
    )
    return assistant.id


def create_investor_thread():
    thread = client.beta.threads.create()
    return thread.id


def create_book_assistant():
    assistant = client.beta.assistants.create(
        name="Book Assistant",
        instructions="You help user do research with their question on the files they upload.",
        model="gpt-3.5-turbo-0125",
        tools=[{"type": "file_search"}],
    )
    return assistant.id


@st.cache_resource(show_spinner="Loads file...")
def loads_file(file):
    file_name = file.name
    file_path = f"./.cache/files/{file_name}"
    file_context = file.read()
    with open(file_path, "wb") as f:
        f.write(file_context)
    return file_path


def create_book_thread(file_path):
    thread = client.beta.threads.create()

    file = client.files.create(file=open(file_path, "rb"), purpose="assistants")

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I want you to help me with this file",
        attachments=[
            {
                "file_id": file.id,
                "tools": [
                    {
                        "type": "file_search",
                    }
                ],
            }
        ],
    )
    return thread.id


def send_thread_message(thread_id, context):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=context,
    )


class EventHandler(AssistantEventHandler):
    def __init__(self, thread_id, assistant_id, messageType):
        super().__init__()
        self.output = None
        self.thread_id = thread_id
        self.assistant_id = assistant_id
        self.run_id = None
        self.run_step = None
        self.response = ""
        self.messageType = messageType

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        self.response += delta.value.replace("$", "\$")
        self.message_box.markdown(self.response)

    @override
    def on_message_done(self, message):
        if self.messageType == "book_messages":
            message_content = message.content[0].text
            annotations = message_content.annotations
            citations = []
            for index, annotation in enumerate(annotations):
                message_content.value = message_content.value.replace(
                    annotation.text, f"[{index}]"
                )
                if file_citation := getattr(annotation, "file_citation", None):
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citations.append(
                        f"[{index}] {cited_file.filename}({annotation.start_index}-{annotation.end_index})"
                    )
            matches = len(re.findall(r"【[^】]*】", self.response))
            for n in range(matches):
                self.response = re.sub(
                    r"【[^】]*】",
                    f"[{n}]",
                    self.response,
                    1,
                )
            self.response += f"\n\nSources: {citations}"
            self.message_box.markdown(self.response)
        save_message(self.messageType, self.response, "ai")

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

            for (
                action
            ) in current_retrieving_run.required_action.submit_tool_outputs.tool_calls:
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
                event_handler=EventHandler(
                    self.thread_id, self.assistant_id, messageType=self.messageType
                ),
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


@st.cache_resource(show_spinner="Loads file....")
def embed_file(file):
    file_name = file.name
    file_path = f"./.cache/files/{file_name}"
    file_context = file.read()
    with open(file_path, "wb") as f:
        f.write(file_context)


def save_message(type, message, role):
    st.session_state[type].append({"message": message, "role": role})


def paint_history(type):
    for dic_message in st.session_state[type]:
        send_message(type, dic_message["message"], dic_message["role"], save=False)


def send_message(type, message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            save_message(type, message, role)


st.set_page_config(
    page_title="AssistantsGPT",
    page_icon="🤣",
)

with st.sidebar:
    select_assistants = st.selectbox(
        "Choose your assistant", ("", "Investor Assistant", "Book Assistant")
    )

if "investor_messages" not in st.session_state:
    st.session_state["investor_messages"] = []

if "book_messages" not in st.session_state:
    st.session_state["book_messages"] = []


if select_assistants == "Investor Assistant":
    st.markdown(
        """
        # Investor GPT

        Welcome to Investor GPT.

        Write down the name of a company and our Assistant will do the resaerch for you.

        Using OpenAI Assistant Not agent
        """
    )

    investor_question = st.chat_input(
        "Write the name of company you are interested on."
    )

    if "investor_assistant_id" not in st.session_state:
        st.session_state["investor_assistant_id"] = create_investor_assistant()

    if "investor_thread_id" not in st.session_state:
        st.session_state["investor_thread_id"] = create_investor_thread()

    paint_history("investor_messages")

    if investor_question:
        send_message("investor_messages", investor_question, "human")
        send_thread_message(st.session_state["investor_thread_id"], investor_question)
        # "I want to know if {company} is a good buy."
        with st.spinner("Create Answer...."):
            with st.chat_message("ai"):
                with client.beta.threads.runs.stream(
                    thread_id=st.session_state["investor_thread_id"],
                    assistant_id=st.session_state["investor_assistant_id"],
                    instructions="You help user do research on publicly traded companies and you help them decide if they should buy the stock or not.",
                    event_handler=EventHandler(
                        thread_id=st.session_state["investor_thread_id"],
                        assistant_id=st.session_state["investor_assistant_id"],
                        messageType="investor_messages",
                    ),
                ) as stream:
                    stream.until_done()
elif select_assistants == "Book Assistant":
    st.markdown(
        """
        # Book GPT

        Welcome to Book GPT.

        Input the file of Book txt and Write down any questions.
        Then, our Assistant will do the answer for you.

        Using OpenAI Assistant Not agent
        """
    )

    paint_history("book_messages")

    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["txt", "pdf", "docx"],
        )

    if file:
        file_path = loads_file(file)
        if "book_assistant_id" not in st.session_state:
            st.session_state["book_assistant_id"] = create_book_assistant()

        if "book_thread_id" not in st.session_state:
            st.session_state["book_thread_id"] = create_book_thread(file_path)

        book_question = st.chat_input("Ask anything about your file....")

        if book_question:
            send_message("book_messages", book_question, "human")
            send_thread_message(st.session_state["book_thread_id"], book_question)
            with st.spinner("Create Answer...."):
                with st.chat_message("ai"):
                    with client.beta.threads.runs.stream(
                        thread_id=st.session_state["book_thread_id"],
                        assistant_id=st.session_state["book_assistant_id"],
                        instructions="You help user do research on publicly traded companies and you help them decide if they should buy the stock or not.",
                        event_handler=EventHandler(
                            thread_id=st.session_state["book_thread_id"],
                            assistant_id=st.session_state["book_assistant_id"],
                            messageType="book_messages",
                        ),
                    ) as stream:
                        stream.until_done()
