import streamlit as st
import json
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationSummaryBufferMemory
from pages.SiteGPT.utils import save_message, send_message


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.response = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(st.session_state["messages"], self.response, "ai")
        self.response = ""

    def on_llm_new_token(self, token, *arg, **kwargs):
        self.response += token
        self.message_box.markdown(self.response)


history_prompt = ChatPromptTemplate.from_template(
    """
    You are given a 'history' that the user and ai talked about.
    BASED on this history If a user asks for something SIMILAR, find the answer in history and print it out.
    If the question is not in history, JUST SAY 'None' 
    DO NOT SAY 'answer: None' and 'Answer: None'

    examples_1
    History:
    human: What is the color of the occean?
    ai: Blue. Source:https://ko.wikipedia.org/wiki/%EB%B0%94%EB%8B%A4%EC%83%89 Date:2024-10-13

    Question : What color is the ocean?
    Answer : Blue. Source:https://ko.wikipedia.org/wiki/%EB%B0%94%EB%8B%A4%EC%83%89 Date:2024-10-13

    examples_2
    History:
    human: What is the capital of Georgia?
    ai: Tbilisi Source:https://en.wikipedia.org/wiki/Capital_of_Georgia Date:2022-08-22

    Question : What are the major cities in Georgia?
    Answer : Tbilisi Source:https://en.wikipedia.org/wiki/Capital_of_Georgia Date:2022-08-22

    examples_3
    human: When was Avator released?
    ai: 2009 Source:https://en.wikipedia.org/wiki/Avatar_(franchise) Date:2022-12-18
    
    Question : What is Avator2?
    Answer : None
    
    examples_4
    History:
    human: What is the capital of the United States?
    ai: Washington, D.C. Source:https://ko.wikipedia.org/wiki/%EB%AF%B8%EA%B5%AD Date:2022-10-18
    
    Question : What is the capital of the Korea?
    Answer : None

    Your turn!
    History: {history}

    Question: {question}
    """
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't
    Just say you don't know, don't make anyting up.

    Then, give a score to the answer between 0 and 5. 0 being not helpful to
    the user and 5 being helpful to the user.

    Make sure to include the answer's score.
    ONLY one result should be output.

    Context : {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question : {question}
    """
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Return the sources of the answers as they are, do not change them.
            You must print out only one answer. and Don't print out the score
            Answer: {answers}

            You also have a past answer. Please refert o them and write your answers
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

history_llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
)

common_llm = ChatOpenAI(
    temperature=0.1,
)

choose_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=common_llm,
        memory_key="history",
        max_token_limit=150,
        return_messages=True,
    )

memory = st.session_state["memory"]


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | common_llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "context": doc.page_content,
                        "question": question,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "history": memory.load_memory_variables({})["history"],
    }


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    history = inputs["history"]
    choose_chain = choose_prompt | choose_llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {"question": question, "answers": condensed, "history": history}
    )


def format_message(messages):
    history = ""
    i = 0
    st.write()
    for message in messages:
        if i is not len(messages) - 1:
            history += f"{message['role']} : {message['message']}\n"
        if i % 2 == 1:
            history += "\n"
        i = i + 1

    return history


def invoke_chain(messages, retriever, question):

    history = format_message(messages)
    history_chain = history_prompt | history_llm

    result = history_chain.invoke({"history": history, "question": question})
    response = result.content

    if response == "None" or response == "Answer: None":
        research_chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            answer = research_chain.invoke(question)
            memory.save_context({"input": question}, {"output": answer.content})
    else:
        send_message(messages, response, "ai")

def initialize_memory():
    memory.clear()
