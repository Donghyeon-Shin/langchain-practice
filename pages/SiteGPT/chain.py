import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from pages.SiteGPT.utils import save_message

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.response = ""

    def on_llm_start(self, *arg, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *arg, **kwargs):
        save_message(st.session_state["messages"], self.response, "ai")

    def on_llm_new_token(self, token, *arg, **kwargs):
        self.response += token
        self.message_box.markdown(self.response)


common_llm = ChatOpenAI(
    temperature=0.1,
)

choose_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't
    Just say you don't know, don't make anyting up.

    Then, give a score to the answer between 0 and 5. 0 being not helpful to
    the user and 5 being helpful to the user.

    Make sure to include the answer's score.

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
            """,
        ),
        
        ("human", "{question}"),
    ]
)


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
    }


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | choose_llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})

def invoke_chain(retriever, question):
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    with st.chat_message("ai"):
        answer = chain.invoke(question)