import asyncio
import sys
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
import streamlit as st

llm = ChatOpenAI(
    temperature=0.1,
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

            Answer: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
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
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers)
    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_resource(show_spinner="Loading website....")
def get_retriever_in_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    url_name = (
        str(url).replace("https://", "").replace(".", "").replace("/sitemapxml", "")
    )
    st.write(url_name)
    cache_dir = LocalFileStore(f"./.cache/Site_embeddings/{url_name}")
    embedder = OpenAIEmbeddings()
    cache_embedder = CacheBackedEmbeddings.from_bytes_store(embedder, cache_dir)
    vector_store = FAISS.from_documents(docs, cache_embedder)
    return vector_store.as_retriever()


st.set_page_config(
    page_title="Site GPT",
    page_icon="🤣",
)
st.title("Site GPT")

st.markdown(
    """ 
    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)

# https://deepmind.google/sitemap.xml

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Stiemap URL")
    else:
        retriever = get_retriever_in_website(url)
        query = st.text_input("Ask a question to the website")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)
            st.markdown(result.content)
