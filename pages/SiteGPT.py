import asyncio
import sys
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

st.set_page_config(
    page_title="Site GPT",
    page_icon="🤣",
)

html2text_transformer = Html2TextTransformer()

st.title("Site GPT")

st.markdown(
    """ 
    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)

if "win32" in sys.platform:
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    cmds = [['C:/Windows/system32/HOSTNAME.EXE']]
else:
    cmds = [
    ['du', '-sh', '/Users/fredrik/Desktop'],
    ['du', '-sh', '/Users/fredrik'],
    ['du', '-sh', '/Users/fredrik/Pictures']
]

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    trsanformde = html2text_transformer.transform_documents(docs)
    st.write(docs)