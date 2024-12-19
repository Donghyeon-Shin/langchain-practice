import asyncio
import sys
from langchain.document_loaders import SitemapLoader
import streamlit as st

@st.cache_data(show_spinner="Loading website....")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 5
    docs = loader.load()
    return docs


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

if "win32" in sys.platform:
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    cmds = [["C:/Windows/system32/HOSTNAME.EXE"]]
else:
    cmds = [
        ["du", "-sh", "/Users/fredrik/Desktop"],
        ["du", "-sh", "/Users/fredrik"],
        ["du", "-sh", "/Users/fredrik/Pictures"],
    ]

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
        docs = load_website(url)
        st.write(docs)
