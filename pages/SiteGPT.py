import asyncio
import sys
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_data(show_spinner="Loading website....")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url, filter_urls=[r"^(.*\/research\/).*"], parsing_function=parse_page
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
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
        docs = load_website(url)
        st.write(docs)
