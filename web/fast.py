
import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from playwright.async_api import async_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document


load_dotenv()

app = FastAPI()

asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Pinecone Setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "index-name"
DIMENSION = 3072  

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))

# Pydantic Models
class CrawlRequest(BaseModel):
    url: HttpUrl
    method: str
    depth: int = Field(..., ge=0, le=2) 

class QueryRequest(BaseModel):
    question: str


# Best-First Recursive Crawling
async def fetch_rendered_html(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url, wait_until="networkidle")  
        html = await page.content()
        await browser.close()
        return html

async def best_first_crawl(start_url: str, max_depth: int):
    max_pages = {1: 50, 2: 200}.get(max_depth, 500)
    rendered_html = await fetch_rendered_html(start_url)
    
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(max_depth=max_depth, include_external=False, max_pages=max_pages),
        markdown_generator=DefaultMarkdownGenerator()
    )
    
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(start_url, config=config, initial_html=rendered_html)
        return results

# Playwright-based Single Page Crawling
async def crawl_single_page(url: str):
    rendered_html = await fetch_rendered_html(url)
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url, initial_html=rendered_html)
        return [{"url": result.url, "text": result.markdown} for result in results]

# Chunking and Storing Embeddings
def store_embeddings(crawled_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000, separators=["\n\n", "\n", " ", ""])
    for result in crawled_data:
        chunks = text_splitter.split_text(result["text"])
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store.add_documents(documents)

# Query GPT-4o
def query_gpt4o(user_query: str):
    retriever = vector_store.as_retriever()
    relevant_chunks = retriever.get_relevant_documents(user_query)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt = f"""
    The following context is extracted from a website:
    {context}
    
    Given this information, provide a concise and clear response to the following query:
    {user_query}
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke(prompt)
    return response.content

# API Endpoint to Start Crawling
@app.post("/crawl/")
def start_crawling(request: CrawlRequest):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        url = str(request.url)  
        if request.method not in ["single", "recursive"]:
            raise HTTPException(status_code=400, detail="Invalid method. Choose 'single' or 'recursive'.")

        if request.method == "recursive":
            crawled_data = loop.run_until_complete(best_first_crawl(url, request.depth))
            results = [{"url": result.url, "text": result.markdown.raw_markdown} for result in crawled_data]
        else:
            results = loop.run_until_complete(crawl_single_page(url))

        loop.run_until_complete(asyncio.to_thread(store_embeddings, results))

        return {
            "message": "Crawling completed and data stored successfully.",
            "pages_crawled": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoint to Query Data
@app.post("/query/")
def query_pinecone(request: QueryRequest):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        answer = loop.run_until_complete(asyncio.to_thread(query_gpt4o, request.question))
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)