
import os
import base64
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from qdrant_client import QdrantClient, models
from langfuse import Langfuse # <-- IMPORT LANGFUSE
from langfuse.openai import AsyncOpenAI
import uuid
import chainlit as cl
from tools.tool_decorator import tool

# ==============================================================================
# CORRECTED: INITIALIZE LANGFUSE AND THE PATCHED CLIENT FOR THE TOOL
# ==============================================================================
langfuse = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    host=os.environ.get("LANGFUSE_HOST"),
)

# This client is now properly traced
tool_client = AsyncOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta",
)
# ==============================================================================


qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_CLOUD_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)

COLLECTION_NAME = "website_content"

# The rest of the file remains the same...

def _setup_qdrant_collection():
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )

async def _get_text_embedding(text: str):
    response = await tool_client.embeddings.create(model="models/embedding-001", input=text)
    return response.data[0].embedding

async def _get_image_description(image_bytes: bytes):
    response = await tool_client.chat.completions.create(
        model="models/gemini-1.5-flash-latest",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image in detail for a retrieval system."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"}},
            ]},
        ],
    )
    return response.choices[0].message.content

@tool
async def scrape_and_embed_website(url: str) -> str:
    """
    Scrapes a website's text and images, creates embeddings, and stores them in the knowledge base.
    This tool MUST be used when a user provides a URL to learn from.
    """
    await cl.Message(content=f"Tool activated: `scrape_and_embed_website`. Processing {url}...").send()
    _setup_qdrant_collection()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        html_content = await page.content()
        await browser.close()

    soup = BeautifulSoup(html_content, "html.parser")
    texts = [p.get_text() for p in soup.find_all("p") if p.get_text().strip()]
    points = []

    for text in texts:
        embedding = await _get_text_embedding(text)
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"type": "text", "content": text}))

    for img_tag in soup.find_all("img"):
        img_url = img_tag.get("src")
        if img_url:
            try:
                full_img_url = requests.compat.urljoin(url, img_url)
                response = requests.get(full_img_url)
                response.raise_for_status()
                description = await _get_image_description(response.content)
                embedding = await _get_text_embedding(description)
                points.append(models.PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"type": "image", "content": description, "url": full_img_url}))
            except Exception as e:
                print(f"Could not process image {img_url}: {e}")
    
    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        return f"Successfully scraped and embedded {len(points)} items from {url} into the knowledge base."
    
    return f"No content was found to scrape from {url}."