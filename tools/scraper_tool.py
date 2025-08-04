import os
import base64
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from qdrant_client import QdrantClient, models
from agents import function_tool
# from langfuse import Langfuse # <-- IMPORT LANGFUSE
# from langfuse.openai import AsyncOpenAI
import uuid
import chainlit as cl
import google.generativeai as genai
import numpy as np
# ==============================================================================
# CORRECTED: INITIALIZE LANGFUSE AND THE PATCHED CLIENT FOR THE TOOL
# ==============================================================================
# langfuse = Langfuse(
# public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
# secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
# host=os.environ.get("LANGFUSE_HOST"),
# )


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
 genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
 result = genai.embed_content(
 model="embedding-001",
 content=text,
 task_type="semantic_similarity"
 )
 embedding = np.array(result['embedding']) / np.linalg.norm(result['embedding'])
 return embedding.tolist()

async def _get_image_description(image_bytes: bytes):
 genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
 model = genai.GenerativeModel('gemini-2.0-flash-exp')

 response = model.generate_content([
 "Describe this image in detail for a retrieval system.",
 {"mime_type": "image/jpeg", "data": image_bytes}
 ])

 return response.text

@function_tool
async def scrape_and_embed_website(url: str) -> str:
 """
 Scrapes a website's text and images, creates embeddings, and stores them in the knowledge base.
 This tool MUST be used when a user provides a URL to learn from.
 """
 # await cl.Message(content=f"Tool activated: `scrape_and_embed_website`. Processing {url}...").send()
 _setup_qdrant_collection()

 async with async_playwright() as p:
     browser = await p.chromium.launch()
     page = await browser.new_page()
     await page.goto(url)
     html_content = await page.content()
     print('paywright_html_content', html_content)
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