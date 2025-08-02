import os
import chainlit as cl
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, FunctionTool, RunContextWrapper, function_tool
import nest_asyncio

# Apply once at the very top
nest_asyncio.apply()

load_dotenv()



# --- Define your tool using FunctionTool ---
async def scrape_and_embed_website(url: str) -> str:
    """Scrapes a website and embeds its content into the knowledge base."""
    # TODO: Add your scraping and embedding logic here
    return f"Pretend scraped and embedded: {url}"

# Register the tool using FunctionTool
scrape_and_embed_website_tool = FunctionTool.from_function(scrape_and_embed_website)

# Set up the OpenAI client
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)




@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! I am a multimodal RAG agent. Provide a URL to scrape, or ask a question.").send()

@cl.on_message
async def main(message: cl.Message):
    loading_msg = cl.Message(content="Thinking...")
    await loading_msg.send()
    user_query = message.content
    # Create the agent with the tool
    agent = Agent(
        name="MultimodalRAGAgent",
        instructions="You are a multimodal RAG agent. Use the scrape_and_embed_website tool if a URL is provided.",
        tools=[scrape_and_embed_website_tool],
        model=client.chat.completions,
    )

    from agents import Runner
    result = await Runner.run(agent, user_query)
    await cl.Message(content=result.final_output).send()
    await loading_msg.remove()