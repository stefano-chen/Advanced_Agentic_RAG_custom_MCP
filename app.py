from pathlib import Path
import json
from utils.processing import save_to_png, stream_response
from utils.agent import build_agent
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
import time
import asyncio

async def main():
    # Load environment variables from .env file
    load_dotenv("./.env")

    # Check configuration files existence
    if not Path("./config/app_config.json").exists():
        raise FileNotFoundError("app_config.json not Found")
    
    if not Path("./config/prompts.json").exists():
        raise FileNotFoundError("prompts.json not Found")

    with open("./config/app_config.json") as f:
        app_config = json.load(f)
    
    with open("./config/prompts.json") as f:
        prompts = json.load(f)

    # Create a In memory chat history
    chat_history = InMemoryChatMessageHistory()
    
    # Compile the graph to an agent
    start = time.time()
    agent = await build_agent(app_config, prompts)
    end = time.time()
    print(f"Agent compiled in {(end - start):.2f}s")

    # read some flags from config file
    verbosity = app_config.get("verbosity", 0)
    save_to_png_flag = app_config.get("save_to_png", False)
    image_name = app_config.get("image_name", "graph.png")

    # create a png representation of the agent/graph
    if save_to_png_flag:
        await save_to_png(agent, image_name)

    user_query = input("Enter: ")
    while user_query.lower() not in ["exit", "quit"]:
        history = "\n".join(msg.content for msg in chat_history.messages)
        chat_history.add_user_message(user_query)
        answer = await stream_response(agent, user_query, history, verbosity)
        chat_history.add_ai_message(answer)
        print(f"\n{'-'*36} Answer {'-'*36}\n{answer}")
        user_query = input("Enter: ")

if __name__ == "__main__":
    asyncio.run(main())