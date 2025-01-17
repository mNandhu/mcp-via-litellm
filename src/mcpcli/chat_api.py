import os
import json
import asyncio
import contextlib
from typing import List, Optional
from mcpcli.config import load_config
from mcpcli.transport.stdio.stdio_client import stdio_client
from mcpcli.messages.send_initialize_message import send_initialize
from mcpcli.llm_client import LLMClient
from mcpcli.tools_handler import convert_to_openai_tools, fetch_tools, handle_tool_call
from mcpcli.chat_handler import generate_system_prompt

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_BLUE = "\033[94m"
ANSI_CYAN = "\033[96m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"


async def initialize_servers(config_path: str, server_names: List[str]) -> tuple:
    """Initialize connections to all servers and return the stream pairs."""
    server_streams = []
    context_managers = []

    for server_name in server_names:
        server_params = await load_config(config_path, server_name)
        cm = stdio_client(server_params)
        read_stream, write_stream = await cm.__aenter__()
        context_managers.append(cm)
        server_streams.append((read_stream, write_stream))

        init_result = await send_initialize(read_stream, write_stream)
        if not init_result:
            print(f"{ANSI_RED}Server initialization failed for {server_name}{ANSI_RESET}")
            return []

    return server_streams, context_managers


async def close_servers(context_managers: List):
    """Properly close all server connections."""
    for cm in context_managers:
            with asyncio.move_on_after(1):  # wait up to 1 second
                await cm.__aexit__()


async def process_conversation(
        client: LLMClient,
        conversation_history: List[dict],
        openai_tools: List[dict],
        server_streams: List[tuple]
) -> None:
    """Process the conversation loop, handling tool calls and responses."""
    while True:
        completion = client.create_completion(
            messages=conversation_history,
            tools=openai_tools,
        )

        response_content = completion.get("response", "No response")
        tool_calls = completion.get("tool_calls", [])

        if tool_calls:
            for tool_call in tool_calls:
                print(f"{ANSI_CYAN}Tool call: {tool_call}{ANSI_RESET}")
                await handle_tool_call(tool_call, conversation_history, server_streams)
            continue

        conversation_history.append({"role": "assistant", "content": response_content})
        break


async def chat_completion(
        server_streams: List[tuple],
        messages: Optional[List[dict]] = None,
        add_system_prompt: bool = False,
        provider: str = None,
        model: str = None
) -> List[dict]:
    """Handle a chat completion request with the given messages."""
    try:
        provider = provider or os.getenv("LLM_PROVIDER", "groq/llama-3.1-8b-instant")
        model = model or os.getenv("LLM_MODEL", provider.split("/")[-1])

        # Fetch and prepare tools
        tools = []
        for read_stream, write_stream in server_streams:
            tools.extend(await fetch_tools(read_stream, write_stream))

        openai_tools = convert_to_openai_tools(tools)
        client = LLMClient(provider=provider, model=model)

        # Initialize or update conversation history
        if messages is None or add_system_prompt:
            system_prompt = generate_system_prompt(tools)
            system_message = [{"role": "system", "content": system_prompt}]
            conversation_history = system_message + (messages or [])
        else:
            conversation_history = messages

        # Process the conversation
        await process_conversation(
            client, conversation_history, openai_tools, server_streams
        )
        return conversation_history

    except Exception as e:
        print(f"{ANSI_RED}Error in chat completion: {e}{ANSI_RESET}")
        return None


async def main():
    """Example usage of the chat API."""
    config_file = r"G:\Projects\mcp-via-litellm\server_config.json"
    with open(config_file, 'r') as f:
        servers = list(json.load(f)['mcpServers'].keys())

    server_streams = []
    context_managers = []
    try:
        # Initialize servers
        server_streams, context_managers = await initialize_servers(config_file, servers)

        # Initial message
        user_message = input(f"{ANSI_YELLOW}Enter your message: {ANSI_RESET}")
        if user_message.lower() == 'exit':
            # Okay, nothing works ... let's try the old way
            print('Forcing my way out!')
            raise KeyboardInterrupt

        # Start chat with system prompt
        chat_history = await chat_completion(
            server_streams,
            messages=[{"role": "user", "content": user_message}],
            add_system_prompt=True
        )
        if chat_history:
            print(f'{ANSI_BLUE}AI: {chat_history[-1]["content"]}{ANSI_RESET}')

        # Continue chat
        while True:
            user_message = input(f"{ANSI_YELLOW}Enter your message (or 'exit' to quit): {ANSI_RESET}")
            if user_message.lower() == 'exit':
                print(f"{ANSI_GREEN}Chat session ended.{ANSI_RESET}")
                break
            elif user_message.lower() == 'history':
                print(f"{ANSI_CYAN}{chat_history}{ANSI_RESET}")
                continue

            # Properly update chat history
            if chat_history:
                chat_history.append({"role": "user", "content": user_message})
                chat_history = await chat_completion(
                    server_streams,
                    messages=chat_history,
                )
                if chat_history:
                    print(f'{ANSI_BLUE}AI: {chat_history[-1]["content"]}{ANSI_RESET}')
        print('While loop exited')

    finally:
        # await stack.aclose()
        # await stack.__aexit__(None, None, None)
        # await stack.push(None)
        # stack.pop_all()
        # Okay, nothing works ... let's try the old way
        # await close_servers(context_managers)
        # print('Forcing my way out!')
        # raise KeyboardInterrupt
        # return
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("Program Closed Successfully")
    except KeyboardInterrupt:
        print(f"\n{ANSI_RED}Chat session terminated.{ANSI_RESET}")
