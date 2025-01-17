import os
import json
import anyio
from typing import List
from mcpcli.config import load_config
from mcpcli.transport.stdio.stdio_client import stdio_client
from mcpcli.messages.send_initialize_message import send_initialize
from mcpcli.llm_client import LLMClient
from mcpcli.system_prompt_generator import SystemPromptGenerator
from mcpcli.tools_handler import convert_to_openai_tools, fetch_tools, handle_tool_call

# Global variables to store server streams and context managers
server_streams = []
context_managers = []

async def initialize(config_path: str, server_names: List[str]) -> None:
    # Load server configurations and establish connections for all servers
    for server_name in server_names:
        server_params = await load_config(config_path, server_name)

        # Establish stdio communication for each server
        cm = stdio_client(server_params)
        (read_stream, write_stream) = await cm.__aenter__()
        context_managers.append(cm)
        server_streams.append((read_stream, write_stream))

        init_result = await send_initialize(read_stream, write_stream)
        if not init_result:
            print(f"[red]Server initialization failed for {server_name}[/red]")
            return


def close_streams():
    for cm in context_managers:
        # with anyio.move_on_after(1):  # wait up to 1 second
        cm.__aexit__()

def chat_completion(messages = None, add_system_prompt = False) -> List | None:
    """Main function to manage server initialization, communication, and shutdown."""

    try:
        provider = os.getenv("LLM_PROVIDER", "groq/llama-3.1-8b-instant")
        model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

        chat_info_text = (
            f"**Provider:** {provider}  |  **Model:** {model} |"
            "Type 'exit' to quit."
        )

        print(chat_info_text)
        tools = []
        for read_stream, write_stream in server_streams:
            tools.extend(fetch_tools(read_stream, write_stream))

        # if not tools:
        #     print("[red]No tools available. Exiting chat mode.[/red]")
        #     return

        openai_tools = convert_to_openai_tools(tools)
        client = LLMClient(provider=provider, model=model)
        # conversation_history = [{"role": "system", "content": system_prompt}]
        if messages is None or add_system_prompt:
            system_prompt_message = [{"role": "system", "content": generate_system_prompt(tools)}]
            messages = system_prompt_message + messages if messages is not None else system_prompt_message

        process_conversation(
            client, messages, openai_tools
        )
        return messages
    except Exception as e:
        print(f"Error in chat mode:{e}")




def process_conversation(
        client, conversation_history, openai_tools
):
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
                # Extract tool_name and raw_arguments as before
                if hasattr(tool_call, "function"):
                    tool_name = getattr(tool_call.function, "name", "unknown tool")
                    raw_arguments = getattr(tool_call.function, "arguments", {})
                elif isinstance(tool_call, dict) and "function" in tool_call:
                    fn_info = tool_call["function"]
                    tool_name = fn_info.get("name", "unknown tool")
                    raw_arguments = fn_info.get("arguments", {})
                else:
                    tool_name = "unknown tool"
                    raw_arguments = {}

                # If raw_arguments is a string, try to parse it as JSON
                if isinstance(raw_arguments, str):
                    try:
                        raw_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, just display as is
                        pass

                # Now raw_arguments should be a dict or something we can pretty-print as JSON
                tool_args_str = json.dumps(raw_arguments, indent=2)

                tool_md = f"**Tool Call:** {tool_name}\n\n```json\n{tool_args_str}\n```"
                print(f"{tool_md=!r}")

                handle_tool_call(tool_call, conversation_history, server_streams)
            continue

        # Assistant panel with Markdown
        assistant_panel_text = response_content if response_content else "[No Response]"
        # print(f"{assistant_panel_text=!r}")
        conversation_history.append({"role": "assistant", "content": response_content})
        break


def generate_system_prompt(tools):
    """
    Generate a concise system prompt for the assistant.

    This prompt is internal and not displayed to the user.
    """
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    system_prompt = prompt_generator.generate_prompt(tools_json)
    system_prompt += """

**GENERAL GUIDELINES:**

1. Step-by-step reasoning:
   - Analyze tasks systematically.
   - Break down complex problems into smaller, manageable parts.
   - Verify assumptions at each step to avoid errors.
   - Reflect on results to improve subsequent actions.

2. Effective tool usage:
   - Explore:
     - Identify available information and verify its structure.
     - Check assumptions and understand data relationships.
   - Iterate:
     - Start with simple queries or actions.
     - Build upon successes, adjusting based on observations.
   - Handle errors:
     - Carefully analyze error messages.
     - Use errors as a guide to refine your approach.
     - Document what went wrong and suggest fixes.

3. Clear communication:
   - Explain your reasoning and decisions at each step.
   - Share discoveries transparently with the user.
   - Outline next steps or ask clarifying questions as needed.

EXAMPLES OF BEST PRACTICES:

- Working with databases:
  - Check schema before writing queries.
  - Verify the existence of columns or tables.
  - Start with basic queries and refine based on results.

- Processing data:
  - Validate data formats and handle edge cases.
  - Ensure integrity and correctness of results.

- Accessing resources:
  - Confirm resource availability and permissions.
  - Handle missing or incomplete data gracefully.

REMEMBER:
- Be thorough and systematic.
- Each tool call should have a clear and well-explained purpose.
- Make reasonable assumptions if ambiguous.
- Minimize unnecessary user interactions by providing actionable insights.

EXAMPLES OF ASSUMPTIONS:
- Default sorting (e.g., descending order) if not specified.
- Assume basic user intentions, such as fetching top results by a common metric.
"""
    return system_prompt


if __name__ == "__main__":
    config_file = r"G:\Projects\mcp-via-litellm\server_config.json"
    with open(config_file, 'r') as f:
        servers = list(json.load(f)['mcpServers'].keys())
    anyio.run(initialize,config_file, servers)

    user_message = input("Enter your message: ")
    chat_history = chat_completion([{"role": "user", "content": user_message}],add_system_prompt=True)
    print('AI Message:', chat_history[-1]["content"])
    while user_message.lower() not in ['exit']:
        user_message = input("Enter your message: ")
        if user_message.lower() in ['history']:
            print(chat_history)
            continue
        chat_history = chat_completion([{"role": "user", "content": user_message}])
        print('AI Message:', chat_history[-1]["content"])

    close_streams()
