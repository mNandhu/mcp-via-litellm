import sys
import json
import asyncio
import requests
from typing import List, Dict
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

SERVER_URL = "http://localhost:8132"

def send_chat_request(messages: List[Dict], add_system_prompt: bool = False) -> List[Dict]:
    """Send a chat request to the server and return the response."""
    try:
        # Ensure messages are in the correct format
        formatted_messages = [
            {
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            }
            for msg in messages
        ]
        
        payload = {
            "messages": formatted_messages,
            "add_system_prompt": add_system_prompt,
            "provider": "openai/gpt-3.5-turbo",  # You can make these configurable
            "model": None
        }
        
        response = requests.post(
            f"{SERVER_URL}/chat",
            json=payload
        )
        response.raise_for_status()
        return response.json()["messages"]
    except requests.exceptions.RequestException as e:
        print(f"[red]Error communicating with server:[/red] {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"[red]Response details:[/red] {e.response.text}")
        return None

def main():
    print(Panel(
        Markdown("# Welcome to MCP Chat Client\n\nType 'exit' to quit, 'clear' to clear history"),
        style="bold blue"
    ))

    # Initialize conversation history
    conversation_history = []
    
    try:
        while True:
            # Get user input
            user_message = Prompt.ask("[bold yellow]You[/bold yellow]")
            
            # Handle commands
            if user_message.lower() == "exit":
                print("[bold red]Goodbye![/bold red]")
                break
            elif user_message.lower() == "clear":
                conversation_history = []
                print("[bold green]History cleared![/bold green]")
                continue

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_message})
            
            # Send request to server
            add_system_prompt = len(conversation_history) == 1  # Only for first message
            updated_history = send_chat_request(
                conversation_history[-1:] if not add_system_prompt else conversation_history,
                add_system_prompt
            )
            
            if updated_history:
                # Update conversation history while preserving the structure
                if add_system_prompt:
                    conversation_history = updated_history
                else:
                    conversation_history.extend(updated_history[len(conversation_history):])
                    
                # Display AI response
                ai_message = updated_history[-1]["content"]
                print(Panel(
                    Markdown(ai_message),
                    title="[bold blue]Assistant[/bold blue]",
                    style="bold white"
                ))
            else:
                print("[red]Failed to get response from server[/red]")
                conversation_history.pop()  # Remove failed message from history

    except KeyboardInterrupt:
        print("\n[bold red]Chat session terminated.[/bold red]")
    except Exception as e:
        print(f"[red]An error occurred:[/red] {str(e)}")

if __name__ == "__main__":
    main()
