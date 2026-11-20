"""Simple CLI wrapper around the Trip Budget Planner agent.

Usage:
    python agent.py          # starts an interactive REPL
"""
import os
from dotenv import load_dotenv
from trip_agent import graph, SYSTEM

load_dotenv()  # pulls GEMINI_API_KEY from .env

def repl():
    config = {"configurable": {"thread_id": "trip-cli"}, "recursion_limit": 30}
    print("Trip Budget Planner ready. Type a question, or 'quit' to exit.")
    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() == "quit":
            break
        result = graph.invoke({"messages": [("user", question)]}, config)
        reply = result["messages"][-1].text
        print("\nAgent:", reply)

if __name__ == "__main__":
    repl()
