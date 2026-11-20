import os
from dotenv import load_dotenv
load_dotenv()
from trip_agent import graph, SYSTEM

config = {"configurable": {"thread_id": "test-cli"}, "recursion_limit": 30}
question = "What is the capital of France?"
result = graph.invoke({"messages": [("user", question)]}, config)
print("Answer:", result["messages"][-1].text)
