from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from trip_agent import graph, SYSTEM

load_dotenv()

app = FastAPI(title="Trip Budget Planner API")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

CONFIG = {"configurable": {"thread_id": "trip-web"}, "recursion_limit": 30}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = graph.invoke({"messages": [("user", req.question)]}, CONFIG)
    answer = result["messages"][-1].text
    return AskResponse(answer=answer)
