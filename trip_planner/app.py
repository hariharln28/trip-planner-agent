from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from trip_agent import graph, SYSTEM

load_dotenv()

app = FastAPI(title="Trip Budget Planner API")

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

CONFIG = {"configurable": {"thread_id": "trip-web"}, "recursion_limit": 30}

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(static_path, "index.html")
    return FileResponse(index_path, media_type="text/html")

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = graph.invoke({"messages": [("user", req.question)]}, CONFIG)
    answer = result["messages"][-1].text
    return AskResponse(answer=answer)
