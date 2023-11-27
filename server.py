import json
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from sse_starlette.sse import EventSourceResponse
from server_utils import *
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)


@app.post("/chat")
async def chat_endpoint(params: str = Form(...), img_file: UploadFile | None = File(...)):
    try:
        event_generator = chat_with_sydney(ChatWithSydneyParams(**json.loads(params)), img_file)
        return EventSourceResponse(event_generator)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
