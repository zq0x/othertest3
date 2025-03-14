from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import time

app = FastAPI()

async def process_task(args, queue):
    await queue.put({"progress": "Starting task..."})
    await asyncio.sleep(1)

    for i in range(10):
        if queue.empty() == False:
          if queue.get() == "STOP":
            return
        await queue.put({"progress": f"Processing step {i + 1}...\n"})
        await asyncio.sleep(1)

    result = f"Task completed with args: {args}"
    await queue.put({"result": result})

async def generate_responses(queue):
    while True:
        item = await queue.get()
        yield json.dumps(item).encode('utf-8') + b'\n'

@app.post("/process")
async def process(request: Request):
    data = await request.json()
    args = data.get("args", [])
    queue = asyncio.Queue()

    task = asyncio.create_task(process_task(args, queue))
    return StreamingResponse(generate_responses(queue), media_type="text/event-stream")