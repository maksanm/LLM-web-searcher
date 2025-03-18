from dotenv import load_dotenv

from graph import SearchConfig, SearchGraphFactory

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time


class UserRequest(BaseModel):
    query: str
    search_config: SearchConfig


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = SearchGraphFactory().create()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.post("/")
async def ask(request: UserRequest):
    start_time = time.time()
    initial_state = {
        "query": request.query,
        "search_config": request.search_config,
    }
    result = graph.invoke(initial_state)
    print("--- %s seconds ---" % (time.time() - start_time))
    return {"execution_time": time.time() - start_time} | result


if __name__ == "__main__":
    uvicorn.run(app)
