from googlesearch import search
from pydantic import BaseModel

from chains.analysis_chain import AnalysisChain


class SearchConfig(BaseModel):
    pages_count: int
    language: str


class InitialAgent:

    def __init__(self):
        self.analysis_chain = AnalysisChain().create()


    def invoke(self, state):
        config = state["search_config"]
        uris = search(state["query"], num=config.pages_count, stop=config.pages_count, lang=config.language, pause=1)
        state = state | {"language": config.language}
        return self.analysis_chain.invoke(state) | {
            "uris": list(uris)
        }
