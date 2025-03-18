from langgraph.graph import START, END, StateGraph
from typing import Annotated, TypedDict, List
from operator import add
from googlesearch import search

from agents.initial_agent import InitialAgent, SearchConfig
from agents.pages_search_agent import PagesSearchAgent
from agents.aggregation_agent import AggregationAgent


class SearchState(TypedDict):
    query: str
    search_config: SearchConfig
    rephrased_query: str
    uris: Annotated[List[str], add]
    source_knowledge_pairs: Annotated[List[tuple[str, str]], add]
    response: str


class SearchGraphFactory:

    def create(self):
        graph = StateGraph(SearchState)

        graph.add_node("initial_agent", InitialAgent().invoke)
        graph.add_node("web_pages_search_agent", PagesSearchAgent().invoke)
        graph.add_node("aggregation_agent", AggregationAgent().invoke)

        graph.add_edge(START, "initial_agent")
        graph.add_edge("initial_agent", "web_pages_search_agent")
        graph.add_edge("web_pages_search_agent", "aggregation_agent")
        graph.add_edge("aggregation_agent", END)

        return graph.compile()
