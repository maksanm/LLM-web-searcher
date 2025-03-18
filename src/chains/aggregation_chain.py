from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


class AggregationChain:
    AGGREGATION_PROMPT_TEMPLATE = """You are an aggregator of knowledge found on the Internet for the following query:
```json
{{
    "query": "{query}"
}}
```

Compile the raw information gathered from various sources to craft an article that succinctly summarizes all pertinent details while addressing the query. Use the unstructured data provided to construct a coherent narrative. Ensure that references for each source cited in your final response are listed at the end. Incorporate numbered citations within the text according to the format illustrated below:
```
1. [Title of link1 from source 1](https://link1.com)
2. [Title of link2 from source 1](https://link2.com)
3. [Title of link1 from source 2](https://link1.com)
...
```

For sources with unstructured knowledge:
```
{formatted_source_knowledge_pairs}
```

Provide the aggregated knowledge in the language of the query, ensuring clarity and conciseness.
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return RunnableParallel({
            "response": RunnablePassthrough().assign(
                formatted_source_knowledge_pairs=self._format_source_knowledge_pairs
            )
            | PromptTemplate.from_template(self.AGGREGATION_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        })

    def _format_source_knowledge_pairs(self, state):
        formatted_entries = []
        for source, knowledge in state["source_knowledge_pairs"]:
            formatted_entry = (
                f"Source: {source}\nKnowledge: {knowledge}\n"
            )
            formatted_entries.append(formatted_entry)
        return "\n---\n".join(formatted_entries)
