from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel


class ExtractionChain:
    EXTRACTION_PROMPT_TEMPLATE = """You are an extractor of knowledge from the raw data found on the Internet at the following source:
```json
{{
    "source": "{source}",
    "raw_webpage_data": "{markdown}"
}}
```

Extract the knowledge related to and potentially helpful for answering the following query:
```json
{{
    "query": "{query}"
}}
```

Please save any helpful links if they are available and different from the source root link.

Extracted knowledge:
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return RunnableParallel({
            "knowledge": PromptTemplate.from_template(self.EXTRACTION_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        })
