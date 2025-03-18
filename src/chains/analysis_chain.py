from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel


class AnalysisChain:
    REPHRASE_PROMPT_TEMPLATE = """
Enhance the query below by including synonyms, related phrases, and typical webpage content relevant to the original query. Avoid repeating terms, maintain clarity, and preserve proper nouns and specific names. Provide the expanded query in {language}, omitting articles and low-value words, so it is best suited for web page content similarity searches.

Query to expand: {query}
Result:
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return RunnableParallel({
            "rephrased_query": PromptTemplate.from_template(self.REPHRASE_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        })
