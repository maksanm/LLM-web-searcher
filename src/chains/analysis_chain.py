from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel


class AnalysisChain:
    REPHRASE_PROMPT_TEMPLATE = """Enhance the query below by inserting relevant synonyms, context details, and common web page terminology related to the original topic. Avoid repetition, preserve clarity, retain key proper names, and provide the result in {language}.

Example of Input Query and Transformation:
Input Query: new smartphone releases
Expanded Query: latest smartphone models, upcoming phone launches, specs comparisons, mobile innovations, brand announcements

Input Query: {query}
Expanded Query: """

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return RunnableParallel({
            "rephrased_query": PromptTemplate.from_template(self.REPHRASE_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        })
