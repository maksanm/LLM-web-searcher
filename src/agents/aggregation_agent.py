from chains.aggregation_chain import AggregationChain


class AggregationAgent:

    def __init__(self):
        self.aggregation_chain = AggregationChain().create()

    def invoke(self, state):
        return self.aggregation_chain.invoke(state)

