from langchain_core.runnables import Runnable
from langchain_chroma import Chroma


class RetrieverWrapper:

    def __init__(self, vectordb: Chroma, k: int = 6):
        self.retriever = vectordb.as_retriever(search_kwargs={"k": k})

    def as_retriever(self) -> Runnable:
        return self.retriever
