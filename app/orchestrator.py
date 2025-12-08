from langchain_core.runnables import Runnable
from .llm_client import OpenAIClient


class QAOrchestrator:
    TEMPLATE = (
        "Answer ONLY using the context below.\n"
        "If answer is not in context, say: 'No relevant information found.'\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    def __init__(self, retriever: Runnable, llm: OpenAIClient):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str):
        docs = self.retriever.invoke(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        prompt = self.TEMPLATE.format(context=context, question=question)
        answer = self.llm.generate(prompt)
        return dict(answer=answer, source_documents=docs)
