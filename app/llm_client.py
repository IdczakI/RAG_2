import os
from langchain_openai import ChatOpenAI


class OpenAIClient:
    def __init__(self):
        model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
        self.llm = ChatOpenAI(model=model, temperature=0.0, )

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content
