import os
import json
import pytest

from app.chroma_manager import ChromaDBManager
from app.embeddings import EmbeddingManager
from app.retriever import RetrieverWrapper
from app.llm_client import OpenAIClient
from app.orchestrator import QAOrchestrator

# Path to questions file (relative to project root)
TEST_DIR = os.path.dirname(__file__)
QUESTIONS_FILE = os.path.join(TEST_DIR, "questions.json")

# Skip whole module if OpenAI key missing
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping end-to-end tests that require OpenAI."
)


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def orchestrator():
    """
    Build orchestrator but reuse existing persistent Chroma DB in project root (chroma_db).
    This fixture does NOT rebuild the DB; it assumes chroma_db exists and is populated.
    """
    # Use the persistent directory (default from ChromaDBManager env)
    chroma_mgr = ChromaDBManager()  # uses CHROMA_PERSIST_DIR or default "chroma_db"
    # We do not build a new DB; instead load existing. To ensure loading path, call build_or_load with empty docs.
    emb_mgr = EmbeddingManager()

    # If the chroma persist dir does not exist or is empty, fail early with clear message
    persist_dir = chroma_mgr.persist_dir
    if not os.path.exists(persist_dir) or not any(os.scandir(persist_dir)):
        pytest.skip(f"Persistent Chroma DB not found or empty at '{persist_dir}'. Populate chroma_db before running E2E tests.")

    vectordb = chroma_mgr.build_or_load([], emb_mgr.get_embedding_fn())
    retriever = RetrieverWrapper(vectordb).as_retriever()
    llm = OpenAIClient()
    orch = QAOrchestrator(retriever, llm)
    return orch


def normalize_text(s):
    if s is None:
        return ""
    return s.lower()


def any_keyword_in_text(keywords, text):
    text_l = normalize_text(text)
    for k in keywords:
        if k.lower() in text_l:
            return True
    return False


questions = load_questions(QUESTIONS_FILE)


@pytest.mark.parametrize("qitem", questions, ids=[str(q["id"]) for q in questions])
def test_question_e2e(orchestrator, qitem):
    """
    End-to-end test: send question to orchestrator (retriever + real LLM),
    assert that answer contains at least one expected keyword and that source_documents is non-empty.
    """
    question = qitem["question"]
    expected_keywords = qitem.get("expected_keywords", [])
    assert expected_keywords, "Test config error: expected_keywords required"

    res = orchestrator.answer(question)
    assert isinstance(res, dict), "orchestrator.answer must return a dict"
    answer = res.get("answer") or ""
    sources = res.get("source_documents") or []

    # Basic checks
    assert len(sources) > 0, "No source_documents returned by retriever"
    # Combine answer and the combined source texts for better matching
    combined_text = answer + " " + " ".join(getattr(d, "page_content", "") for d in sources)

    # Check if any of expected keywords is present in combined_text
    found = any_keyword_in_text(expected_keywords, combined_text)

    # If not found, provide debug info in assertion message
    assert found, (
        f"Expected one of keywords {expected_keywords} not found for question: {question}\n"
        f"Answer: {answer}\n"
        f"Sources count: {len(sources)}\n"
        f"First source snippet: {getattr(sources[0], 'page_content', '')[:300]}"
    )
