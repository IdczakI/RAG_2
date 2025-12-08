## Module descriptions

---

### `loader.py` (PDFLoader)
Loads PDF files and converts them into page-level `Document` objects using `pypdf`.

Steps:
1. Reads the directory path from constructor or environment (`DOCS_DIR`).
2. Verifies the directory exists.
3. Iterates through all `.pdf` files in sorted order.
4. For each PDF:
   - Uses `pypdf.PdfReader` to read the file.
   - Extracts text from each page with `page.extract_text()` (fallback to empty string if None).
5. For each loaded page:
   - Builds `metadata` including `source` (filename) and `page_number`.
   - Creates a LangChain `Document(page_content=text, metadata=metadata)`.
   - Appends the page Document to the output list.
6. Returns a list of page-level `Document` objects.

---

### `splitter.py` (SectionSplitter)
Splits pages into smaller overlapping text chunks.

Steps:
1. Stores splitter configuration (`chunk_size`, `chunk_overlap`) in the constructor.
2. In `split_page()`:
    - Splits the page text into lines.
    - Accumulates lines into a buffer until a section boundary is reached.
    - Creates section-level documents when a boundary is detected.
    - Handles final buffered section.
3. Creates a `RecursiveCharacterTextSplitter`.
4. Splits each section document into final chunks.
5. In `split_documents()`:
    - Iterates through all pages.
    - Splits each page using `split_page()`.
    - Collects all chunks into a single list.
6. Returns the list of chunk documents ready for embedding.

---

### `embeddings.py` (EmbeddingManager)
Creates and exposes the embedding function.

Steps:
1. Reads embedding model name from environment (`EMBEDDING_MODEL`) or uses a default.
2. In `get_embedding_fn()`:
    - Instantiates `OpenAIEmbeddings` with the selected model.
    - Returns the embedding function object for vector encoding.

---

### `chroma_manager.py` (ChromaDBManager)
Builds or loads the persistent Chroma vector store.

Steps:
1. Constructor:
    - Determines the persistence directory from argument or environment (`CHROMA_PERSIST_DIR`).
    - Stores the collection name.
2. In `build_or_load()`:
    - Checks if the persistence directory exists and is non-empty.
        - If yes, loads the existing Chroma database and returns it.
    - If not:
        - Prints a creation message.
        - Calls `Chroma.from_documents()` to embed the documents.
        - Stores them in the persistent database directory.
    - Returns the Chroma vector store instance.

---

### `retriever.py` (RetrieverWrapper)
Wraps the Chroma vector store into a retriever interface.

Steps:
1. Constructor:
   - Accepts the Chroma instance (from `langchain_chroma` / `ChromaDBManager`).
   - Calls `vectordb.as_retriever(search_kwargs={"k": k})` to configure top-k retrieval.
2. `as_retriever()` returns this configured retriever (a LangChain `Runnable`) for use by the orchestrator.

---

### `llm_client.py` (OpenAIClient)
Lightweight wrapper around the OpenAI chat model.

Steps:
1. Constructor:
    - Reads the model name from environment (`OPENAI_MODEL`) or uses a default.
    - Instantiates `ChatOpenAI` with temperature `0.0` for deterministic responses.
2. `generate(prompt)`:
    - Calls `self.llm.invoke(prompt)`.
    - Returns only `.content` to keep output clean and free of metadata.

---

### `orchestrator.py` (QAOrchestrator)
Coordinates the full RAG pipeline: retrieval → prompt-building → LLM answer.

Steps:
1. Defines a static prompt template with instructions and placeholders.
2. Constructor:
    - Receives the retriever and LLM client.
3. `answer(question)`:
    - Sends the question to the retriever.
    - Receives top relevant documents (chunks).
    - Joins their content into a `context` string.
    - Fills the template with the context + question.
    - Calls the LLM via `llm.generate(prompt)`.
    - Returns a dictionary with:
        - `"answer"` — final LLM response
        - `"source_documents"` — documents retrieved for context

---

### `cli.py`
The command-line entry point. It initializes all components and runs the interactive Q&A loop.

Steps:
1. Imports all pipeline modules.
2. Defines `build_or_load_index()`:
    - Creates instances of loader, splitter, embedding manager, and Chroma manager.
    - If `--force-rebuild` is set, removes the existing Chroma directory.
    - If a persisted Chroma database exists and is not empty, loads it.
    - Otherwise:
        - Loads all PDFs.
        - Splits documents into chunks.
        - Builds a new Chroma vector database and saves it.
    - Returns the vector database instance.
3. Defines `main()`:
    - Parses CLI arguments.
    - Builds/loads the index.
    - Creates retriever, LLM client, and orchestrator.
    - Prints startup message.
    - Enters an infinite `while` loop reading user questions.
    - For each question:
        - Calls `orchestrator.answer()`.
        - Prints the final answer.
        - Prints source documents used for retrieval.
4. Executes `main()` when run as a script.

---

## Test module description

### `test_e2e_questions.py`

This module verifies that the entire RAG pipeline works correctly on real content extracted from PDF documents.
It is not a unit-test of individual components — it exercises the system end-to-end:

### The test step by step:

1. Load question scenarios
   - Reads all test cases from tests/questions.json.
   - Each entry contains:
     - id — test identifier
     - question — natural language query sent to the system
     - expected_keywords — substrings required to consider the answer correct
2. Load an existing vector database
   - Opens the previously persisted Chroma DB from ./chroma_db
   - The DB must already exist before running the tests
   - No new embedding generation or indexing occurs during these tests
3. Initialize a retriever
   - Uses ChromaDB as the source of context
   - Retrieves the top-K most relevant text chunks based on semantic similarity
4. Initialize the LLM client
   - Uses OpenAI model defined via environment variable
   (OPENAI_MODEL, typically gpt-4o-mini or similar)
   - Requires OPENAI_API_KEY to be set
5. Perform a full RAG query
   - Sends the natural language question into the RAG pipeline
   - Retrieves context from the database
   - Builds the final prompt
   - Generates an answer using the LLM
6. Evaluate correctness
   - The answer text is inspected for at least one of the expected_keywords
   - Matching is case-insensitive and substring-based
   - If a keyword is found → test passes
   - If none are found → test fails
7. Test execution is conditional
   - Tests are automatically skipped if:
     - OPENAI_API_KEY is not configured
     - The chroma_db directory does not exist or is empty

### Purpose
- Validates that the system can correctly extract information from the indexed PDF
- Ensures embedding search retrieves the relevant document fragments
- Confirms the LLM is properly constrained by the retrieved context
- Detects problems in chunking, embeddings compatibility, or model hallucinations

### Notes for usage
- This module is intended for integration checks — not for CI/CD without a budget
- Each test calls the live OpenAI API → billing applies
- Best used during development after the first successful RAG build
- If chunking strategy or embedding model changes, tests should be rerun