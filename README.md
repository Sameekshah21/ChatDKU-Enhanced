# ChatDKU — Enhanced Local Agent and Student Learning Prototype

This project builds on [Edge Intelligence Lab's ChatDKU-Public](https://github.com/Edge-Intelligence-Lab/ChatDKU-Public), a **local-first, open-source agent harness** for answering questions using an operator-supplied set of Duke Kunshan University (DKU) documents. It is **not** the production ChatDKU website or backend.

The original repository's ingestion pipeline, local Qdrant document index, semantic/vector retrieval, keyword retrieval, and `agent` command remain available. The additional Python modules in this submission explore evidence selection, response generation, and student-facing learning workflows.

> **Project status:** Working local prototype, not a production service. The enhanced command and Streamlit interface use an additional generation path; **the original `uv run agent` command has not been replaced with the new pipeline**. Performance and accuracy improvements should not be treated as proven without a controlled evaluation.

## What was added

### 1. Evidence-oriented retrieval (`chatdku_improvement1_auto_v6.py`)

- Expands some student questions into a small number of focused retrieval queries using topic-based rules.
- Calls the **existing** ChatDKU vector and keyword retrievers; these two retrievers are part of the original project, not newly invented tools.
- Merges ranked results using reciprocal rank fusion, removes duplicate chunks, and selects passages with a heuristic preference for covering different subqueries.
- Builds numbered evidence excerpts with document names and chunk identifiers and limits the total evidence context sent to the model.

These changes aim to make retrieved evidence more varied and relevant. Heuristic query coverage does **not** guarantee that selected passages answer every part of a question.

### 2. Adaptive, source-grounded generation (`chatdku_improvement1_auto_v6.py`)

- Chooses smaller evidence and output budgets for straightforward questions and larger budgets for more complex questions.
- Requests student-facing answers with numbered citations, rather than long deliberative text.
- Uses a direct local Ollama-compatible generation request with thinking disabled when supported.
- Checks for an output-length stop or unusable answer and retries **once** with a larger generation budget.
- Offers an optional `--audit` pass for an experimental answer review; this is **not** independent factual verification.

Retrying may improve completeness, but cannot guarantee that an answer is comprehensive or correct. Citation markers and source listings identify retrieval evidence; they do not automatically prove that each claim is supported.

### 3. Streamlit student experience (`chatdku_experience_v6.py`)

The optional interface provides three tabs:

| Tab | Functions |
| --- | --- |
| **Ask DKU** | Ask policy questions against the configured official-document index; inspect retrieved source passages. |
| **Learn from my materials** | Upload PDF, DOCX, PPTX, or TXT files; ask questions, request step-by-step lessons, generate practice quizzes, and create revision plans. |
| **Academic roadmap** | Enter year, intended major, courses and goals; generate source-assisted planning drafts and explore what-if scenarios. |

Uploaded course files are parsed into text chunks and searched separately using a lightweight lexical/BM25-style ranking method. They are not automatically ingested into the main official DKU Qdrant collection. Scanned/image-only PDFs require OCR, which this prototype does not include. The roadmap is **not** a registrar-verified degree audit, course-prerequisite checker, or guarantee of graduation eligibility.

## Repository layout

```text
src/chatdku/                        Original ChatDKU agent, ingestion and retrieval code
  agent.py                          Original conversational agent workflow
  tools/                            Original vector/keyword retrievers
  ingestion/                        Document ingestion pipeline
chatdku_improvement1_auto_v6.py     Experimental enhanced retrieval/generation command
chatdku_experience_v6.py            Student-facing Streamlit prototype
chatdku_experience_requirements.txt Website-specific dependencies
README.md                           This guide
docs/                               Original agent and ingestion documentation
```

There are **no benchmark scripts or benchmark graphs included in this ZIP**; any experimental timing results should be reported separately with their raw outputs and evaluation criteria. In particular, preliminary answer-detection tests had a parsing issue for unlabeled original-agent answers and should not be represented as a valid completion-rate comparison.

## Prerequisites

- Windows, macOS, or Linux, with **Python 3.13+** (as required by `pyproject.toml`).
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
- A compatible local or hosted language-model endpoint. The enhanced v6 path is designed for **Ollama**; the example below assumes Ollama and the `qwen3:4b` model.
- A local embedding model or a configured embeddings endpoint for ingestion and retrieval.
- The DKU source documents that you are authorized to use. The upstream repository does not provide a hosted model, source documents, or API credentials.

## Quick start (Windows Command Prompt)

Run all commands from the repository root (the folder containing `pyproject.toml`).

**1. Install/synchronize project dependencies:**

```cmd
uv sync
```

**2. Configure the local model:**

```cmd
copy .env.example .env
ollama pull qwen3:4b
ollama serve
```

If Ollama is already running, you do not need to start another `ollama serve` process. In `.env`, use:

```dotenv
CHATDKU_LLM_MODEL=qwen3:4b
CHATDKU_LLM_BASE_URL=http://localhost:11434/v1
CHATDKU_EMBEDDING_PROVIDER=local
CHATDKU_LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
CHATDKU_DATA_DIR=./data
CHATDKU_VECTOR_COLLECTION=chatdku
```

Keep `.env` private. `.env.example` documents configuration without supplying secrets. The enhanced v6 path relies on an Ollama-compatible chat endpoint; not every generic OpenAI-compatible server supports the same options.

**3. Add authorized source documents** to `data/documents/`, then check embeddings and build the local index:

```cmd
uv run agent --check-embeddings
uv run chatdku-ingest
```

If the index has already been built from the same documents and embedding model, ingestion is not needed again. If you change embedding models or source documents, see [`docs/ingestion.md`](docs/ingestion.md) for the deliberate `--replace` procedure. **Do not** rebuild blindly because this deletes and recreates the existing collection.

**4. Test the original agent:**

```cmd
uv run agent "What is the role of an academic advisor at DKU?"
```

**5. Test the experimental enhanced v6 command:**

```cmd
uv run python chatdku_improvement1_auto_v6.py "What is the role of an academic advisor at DKU?"
```

Optional experimental audit (slower):

```cmd
uv run python chatdku_improvement1_auto_v6.py --audit "What is the role of an academic advisor at DKU?"
```

**6. Run the Streamlit interface:**

```cmd
uv run --with streamlit --with pypdf --with python-pptx --with python-docx streamlit run chatdku_experience_v6.py
```

Open the local URL displayed by Streamlit (typically `http://localhost:8501`). The **Ask DKU** tab needs the local official-document index; the uploaded-material tutor uses a separate session-local collection of parsed text chunks. The interface imports the enhanced v6 module directly.

For macOS/Linux, use `cp .env.example .env` instead of the Windows `copy` command. Follow Ollama's installation instructions for your operating system.

## Evaluation and reproducibility

For a meaningful original-versus-enhanced comparison:

1. Use the **same questions, source-document index, machine, and model** for both implementations.
2. Save full raw answers and failures. Do not equate a process exit code of zero with a completed answer.
3. Check that both answer formats are parsed correctly. The original CLI may return an unlabeled paragraph, whereas v6 prints an `ANSWER:` section.
4. Record wall-clock latency, whether the answer genuinely completes, and whether it reaches the output limit.
5. Manually verify factual claims and whether cited passages actually support them; citation count or answer length is **not** accuracy.
6. Include failures, limitations and hardware constraints in the final report.

Initial local experiments identified verbose model reasoning and output-limit truncation in an earlier enhanced prototype. v6 attempts to address these, but no reproducible accuracy gain or speedup is claimed here.

## Limitations and safety

- The standalone enhanced script and the Streamlit app **do not replace the original `agent` CLI implementation**. Further work is necessary to integrate and test enhancements in the original agent entry point.
- The simple/complex budget classifier and query expansion rules are heuristic; some questions may be misclassified or incompletely retrieved.
- A valid citation marker does not guarantee factual entailment, and source excerpts can be truncated.
- AI-generated academic plans are drafts only. Verify prerequisites, policies, course offerings and degree requirements with official DKU publications and advisors.
- The website is a local prototype without production authentication, a multi-user permissions system or guaranteed secure retention of uploaded documents. Do not upload sensitive private student records.
- Runtime depends heavily on the local hardware, model and indexed document set. CPU-only inference may be slow.

## Original documentation and attribution

This work is based on the Edge Intelligence Lab's **ChatDKU-Public** repository. Its original modules for ingestion, vector and keyword retrieval, and agent orchestration remain credited to the upstream project. See the original [`docs/agent.md`](docs/agent.md) and [`docs/ingestion.md`](docs/ingestion.md) for details of the unmodified workflows.


## Integrated enhanced agent (candidate improvement)

The enhanced agent is integrated into the original `src/chatdku/` package. The
existing `agent` CLI remains the baseline; `--enhanced` opts into the enhanced
retrieval and generation path. Both modes use the existing local document index.

```powershell
uv sync
# Run local Ollama and configure .env following the baseline setup above.
# Ingest your own authorized DKU source documents following docs/ingestion.md.
uv run agent "What is the role of an academic advisor at DKU?"
uv run agent --enhanced "What is the role of an academic advisor at DKU?"
```

Implementation: `src/chatdku/enhanced.py`; integration: `src/chatdku/agent.py`
and `src/chatdku/cli.py`. The backward-compatible root script
`chatdku_improvement1_auto_v6.py` imports from the same implementation.

The optional Streamlit prototype also shares the same enhanced retrieval module:

```powershell
uv run --with streamlit --with pypdf --with python-pptx --with python-docx streamlit run chatdku_experience_v6.py
```

**Improvement 1:** multi-query vector/keyword retrieval, reciprocal-rank fusion,
deduplication, and query-coverage-aware passage selection. **Improvement 2:**
adaptive evidence and output budgets, structured answer generation, and one
retry on an apparent truncated response. These are implementation mechanisms,
not independently established accuracy gains.

**Validation and limitations:** Syntax checks can be performed with
`uv run python -m compileall src chatdku_experience_v6.py`. Run both CLI modes
on an ingested local index and compare outputs against the cited documents;
reliability and factual accuracy must be manually verified. The enhanced path
requires a local **Ollama `/api/chat`** endpoint and an existing document index;
other OpenAI-compatible backends are supported by the original agent, not this
enhancement. References are diagnostic, not proof of factual support.
The enhanced direct-response path does not currently preserve conversation
context across questions, unlike the baseline multi-turn DSPy pipeline.
The Streamlit application is an additional demonstration, not a replacement
for the integrated CLI enhancements.
