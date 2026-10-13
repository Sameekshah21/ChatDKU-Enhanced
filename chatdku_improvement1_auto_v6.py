"""Backward-compatible entry point for the enhanced ChatDKU agent.

The implementation now lives in `src/chatdku/enhanced.py` so both the original
CLI (with --enhanced) and Streamlit share exactly the same retrieval logic.
"""
from chatdku.enhanced import (
    source_name, query_plan, retrieve, build_evidence, ollama, final_content,
    automatic_budget, answer_question, SYSTEM, AUDIT_SYSTEM, main,
)

if __name__ == "__main__":
    main()
