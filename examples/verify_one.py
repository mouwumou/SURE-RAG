from surerag import SureRAGVerifier

verifier = SureRAGVerifier.toy()
result = verifier.verify(
    {
        "question": "Who wrote The Hobbit?",
        "answer": "J.R.R. Tolkien",
        "evidence": [{"id": "e1", "text": "The Hobbit was written by J.R.R. Tolkien."}],
    }
)
print(result.model_dump_json(indent=2))
