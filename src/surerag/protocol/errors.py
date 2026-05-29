"""Protocol-level exceptions."""


class SureRAGError(Exception):
    """Base exception for SURE-RAG."""


class ArtifactError(SureRAGError):
    """Raised when an artifact cannot be loaded or saved."""
