from .base import Retriever


def build_retriever(kind: str = "faiss", **kwargs) -> Retriever:
    if kind == "faiss":
        from .faiss_retriever import FaissRetriever

        return FaissRetriever(**kwargs)
    raise ValueError(f"Unknown retriever kind: {kind!r}")


__all__ = ["Retriever", "build_retriever"]
