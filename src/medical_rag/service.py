"""Production dependency construction."""

from .config import Settings
from .embeddings import SentenceTransformerEmbedder
from .search import SearchEngine
from .storage import CorpusStore


def create_engine(settings: Settings) -> SearchEngine:
    store = CorpusStore(settings.database)
    embedder = SentenceTransformerEmbedder(
        settings.model,
        settings.model_revision,
        local_only=settings.local_model_only,
    )
    return SearchEngine(store, embedder)
