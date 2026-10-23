import hashlib
import json
import logging
import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
COLLECTION_NAME = "acs880_firmware_manual"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def resolve_pdf_path(project_dir: Path = PROJECT_DIR) -> Path:
    configured_path = os.getenv("PDF_PATH")
    if configured_path:
        pdf_path = Path(configured_path).expanduser()
        if not pdf_path.is_absolute():
            pdf_path = project_dir / pdf_path
        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF_PATH does not point to an existing file: {pdf_path}"
            )
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF_PATH must point to a PDF file: {pdf_path}")
        return pdf_path.resolve()

    candidates = [
        path
        for path in project_dir.glob("*.pdf")
        if "acs880" in path.name.lower()
        and re.search(r"firmware|(?:^|[_\s-])fw(?:[_\s.-]|$)", path.name, re.I)
    ]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            "Could not find an ABB ACS880 firmware manual PDF in the project "
            "directory. Set PDF_PATH in .env to the manual's path."
        )
    raise ValueError(
        "More than one ABB ACS880 firmware manual PDF was found. "
        "Set PDF_PATH in .env to the intended manual."
    )


def _validate_pdf(pdf_path: Path) -> None:
    try:
        with pdf_path.open("rb") as pdf_file:
            if b"%PDF-" not in pdf_file.read(1024):
                raise ValueError(f"The configured file is not a readable PDF: {pdf_path}")
    except OSError as exc:
        raise OSError(f"Unable to read the configured PDF at {pdf_path}: {exc}") from exc


def load_and_split_pdf(
    pdf_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be a positive integer.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")
    _validate_pdf(pdf_path)
    try:
        pages = PyPDFLoader(str(pdf_path)).load()
        pages = [page for page in pages if page.page_content.strip()]
        if not pages:
            raise ValueError(
                f"The PDF contains no extractable text: {pdf_path}. "
                "Use a text-based PDF or provide an OCR-processed manual."
            )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError(f"No text chunks could be created from the PDF: {pdf_path}")
        return chunks
    except (ValueError, OSError):
        raise
    except Exception as exc:
        logger.exception("Failed to load or split PDF %s", pdf_path)
        raise RuntimeError(f"Failed to process the PDF at {pdf_path}: {exc}") from exc


def _content_hash(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _new_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_or_load_vector_store(
    pdf_path: Path,
    persist_directory: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embeddings=None,
) -> Chroma:
    try:
        pdf_hash = _content_hash(pdf_path)
        marker_path = persist_directory / "index_metadata.json"
        if marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if marker.get("source_hash") == pdf_hash:
                    existing_store = Chroma(
                        collection_name=COLLECTION_NAME,
                        persist_directory=str(persist_directory),
                        embedding_function=embeddings or _new_embeddings(),
                    )
                    existing_ids = existing_store.get().get("ids", [])
                    if existing_ids and len(existing_ids) == marker.get("chunk_count"):
                        logger.info("Reusing existing Chroma index for %s", pdf_path)
                        return existing_store
            except Exception:
                logger.exception("Existing Chroma index could not be validated; rebuilding")

        chunks = load_and_split_pdf(pdf_path, chunk_size, chunk_overlap)
        embedding_function = embeddings or _new_embeddings()
        persist_directory.mkdir(parents=True, exist_ok=True)

        stale_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(persist_directory),
            embedding_function=embedding_function,
        )
        stale_store.delete_collection()
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            collection_name=COLLECTION_NAME,
            persist_directory=str(persist_directory),
        )

        temporary_marker = marker_path.with_suffix(".json.tmp")
        temporary_marker.write_text(
            json.dumps({"source_hash": pdf_hash, "chunk_count": len(chunks)}),
            encoding="utf-8",
        )
        temporary_marker.replace(marker_path)
        logger.info("Created Chroma index with %d chunks for %s", len(chunks), pdf_path)
        return vector_store
    except Exception as exc:
        logger.exception("Failed to initialize the Chroma index")
        raise RuntimeError(
            "Could not create or load the document index. Check the PDF, "
            "embedding dependencies, and CHROMA_DIR permissions."
        ) from exc


def create_vector_store_from_environment() -> Chroma:
    pdf_path = resolve_pdf_path()
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    persist_directory = Path(os.getenv("CHROMA_DIR", "./chroma_db")).expanduser()
    if not persist_directory.is_absolute():
        persist_directory = PROJECT_DIR / persist_directory
    return build_or_load_vector_store(
        pdf_path=pdf_path,
        persist_directory=persist_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
