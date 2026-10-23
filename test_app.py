import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

import app
from classifier import (
    QuestionCategory,
    QuestionClassification,
    classify_question,
)
import ingest


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), float(text.count(" "))] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), float(text.count(" "))]


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.k = None

    def similarity_search(self, query, k):
        self.k = k
        return self.documents[:k]


class FakeChroma:
    indexed_ids = []
    next_id = 0

    def __init__(self, **kwargs):
        pass

    def get(self):
        return {"ids": list(self.indexed_ids)}

    def delete_collection(self):
        type(self).indexed_ids = []

    @classmethod
    def from_documents(cls, documents, **kwargs):
        cls.next_id += 1
        cls.indexed_ids = [str(cls.next_id) for _ in documents]
        return cls()


class FakeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": "The excerpts do not specify that."})()


class FakeClassificationLLM:
    def __init__(self, category):
        self.category = category
        self.schema = None
        self.messages = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        return QuestionClassification(category=self.category)


class IngestTests(unittest.TestCase):
    def test_pdf_path_resolution_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "ABB_ACS880_Firmware.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(ingest.resolve_pdf_path(root), pdf.resolve())
                pdf.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "PDF_PATH"):
                    ingest.resolve_pdf_path(root)

    def test_text_splitting_and_empty_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "manual.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            pages = [Document(page_content="ACS880 parameter information " * 30)]
            with patch.object(ingest.PyPDFLoader, "load", return_value=pages):
                chunks = ingest.load_and_split_pdf(pdf, chunk_size=100, chunk_overlap=10)
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(len(chunk.page_content) <= 100 for chunk in chunks))
            with patch.object(
                ingest.PyPDFLoader, "load", return_value=[Document(page_content="  ")]
            ):
                with self.assertRaisesRegex(ValueError, "no extractable text"):
                    ingest.load_and_split_pdf(pdf)

    def test_index_creation_reuse_and_pdf_change_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "manual.pdf"
            pdf.write_bytes(b"%PDF-1.4 first")
            chunks = [Document(page_content="ACS880 firmware information")]
            FakeChroma.indexed_ids = []
            with (
                patch.object(ingest, "load_and_split_pdf", return_value=chunks) as load,
                patch.object(ingest, "Chroma", FakeChroma),
            ):
                first = ingest.build_or_load_vector_store(
                    pdf, root / "index", embeddings=FakeEmbeddings()
                )
                second = ingest.build_or_load_vector_store(
                    pdf, root / "index", embeddings=FakeEmbeddings()
                )
                self.assertEqual(first.get()["ids"], second.get()["ids"])
                self.assertEqual(load.call_count, 1)

                pdf.write_bytes(b"%PDF-1.4 changed")
                ingest.build_or_load_vector_store(
                    pdf, root / "index", embeddings=FakeEmbeddings()
                )
                self.assertEqual(load.call_count, 2)


class QuestionTests(unittest.TestCase):
    def test_classifier_returns_one_of_the_two_categories(self):
        classifier_llm = FakeClassificationLLM(QuestionCategory.BROAD_QUESTION)
        category = classify_question(
            "How do the drive's control features work together?", classifier_llm
        )
        self.assertEqual(category, QuestionCategory.BROAD_QUESTION)
        self.assertIs(classifier_llm.schema, QuestionClassification)
        self.assertIn("Factual direct lookup", classifier_llm.messages[0].content)
        self.assertIn("Broad question", classifier_llm.messages[0].content)

    def test_classification_display_names_both_categories(self):
        factual_text = app.format_classification(
            QuestionCategory.FACTUAL_DIRECT_LOOKUP
        )
        broad_text = app.format_classification(QuestionCategory.BROAD_QUESTION)
        self.assertIn("classified as: Factual direct lookup", factual_text)
        self.assertIn("up to 2 excerpts", factual_text)
        self.assertIn("classified as: Broad question", broad_text)
        self.assertIn("up to 10 excerpts", broad_text)

    def test_factual_lookup_uses_top_two(self):
        documents = [
            Document(
                page_content=f"manual excerpt {index}",
                metadata={"page": index},
            )
            for index in range(12)
        ]
        store = FakeVectorStore(documents)
        llm = FakeLLM()
        with patch.object(
            app, "classify_question", return_value=QuestionCategory.FACTUAL_DIRECT_LOOKUP
        ):
            answer, context, category = app.answer_question(
                "What does the manual say?", store, llm
            )
        self.assertEqual(category, QuestionCategory.FACTUAL_DIRECT_LOOKUP)
        self.assertEqual(store.k, 2)
        self.assertEqual(context.count("manual excerpt"), 2)
        self.assertIn("[Page 1]", context)
        self.assertIn("do not provide enough information", llm.messages[0].content)
        self.assertEqual(answer, "The excerpts do not specify that.")

    def test_broad_question_uses_top_ten(self):
        documents = [
            Document(page_content=f"manual excerpt {index}")
            for index in range(12)
        ]
        store = FakeVectorStore(documents)
        llm = FakeLLM()
        with patch.object(
            app, "classify_question", return_value=QuestionCategory.BROAD_QUESTION
        ):
            answer, context, category = app.answer_question(
                "How do the drive's control features work together?", store, llm
            )
        self.assertEqual(category, QuestionCategory.BROAD_QUESTION)
        self.assertEqual(store.k, 10)
        self.assertEqual(context.count("manual excerpt"), 10)
        self.assertTrue(answer)

    def test_empty_question_and_history_append_clear(self):
        store = FakeVectorStore([])
        with self.assertRaisesRegex(ValueError, "enter a question"):
            app.answer_question("   ", store, FakeLLM())
        self.assertIsNone(store.k)
        with (
            patch.object(
                app,
                "classify_question",
                return_value=QuestionCategory.FACTUAL_DIRECT_LOOKUP,
            ),
            self.assertRaisesRegex(ValueError, "No relevant manual excerpts"),
        ):
            app.answer_question("Question", store, FakeLLM())
        self.assertEqual(store.k, 2)

        history = app.append_history([], "Question", "Answer")
        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(app.clear_history(), [])


if __name__ == "__main__":
    unittest.main()
