# ABB ACS880 Firmware Manual RAG Demo

A small Gradio RAG application for ABB ACS880 firmware manual questions. It classifies questions as **Factual direct lookup** or **Broad question**, retrieves up to 2 or 10 relevant chunks respectively, and answers with OpenAI GPT-5.4 Mini using locally generated sentence-transformer embeddings and persistent ChromaDB storage.

## Setup

1. Use Python 3.11 or later and install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Create a local `.env` file containing your own key (do not commit it):

   ```text
   OPENAI_API_KEY=your-key-here
   ```

3. Obtain the ABB ACS880 firmware manual separately and place its PDF in the project root. The copyrighted manual is not included in this repository. Alternatively, set `PDF_PATH` in `.env` to its location.

4. Start the app:

   ```powershell
   python app.py
   ```

The app discovers a matching ACS880 firmware manual PDF in the project root, creates/reuses a local `chroma_db` index, and starts Gradio. Optional settings: `PDF_PATH`, `CHROMA_DIR`, `CHUNK_SIZE` (default `1000`), and `CHUNK_OVERLAP` (default `150`).

## Tests

```powershell
python -m unittest -v test_app.py
```
