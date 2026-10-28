# PaperPilot — PDF RAG Assistant

A Streamlit PDF question-answering app using LangChain, OpenAI embeddings/chat models, and FAISS.

## Features

- API-key entry screen before the workspace opens
- Upload and index a PDF
- Ask questions grounded in the document
- Page-level source excerpts
- Model and retrieval settings
- Clear chat, start a new document, and disconnect the API key
- Responsive custom UI designed for Streamlit Community Cloud

## Run locally

Use Python 3.10–3.12 for best compatibility.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

macOS/Linux:
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run main.py
```

Open the local URL printed by Streamlit, then enter your own OpenAI API key.

## Deploy on Streamlit Community Cloud

1. Push `main.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Sign in to Streamlit Community Cloud and create a new app.
3. Select the repository, branch, and `main.py` as the main file.
4. Deploy the app.
5. Users enter their OpenAI API key on the app's welcome screen.

## Security and costs

- This app uses an API-key gate, not user accounts or full authentication.
- The API key is kept in Streamlit session state for the current session and is not intentionally written to a file or logged by the app.
- Do not use this as a multi-user secure credential vault. For production multi-user deployments, consider server-side authentication, secret management, rate limits, and per-user isolation.
- OpenAI API calls can incur charges. PDF text is sent to the configured OpenAI embedding/chat APIs as part of processing.
- Avoid uploading sensitive documents unless you are authorized to process them with the configured services.

## Notes

- The app handles text-based PDFs. Scanned/image-only PDFs generally need OCR before upload.
- Re-indexing is required when chunk size or overlap changes.
- Streamlit Community Cloud resource limits may affect very large PDFs.
