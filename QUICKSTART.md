# Quick Start Guide

Get your RAG system running in 5 minutes!

## Automated Setup (Easiest)

```bash
cd /Users/jonas/CurRag
./setup.sh
```

Then skip to Step 3 (Add PDFs). Otherwise, follow manual steps below:

---

## Manual Setup

## Step 1: Setup Virtual Environment & Install (1 min)

```bash
# Navigate to project
cd /Users/jonas/CurRag

# Create virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Important:** Always activate `.venv` before running any commands!

## Step 2: Set API Key (30 sec)

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

Or create a `.env` file:
```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Step 3: Add PDFs (1 min)

```bash
# Create directory and add your notes
mkdir -p data/pdfs
cp ~/path/to/your/lecture_notes.pdf data/pdfs/
```

## Step 4: Index Documents (2 min)

**Activate virtual environment first:**
```bash
source .venv/bin/activate
```

Then index:
```bash
python scripts/index_documents.py --pdf-dir ./data/pdfs
```

Expected output:
```
Loading 3 PDF files...
✓ Created 247 chunks from 3 PDFs
✓ Vector store created with 247 documents
```

## Step 5: Run the App (30 sec)

**Make sure .venv is activated:**
```bash
source .venv/bin/activate
streamlit run app.py
```

Open http://localhost:8501 in your browser and start querying!

## Test Query (Optional)

```bash
python rag_pipeline.py ./data/pdfs "What are the main topics?"
```

## Troubleshooting

**No PDFs found?**
```bash
ls data/pdfs/  # Check files are there
```

**API key error?**
```bash
echo $OPENAI_API_KEY  # Verify it's set
```

**Import errors?**
```bash
pip install -r requirements.txt  # Reinstall dependencies
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [prd.md](prd.md) for architecture details
- Run tests: `python tests/test_query.py`
- Benchmark: `python tests/benchmark_retrieval.py`

Enjoy your RAG system! 🚀

