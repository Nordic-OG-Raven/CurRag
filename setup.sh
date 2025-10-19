#!/bin/bash
# Quick setup script for CurRag project

set -e  # Exit on error

echo "=================================="
echo "CurRag Setup Script"
echo "=================================="
echo ""

# Check if Python is available (prefer python3 on macOS)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

echo "✓ Python found: $($PYTHON_CMD --version)"
echo ""

# Create virtual environment
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists at .venv"
    read -p "Delete and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        echo "✓ Deleted existing .venv"
    else
        echo "✓ Using existing .venv"
    fi
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo "✓ Virtual environment created"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/pdfs
mkdir -p chroma_db
echo "✓ Directories created"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found"
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "✓ Created .env from env.example"
        echo ""
        echo "📝 IMPORTANT: Edit .env and add your OPENAI_API_KEY"
        echo "   nano .env"
    else
        echo "❌ env.example not found"
    fi
else
    echo "✓ .env file exists"
fi

echo ""
echo "=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Add your OpenAI API key to .env"
echo ""
echo "3. Add PDF files to data/pdfs/"
echo "   cp your_notes.pdf data/pdfs/"
echo ""
echo "4. Index your documents:"
echo "   python scripts/index_documents.py --pdf-dir ./data/pdfs"
echo ""
echo "5. Run the app:"
echo "   streamlit run app.py"
echo ""
echo "=================================="

