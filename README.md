# LLM Council - Multi-Expert AI System

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Ollama](https://img.shields.io/badge/ollama-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A sophisticated AI system that routes queries to specialized local LLM experts using Ollama, with real-time web search capabilities and RAG-powered knowledge retrieval.

## Features

### Four Specialized Experts

- **Math Expert** (`qwen2-math:7b`) - Calculus, integrals, derivatives, proofs
- **Coding Expert** (`deepseek-coder:6.7b`) - Algorithm implementation in Python, C++, Java, etc.
- **Vision Expert** (`llava:7b-v1.6`) - Image analysis and text extraction
- **Knowledge Expert** (`qwen3:8b`) - Statistics, CS theory with RAG from 2,679 textbook pages

### Research Capabilities

- Real-time web search via DuckDuckGo
- Automatic citation with source links
- News and stock market queries

### Smart Routing

- **Weighted keyword matching** (~100ms routing time)
- **Semantic similarity fallback** using sentence-transformers
- **Context-aware conversations** with follow-up detection

### Conversational Memory

- Tracks conversation history
- Expands implicit follow-ups (e.g., "now do it in Python")
- Maintains expert context across turns

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running
- **8GB+ RAM** (for model loading)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/nimeshk03/llm-council.git
cd llm-council
```

2. **Create virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Pull required Ollama models**

```bash
ollama pull qwen2-math:7b-instruct
ollama pull deepseek-coder:6.7b-instruct
ollama pull llava:7b-v1.6
ollama pull qwen3:8b
```

5. **Prepare RAG database** (optional, if you have textbook PDFs)

```bash
# Place PDFs in data/books/
python -m experts.rag_knowledge  # Indexes documents
```

### Usage

#### CLI Interface

```bash
python -m experts.session_manager
```

**Example queries:**

```
You: Find the integral of x^3
You: Implement binary search in C++
You: Latest NVIDIA news
You: /path/to/image.png extract text
```

**Follow-up examples:**

```
You: Evaluate the integral x^3
Assistant: [answers with result]
You: Now find the derivative
Assistant: [expands to "derivative of x^3"]
```

#### Web Interface (Gradio)

```bash
python web_interface.py
```

Access at:
- **Local:** `http://localhost:7860`
- **Network:** `http://<your-ip>:7860` (from other devices on same WiFi)

## Project Structure

```
llm-council/
├── experts/
│   ├── __init__.py
│   ├── supervisor.py          # Expert orchestration
│   ├── semantic_router.py     # Query routing logic
│   ├── session_manager.py     # Conversation management
│   ├── rag_knowledge.py       # RAG + ChromaDB
│   └── research_expert.py     # Web search + synthesis
├── data/
│   ├── chroma/                # Vector database (auto-generated)
│   └── books/                 # Source PDFs (optional)
├── tests/
│   └── test_vision.py         # Vision model debugging
├── web_interface.py           # Gradio UI
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Testing

### Test Individual Components

```bash
# Test semantic routing
python -m experts.semantic_router

# Test RAG knowledge base
python -m experts.rag_knowledge

# Test vision model with debug
python tests/test_vision.py
```

### Example Test Queries

| Expert | Query Example |
|--------|---------------|
| Math | "Find the derivative of sin(x) * e^x" |
| Coding | "Implement quicksort in Python" |
| Vision | "/path/to/image.png extract text" |
| Knowledge | "Explain hypothesis testing in statistics" |
| Research | "Latest Tesla stock news" |

## Configuration

### Model Selection

Edit `experts/supervisor.py`:

```python
EXPERT_MODELS = {
    ExpertType.MATH: "qwen2-math:7b-instruct",
    ExpertType.CODING: "deepseek-coder:6.7b-instruct",
    ExpertType.VISION: "llava:7b-v1.6",
    ExpertType.KNOWLEDGE: "qwen3:8b",
}
```

### RAG Settings

Edit `experts/rag_knowledge.py`:

```python
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks
TOP_K_RESULTS = 4          # Number of results to retrieve
```

### Routing Keywords

Edit `experts/semantic_router.py` to customize keyword weights:

```python
self.weighted_keywords = {
    ExpertType.MATH: [
        (r'\b(integral|derivative)\b', 10.0),
        (r'\b(calculus|theorem)\b', 8.0),
    ],
    # ... add more patterns
}
```

## Performance

- **Routing**: ~100ms (keyword-based), ~300ms (semantic fallback)
- **Model Loading**: 3-8 seconds per expert
- **Response Time**: 5-30 seconds depending on query complexity
- **Memory Usage**: ~4-6GB per loaded model

## Troubleshooting

### Models Too Large for RAM

Use smaller quantized models:

```bash
ollama pull qwen2-math:1b
ollama pull llava:7b  # Smaller than qwen2.5vl:3b
```

### Vision Model 500 Error

Check available memory:

```bash
ollama ps  # See what models are loaded

# Unload models to free memory
curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"qwen3:8b","prompt":"","keep_alive":0}'
```

### Web Search Not Working

Install DuckDuckGo search package:

```bash
pip install ddgs
```

### ChromaDB Errors

Delete and rebuild the database:

```bash
rm -rf data/chroma/
python -m experts.rag_knowledge
```

### Firewall Blocking Web Interface

Allow port 7860:

```bash
sudo ufw allow 7860
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run linting
flake8 experts/
black experts/

# Run tests
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Ollama** - Local LLM inference
- **ChromaDB** - Vector database for RAG
- **sentence-transformers** - Semantic routing embeddings
- **Gradio** - Web interface framework
- **DuckDuckGo** - Web search API
- **qwen2-math** - Mathematics expert model
- **deepseek-coder** - Coding expert model
- **llava** - Vision-language model
- **qwen3** - Knowledge synthesis model

## Contact

**Nimesh K** - [@nimeshk03](https://github.com/nimeshk03)

Project Link: [https://github.com/nimeshk03/llm-council](https://github.com/nimeshk03/llm-council)

---

**Star this repo if you find it useful!**