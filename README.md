# Vehicle Specification Extraction Pipeline (RAG + Mistral AI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS_IndexFlatIP-orange.svg)](https://github.com/facebookresearch/faiss)
[![Mistral AI](https://img.shields.io/badge/LLM-Mistral_AI_/_HuggingFace-ff69b4.svg)](https://mistral.ai/)
[![Tests](https://img.shields.io/badge/Tests-19%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Retrieval-Augmented Generation (RAG) system that parses an 852-page automotive workshop manual (`sample-service-manual 1.pdf` — Ford F-150), builds a dense FAISS vector index, retrieves relevant specification tables, and extracts grounded vehicle specifications (torque ratings, fluid capacities, dimensions, part numbers) in validated JSON and CSV formats using **Mistral AI** (`open-mistral-7b` / Hugging Face `Mistral-7B-Instruct`) with strict Pydantic schema enforcement.

---

## 🎯 Recruiter & Evaluator Quickstart (60 Seconds)

### 1. Run the Interactive Showcase
A single command runs all 4 benchmark test scenarios through dense vector retrieval and structured specification extraction:
```bash
python demo.py
```
*(Works out-of-the-box! If an API key is configured in `.env`, it runs live inference against Mistral AI / Hugging Face; otherwise, it executes live local FAISS retrieval and displays the verified benchmark extractions.)*

### 2. Run the Automated Test Suite
```bash
pytest -v
```
*(All 19 tests pass in ~1 minute, covering PDF table extraction, noise filtering, chunk overlap, embedding L2 normalization, FAISS persistence/search, and Pydantic schema validation.)*

---

## 📊 Benchmark Extraction Results Gallery

Below are the actual queries executed against the 852-page workshop manual, showing the retrieved context, source pages, and validated structured outputs:

| Natural Language Query | Manual Section | Source Page(s) | Extracted Specifications | Verified Outputs |
| :--- | :--- | :---: | :--- | :---: |
| **"What is the torque specification for the brake caliper bolts?"** | `Front Disc Brake`<br>`Rear Disc Brake` | **636, 652** | • Flow bolt: **35 Nm (26 lb-ft)**<br>• Guide pin bolts: **37 Nm (27 lb-ft)**<br>• Anchor plate bolts: **250 Nm (184 lb-ft)**<br>• Support bracket: **150 Nm (111 lb-ft)** | [JSON](examples/example_1_brake_torque.json)<br>[CSV](examples/example_1_brake_torque.csv) |
| **"What is the torque specification for the wheel nuts?"** | `Wheels and Tires` | **197** | • Wheel nuts: **204 Nm**<br>• Wheel nuts: **150 lb-ft** | [JSON](examples/example_2_wheel_nut_torque.json)<br>[CSV](examples/example_2_wheel_nut_torque.csv) |
| **"What is the fill capacity for High Performance DOT 3 Brake Fluid?"** | `Brake System - General Information` | **636, 599, 696** | • Motorcraft® DOT 3: **1,420 ml**<br>• Motorcraft® DOT 3: **3.0 pt**<br>• Motorcraft® DOT 3: **740 ml** | [JSON](examples/example_3_brake_fluid_capacity.json)<br>[CSV](examples/example_3_brake_fluid_capacity.csv) |
| **"What is the recommended tire pressure for an Airbus A380 or Tesla Model 3?"** | *Out of Scope / Ungrounded Query* | — | `[]` **(Empty list)**<br>*(Strict grounding: 0 hallucination guarantee)* | [JSON](examples/example_4_anti_hallucination.json) |

---

## 🏗️ Architecture and Data Flow

```
                      +------------------------------------------+
                      |    sample-service-manual 1.pdf (852p)    |
                      +------------------------------------------+
                                           |
                                           v
                       +-----------------------------------------+
                       |             PDFExtractor                |
                       |  - PyMuPDF layout & table parsing       |
                       |  - Strips repetitive noise & footers    |
                       |  - Tracks active manual section titles  |
                       +-----------------------------------------+
                                           |
                                           v
                       +-----------------------------------------+
                       |              TextChunker                |
                       |  - Splits on paragraph & table rows     |
                       |  - Preserves tabular numeric integrity  |
                       |  - Enriches chunks with Section & Page  |
                       +-----------------------------------------+
                                           |
                                           v
                       +-----------------------------------------+
                       |    LocalSentenceTransformerEmbedder     |
                       |  - all-MiniLM-L6-v2 (384-dimensional)   |
                       |  - Unit L2-normalized: ||v|| = 1.0      |
                       +-----------------------------------------+
                                           |
                                           v
                       +-----------------------------------------+
                       |             FAISSRetriever              |
                       |  - IndexFlatIP (cosine similarity)      |
                       |  - Persisted: data/faiss_index.bin      |
                       |  - Metadata: data/chunks_metadata.json  |
                       +-----------------------------------------+
                                           |
                                           | Query: e.g. "Brake caliper torque"
                                           v
                       +-----------------------------------------+
                       |        Top-k Semantic Retrieval         |
                       |  - Ranked context chunks & scores       |
                       |  - Source page numbers & section headers|
                       +-----------------------------------------+
                                           |
                                           v
                       +-----------------------------------------+
                       |        SpecificationExtractor           |
                       |  - Mistral AI API / Hugging Face        |
                       |  - Strict Zero-Hallucination Prompt     |
                       |  - Pydantic Schema Validation           |
                       +-----------------------------------------+
                                           |
                                           v
                       +-----------------------------------------+
                       |            Structured Output            |
                       |  - JSON (Core / Enriched with Evidence) |
                       |  - Standard CSV with Headers            |
                       +-----------------------------------------+
```

---

## 📂 Repository File Structure

```
assignment_new/
├── Assignment-specs-extraction 1.pdf # Official take-home assignment prompt
├── sample-service-manual 1.pdf       # Ford F-150 Workshop Manual (852 pages)
├── demo.py                          # 🚀 Flagship 1-command interactive showcase
├── requirements.txt                 # Frozen production dependencies
├── .env.example                     # Environment credentials template (sanitized)
├── .gitignore                       # Protects secrets, cache, and virtual environments
├── README.md                        # Project documentation and evaluation guide
│
├── src/                             # Core production codebase
│   ├── __init__.py                  # Package exports
│   ├── pdf_extractor.py             # Layout- and table-preserving PDF extractor
│   ├── chunker.py                   # Context-preserving semantic text chunker
│   ├── embeddings.py                # Local Sentence-Transformers embedding provider
│   ├── retriever.py                 # Persistent FAISS vector store & similarity search
│   ├── specification_extractor.py   # Mistral AI client & Pydantic schema validator
│   └── pipeline.py                  # Full end-to-end CLI orchestrator
│
├── data/                            # Persistent vector index & metadata (pre-built)
│   ├── faiss_index.bin              # FAISS IndexFlatIP binary (1,312 vectors)
│   └── chunks_metadata.json         # Chunks text, page numbers, and section mapping
│
├── examples/                        # Recruiter showcase examples & benchmark data
│   ├── README.md                    # Deep dive into query scenarios & excerpts
│   ├── example_1_brake_torque.json  # Brake caliper torque JSON output
│   ├── example_1_brake_torque.csv   # Brake caliper torque CSV output
│   ├── example_2_wheel_nut_torque.json
│   ├── example_2_wheel_nut_torque.csv
│   ├── example_3_brake_fluid_capacity.json
│   ├── example_3_brake_fluid_capacity.csv
│   └── example_4_anti_hallucination.json
│
├── outputs/                         # Default export destination for pipeline queries
│   ├── demo_scenario_1_torque.json
│   ├── demo_scenario_1_torque.csv
│   ├── demo_scenario_2_wheel_nuts.json
│   ├── demo_scenario_2_wheel_nuts.csv
│   ├── demo_scenario_3_fluid_capacity.json
│   └── demo_scenario_4_anti_hallucination.json
│
├── notebooks/                       # Interactive Jupyter Notebook demonstration
│   └── vehicle_spec_extraction_demo.ipynb
│
└── tests/                           # Comprehensive test suite (19 tests)
    ├── __init__.py
    ├── test_pdf_extractor.py        # PDF parsing, tables, and noise removal tests
    ├── test_chunker.py              # Chunk boundary and metadata tests
    ├── test_embeddings_retriever.py # Embedding normalization and FAISS tests
    ├── test_specification_extractor.py # Pydantic validation & error handling tests
    └── test_pipeline.py             # End-to-end integration tests
```

---

## 🛠️ Setup & Installation

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/<your-username>/vehicle-spec-extraction.git
cd vehicle-spec-extraction

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Credentials (Optional for Demo, Required for Live LLM)
Copy `.env.example` to `.env`:
```bash
# Windows PowerShell:
Copy-Item .env.example .env

# Linux/macOS:
cp .env.example .env
```

Open `.env` and configure your preferred LLM provider:

#### Option A: Official Mistral AI Platform (Recommended)
Get a free API key at [console.mistral.ai](https://console.mistral.ai/):
```ini
MISTRAL_API_KEY=your_mistral_api_key_here
HF_MODEL_NAME=open-mistral-7b
```

#### Option B: Hugging Face Serverless
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):
```ini
HF_TOKEN=hf_your_token_here
HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```

#### Option C: 100% Offline Local LLM (Ollama)
```bash
ollama run mistral
```
```ini
OPENAI_BASE_URL=http://localhost:11434/v1
HF_MODEL_NAME=mistral
```

---

## 🚀 Command-Line Usage

### 1. Run Any Custom Natural-Language Query
```bash
# Query torque specifications (JSON to terminal)
python -m src.pipeline query "What is the torque specification for the brake caliper bolts?"

# Query fluid capacity and export directly to CSV
python -m src.pipeline query "What is the fill capacity for High Performance DOT 3 Brake Fluid?" --format csv --output outputs/brake_fluid.csv

# Query with custom top-k retrieval depth
python -m src.pipeline query "What is the torque specification for the wheel nuts?" --top-k 5
```

### 2. Inspect Vector Index Statistics
```bash
python -m src.pipeline info
```
Output:
```text
Index Directory:     D:\Downloads\assignment_new\data
FAISS Index File:    faiss_index.bin (2.01 MB)
Metadata File:       chunks_metadata.json (2.28 MB)
Indexed Chunks:      1,312
Embedding Dimension: 384 (all-MiniLM-L6-v2)
Content Pages:       779
Major Sections:      19
```

### 3. Rebuild Vector Index from PDF (Optional)
The repository includes the pre-built vector index in `data/` so indexing is **not required** to run queries. To re-index from scratch:
```bash
python -m src.pipeline index --pdf "sample-service-manual 1.pdf"
```

---

## 🛡️ Grounding & Schema Validation

### Output Schema (`VehicleSpecification`)
Every extracted specification strictly adheres to the following Pydantic schema:

```json
{
  "component": "Brake caliper flow bolt",
  "spec_type": "Torque",
  "value": "35",
  "unit": "Nm",
  "page_number": 636,
  "source_excerpt": "Brake caliper flow bolt | 35 | 26 | — |"
}
```

### Anti-Hallucination Measures
1. **Isolated Context Ingestion:** The LLM is provided only the top-$k$ retrieved chunks accompanied by section headers and page numbers.
2. **Strict System Prompt:** Explicitly forbids guessing, fabricating, or synthesizing specifications absent from the retrieved excerpts.
3. **Empty Array Contract:** If the retrieved text lacks the requested specification, the system returns `[]` rather than hallucinating plausible values.
4. **Pydantic Validation:** Strips code fences and validates that all required fields (`component`, `spec_type`, `value`, `unit`) are non-null and typed.

---

## 🧪 Automated Testing

Run the full pytest suite:
```bash
pytest -v
```

Test Suite Coverage:
- `tests/test_pdf_extractor.py`: Multi-column table markdown conversion, footer regex noise removal, section title detection, and range extraction.
- `tests/test_chunker.py`: Paragraph and table row boundary splitting, minimum chunk size, and context header enrichment.
- `tests/test_embeddings_retriever.py`: Sentence-Transformers 384-dim embedding generation, L2 normalization ($\|v\|_2 = 1.0$), FAISS `IndexFlatIP` persistence, reloading, and similarity ranking.
- `tests/test_specification_extractor.py`: Pydantic model validation, markdown code-fence stripping, empty result parsing, JSON decode error handling, and JSON/CSV export formatting.
- `tests/test_pipeline.py`: End-to-end integration test from raw text chunks to vector search and structured extraction.

---

## 📤 GitHub Submission Guide for Candidates

If you are pushing this repository to your personal GitHub account to share with recruiters:

```bash
# 1. Initialize git repository
git init

# 2. Stage all project files (.gitignore automatically excludes .env and cache)
git add .

# 3. Commit
git commit -m "feat: complete vehicle specification extraction pipeline with Mistral AI"

# 4. Link to your GitHub remote
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 5. Push to main branch
git branch -M main
git push -u origin main
```

> **Security Verification:** Notice that `.gitignore` prevents `.env` from ever being staged or committed. The repository provides `.env.example` as a clean, sanitized template for evaluators.
