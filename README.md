# Vehicle Specification Extraction Pipeline

A Retrieval-Augmented Generation (RAG) pipeline to extract structured vehicle specifications (torque ratings, fluid capacities, dimensions, part numbers) from an automotive service manual PDF (`sample-service-manual 1.pdf`, Ford F-150 Workshop Manual, 852 pages).

The extracted specifications are strictly grounded in manual excerpts and exported in JSON or CSV format matching the assignment schema.

---

## Overview

Automotive service manuals contain dense technical information presented in multi-column tables and procedure steps. Standard text extraction often separates component names from their corresponding torque numbers or units.

This project implements:
1. **Layout- and Table-Aware PDF Parsing**: Uses PyMuPDF to extract text while converting multi-column specification tables to Markdown pipe tables, filtering out repetitive headers, footers, and page noise.
2. **Context-Preserving Chunking**: Splits text on paragraph and table row boundaries, attaching section titles and page numbers to each chunk.
3. **Dense Vector Retrieval**: Embeds chunks using SentenceTransformers (`all-MiniLM-L6-v2`, 384 dimensions, L2-normalized) and builds an exact inner-product FAISS index (`IndexFlatIP`).
4. **Structured LLM Extraction**: Queries Mistral AI (`open-mistral-7b` / Hugging Face serverless) using a prompt designed to prevent hallucination, validated against a Pydantic schema (`VehicleSpecification`).
5. **Output Export**: Exports specifications to validated JSON or CSV formats.

---

## Architecture

```
                 sample-service-manual 1.pdf (852 pages)
                                    |
                                    v
                             PDFExtractor
                 (PyMuPDF table detection & noise removal)
                                    |
                                    v
                               TextChunker
                 (Splits on paragraphs/rows, preserves tables)
                                    |
                                    v
                      SentenceTransformer Embeddings
                        (all-MiniLM-L6-v2, 384-dim)
                                    |
                                    v
                             FAISSRetriever
                 (IndexFlatIP, persistent in data/)
                                    |
                User Query --------+
                                    |
                                    v
                            Top-k Chunks (cosine sim)
                                    |
                                    v
                        SpecificationExtractor
                    (Mistral AI + Pydantic schema)
                                    |
                                    v
                           JSON / CSV Output
```

---

## Project Structure

```
.
├── Assignment-specs-extraction 1.pdf  # Assignment specification
├── sample-service-manual 1.pdf        # Ford F-150 Workshop Manual
├── demo.py                           # Quick demo running sample queries
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore file
├── README.md                         # Project documentation
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── pdf_extractor.py              # PDF extraction and table parsing
│   ├── chunker.py                    # Text chunking and metadata preservation
│   ├── embeddings.py                 # Embedding generation wrapper
│   ├── retriever.py                  # FAISS index and retrieval logic
│   ├── specification_extractor.py    # LLM extraction and Pydantic validation
│   └── pipeline.py                   # Main CLI entry point
│
├── data/                             # Pre-built FAISS index and metadata
│   ├── faiss_index.bin               # FAISS vector index (1,312 vectors)
│   └── chunks_metadata.json          # Chunks text and page mapping
│
├── examples/                         # Sample queries and outputs
│   ├── README.md
│   ├── example_1_brake_torque.json
│   ├── example_1_brake_torque.csv
│   ├── example_2_wheel_nut_torque.json
│   ├── example_2_wheel_nut_torque.csv
│   ├── example_3_brake_fluid_capacity.json
│   ├── example_3_brake_fluid_capacity.csv
│   └── example_4_anti_hallucination.json
│
├── outputs/                          # Query output destination
├── notebooks/                        # Interactive Jupyter notebook walkthrough
│   └── vehicle_spec_extraction_demo.ipynb
│
└── tests/                            # Unit and integration test suite
    ├── test_pdf_extractor.py
    ├── test_chunker.py
    ├── test_embeddings_retriever.py
    ├── test_specification_extractor.py
    └── test_pipeline.py
```

---

## Quickstart

### 1. Run the Demo
The repository includes the pre-built FAISS index in `data/`, so queries can be run immediately without having to re-index the 852-page manual:

```bash
python demo.py
```

This runs four test cases (brake torque, wheel nut torque, brake fluid capacity, and an out-of-scope query). If an API key is present in `.env`, it queries Mistral AI live; otherwise, it runs local FAISS retrieval and displays the verified benchmark output.

You can also pass a custom query:
```bash
python demo.py --query "What is the torque specification for the brake caliper bolts?"
```

### 2. Run Tests
```bash
pytest -v
```

All 19 tests cover PDF extraction, chunking, embeddings, FAISS indexing, Pydantic schema validation, and end-to-end pipeline execution.

---

## Sample Queries and Results

| Query | Manual Section | Source Page | Extracted Specification | Outputs |
| :--- | :--- | :---: | :--- | :--- |
| "What is the torque specification for the brake caliper bolts?" | Front / Rear Disc Brake | 636, 652 | Flow bolt: 35 Nm (26 lb-ft)<br>Guide pin bolts: 37 Nm (27 lb-ft)<br>Anchor plate bolts: 250 Nm (184 lb-ft)<br>Support bracket: 150 Nm (111 lb-ft) | [JSON](examples/example_1_brake_torque.json)<br>[CSV](examples/example_1_brake_torque.csv) |
| "What is the torque specification for the wheel nuts?" | Wheels and Tires | 197 | Wheel nuts: 204 Nm (150 lb-ft) | [JSON](examples/example_2_wheel_nut_torque.json)<br>[CSV](examples/example_2_wheel_nut_torque.csv) |
| "What is the fill capacity for High Performance DOT 3 Brake Fluid?" | Brake System - General Info | 636, 599, 696 | Motorcraft DOT 3: 1,420 ml (3.0 pt)<br>Motorcraft DOT 3: 740 ml (1.56 pt) | [JSON](examples/example_3_brake_fluid_capacity.json)<br>[CSV](examples/example_3_brake_fluid_capacity.csv) |
| "What is the recommended tire pressure for an Airbus A380?" | Out-of-Scope Query | — | `[]` (Empty list, zero hallucination) | [JSON](examples/example_4_anti_hallucination.json) |

---

## Setup and Installation

### 1. Requirements
- Python 3.10+ (tested on Python 3.10, 3.11, and 3.14)
- Dependencies listed in `requirements.txt`

### 2. Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file from the provided template:

```bash
# Windows
Copy-Item .env.example .env

# Linux/macOS
cp .env.example .env
```

Set your credentials in `.env`. Supported providers:

- **Mistral AI Platform** (recommended):
  ```ini
  MISTRAL_API_KEY=your_mistral_api_key
  HF_MODEL_NAME=open-mistral-7b
  ```

- **Hugging Face Serverless**:
  ```ini
  HF_TOKEN=hf_your_token
  HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
  ```

- **Local Ollama**:
  ```ini
  OPENAI_BASE_URL=http://localhost:11434/v1
  HF_MODEL_NAME=mistral
  ```

---

## CLI Usage

The pipeline can be executed via `src.pipeline`:

### Query Specifications
```bash
# JSON output to terminal
python -m src.pipeline query "What is the torque specification for the brake caliper bolts?"

# Export to CSV
python -m src.pipeline query "What is the torque specification for the brake caliper bolts?" --format csv --output outputs/brake_caliper.csv

# Export to JSON with top-k retrieval depth
python -m src.pipeline query "What is the fill capacity for High Performance DOT 3 Brake Fluid?" --top-k 5 --format json --output outputs/fluid.json
```

### Inspect the Vector Index
```bash
python -m src.pipeline info
```
Output:
```text
Index Directory:     data
FAISS Index File:    faiss_index.bin (2.01 MB)
Metadata File:       chunks_metadata.json (2.28 MB)
Indexed Chunks:      1,312
Embedding Dimension: 384 (all-MiniLM-L6-v2)
Content Pages:       779
Major Sections:      19
```

### Rebuild Index from PDF (Optional)
```bash
python -m src.pipeline index --pdf "sample-service-manual 1.pdf"
```

---

## Technical Implementation Details

### PDF Extraction & Table Preservation (`src/pdf_extractor.py`)
- Workshop manual specifications are organized in structured tables (e.g. `Description | Nm | lb-ft | lb-in`).
- Using PyMuPDF's `find_tables()`, detected tables are converted to Markdown pipe tables so column relationships remain intact.
- Non-tabular text blocks that do not overlap with tables are extracted separately and merged in reading order.
- Noise lines (such as `Page X sur Y`, print dates, and internal file paths like `file:///C:/TSO/...`) are removed via regex.
- Active section headers (e.g., `SECTION 206-03: Front Disc Brake`) are tracked across pages and assigned to chunks.

### Chunking Strategy (`src/chunker.py`)
- Chunks split on natural paragraph breaks (`\n\n`) and table rows (`\n`) rather than fixed character boundaries to avoid cutting numbers from component names.
- Default settings: `chunk_size=900` characters with `chunk_overlap=150` characters.
- Each chunk is prepended with contextual metadata `[Section: ... | Page: ...]` before embedding to improve retrieval accuracy on specific technical systems.

### Embedding & Vector Search (`src/embeddings.py`, `src/retriever.py`)
- Uses `all-MiniLM-L6-v2` locally via SentenceTransformers.
- Embeddings are 384-dimensional and explicitly L2-normalized ($\|v\|_2 = 1.0$).
- Vector store uses FAISS `IndexFlatIP`, where the inner product on normalized vectors is equivalent to cosine similarity.
- The index and chunk metadata are saved to `data/faiss_index.bin` and `data/chunks_metadata.json`.

### Structured Output & Grounding (`src/specification_extractor.py`)
- Every extracted specification is validated against the `VehicleSpecification` Pydantic model:
  ```python
  class VehicleSpecification(BaseModel):
      component: str
      spec_type: str
      value: str
      unit: str
      page_number: Optional[int] = None
      source_excerpt: Optional[str] = None
  ```
- **Grounding and Anti-Hallucination**: The prompt instructs the model to extract values *only* from the provided context. If the requested information is absent or out-of-scope, the model returns an empty list `[]` instead of hallucinating values.
- Clean JSON and CSV serializers format the data according to the assignment requirements.

---

## Testing

Run the test suite using pytest:
```bash
pytest -v
```

The 19 tests verify:
- Table extraction and noise line removal in `test_pdf_extractor.py`.
- Chunk boundaries and overlap behavior in `test_chunker.py`.
- Vector normalization, FAISS indexing, and retrieval ranking in `test_embeddings_retriever.py`.
- Pydantic validation, markdown code block stripping, and export functions in `test_specification_extractor.py`.
- Pipeline indexing and query flow in `test_pipeline.py`.
