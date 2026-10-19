"""
End-to-end pipeline CLI for Vehicle Specification Extraction.
Supports Hugging Face serverless, Mistral AI API, and local Ollama.
Commands:
  - index: Build and persist FAISS index from the service manual.
  - query: Retrieve relevant manual chunks and extract specifications via LLM.
  - info: Display index statistics.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional
from dotenv import load_dotenv

from src.chunker import TextChunker
from src.embeddings import get_embedder
from src.pdf_extractor import PDFExtractor
from src.retriever import FAISSRetriever
from src.specification_extractor import (
    DEFAULT_HF_MODEL,
    MissingConfigurationError,
    SpecificationExtractor,
)

# Load environment variables from .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


class SpecificationPipeline:
    """End-to-end RAG pipeline for automotive service manual specification extraction."""

    def __init__(
        self,
        index_dir: str | Path = "data",
        embedder_provider: str = "local",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = get_embedder(provider=embedder_provider, model_name=embedding_model)
        self.retriever = FAISSRetriever(embedder=self.embedder, index_dir=self.index_dir)
        self.extractor = SpecificationExtractor(model_name=llm_model, hf_token=hf_token)

    def build_index(
        self,
        pdf_path: str | Path = "sample-service-manual 1.pdf",
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        max_pages: Optional[int] = None,
    ) -> None:
        """
        Parses PDF, cleans text, chunks content, and builds persistent FAISS index.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Service manual not found at: {pdf_file.resolve()}")

        print(f"\n=======================================================")
        print(f"STEP 1: Extracting text from {pdf_file.name}...")
        print(f"=======================================================")
        extractor = PDFExtractor()
        pages, summary = extractor.extract_document(pdf_file, max_pages=max_pages)

        print(f"  Total PDF pages:        {summary.total_pages_in_pdf}")
        print(f"  Content pages kept:     {summary.content_pages}")
        print(f"  Empty/diagrams skipped: {summary.skipped_empty_pages}")
        print(f"  Characters extracted:   {summary.total_characters:,}")
        print(f"  Major sections found:   {len(summary.sections_found)}")

        print(f"\n=======================================================")
        print(f"STEP 2: Chunking document (size={chunk_size}, overlap={chunk_overlap})...")
        print(f"=======================================================")
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(pages)
        print(f"  Generated {len(chunks)} searchable chunks.")

        print(f"\n=======================================================")
        print(f"STEP 3: Generating embeddings & building FAISS index...")
        print(f"=======================================================")
        self.retriever.build_index(chunks)
        print(f"  Index saved to: {self.retriever.index_path}")
        print(f"  Metadata saved to: {self.retriever.metadata_path}")
        print("  Index build COMPLETE.\n")

    def run_query(
        self,
        query: str,
        top_k: int = 5,
        output_format: str = "json",
        output_file: Optional[str] = None,
        include_metadata: bool = True,
    ) -> None:
        """
        Executes dense retrieval and extracts structured vehicle specifications via LLM.
        """
        print(f"\n=======================================================")
        print(f"QUERY: {query}")
        print(f"=======================================================")

        # Step 1: Semantic Retrieval
        results = self.retriever.retrieve(query, top_k=top_k)
        print(f"\nRetrieved Top {len(results)} Relevant Manual Chunks:")
        for r in results:
            print(
                f"  [Rank {r.rank}] Score: {r.score:.4f} | "
                f"Page {r.page_number} | Section: {r.section}"
            )
            preview = r.text.replace("\n", " ")[:140]
            print(f"          Snippet: {preview}...\n")

        # Step 2: LLM Extraction
        print(f"=======================================================")
        print(f"LLM SPECIFICATION EXTRACTION ({self.extractor.provider}: {self.extractor.model_name}):")
        print(f"=======================================================")

        try:
            specs = self.extractor.extract_specifications(query, results)
            print(f"Successfully extracted {len(specs)} specification(s):\n")

            if output_format.lower() == "csv":
                formatted_output = self.extractor.to_csv(specs, include_metadata=include_metadata)
            else:
                formatted_output = self.extractor.to_json(specs, include_metadata=include_metadata)

            print(formatted_output)

            # Save to output file if requested
            if output_file:
                out_path = Path(output_file)
                self.extractor.save_results(
                    specs,
                    out_path,
                    fmt=output_format,
                    include_metadata=include_metadata,
                )
                print(f"\nSaved structured output to: {out_path.resolve()}")

        except MissingConfigurationError as e:
            print(f"\n[CONFIGURATION NOTICE] {e}")
            print("\nRetrieved source evidence from the service manual is displayed above.")

    def show_info(self) -> None:
        """Prints current index statistics."""
        if not self.retriever.metadata_path.exists():
            print(f"No index found in {self.index_dir}. Run 'index' command first.")
            return

        with open(self.retriever.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("\n=== Vector Index Information ===")
        print(f"Directory:       {self.index_dir.resolve()}")
        print(f"Embedding Model: {data.get('model_name')}")
        print(f"Vector Dim:      {data.get('dimension')}")
        print(f"Total Chunks:    {data.get('total_chunks')}")

        pages = {c['page_number'] for c in data.get('chunks', [])}
        sections = {c['section'] for c in data.get('chunks', [])}
        print(f"Distinct Pages:  {len(pages)}")
        print(f"Unique Sections: {len(sections)}")
        print(f"Active Provider: {self.extractor.provider} ({self.extractor.model_name})")
        print("Sample Sections:")
        for s in list(sections)[:5]:
            print(f"  - {s}")
        print("===============================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vehicle Specification Extraction Pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: index
    index_parser = subparsers.add_parser("index", help="Build vector index from service manual PDF")
    index_parser.add_argument(
        "--pdf",
        default="sample-service-manual 1.pdf",
        help="Path to automotive service manual PDF (default: 'sample-service-manual 1.pdf')",
    )
    index_parser.add_argument(
        "--index-dir",
        default="data",
        help="Directory to persist FAISS index and metadata (default: 'data')",
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="Chunk size in characters (default: 900)",
    )
    index_parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters (default: 150)",
    )
    index_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit (for rapid testing)",
    )

    # Subcommand: query
    query_parser = subparsers.add_parser("query", help="Query the indexed service manual")
    query_parser.add_argument(
        "query",
        type=str,
        help="Specification query (e.g., 'What is the torque specification for the brake caliper bolts?')",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    query_parser.add_argument(
        "--model",
        default=None,
        help=f"Model identifier (default: {DEFAULT_HF_MODEL})",
    )
    query_parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format: 'json' or 'csv' (default: 'json')",
    )
    query_parser.add_argument(
        "--output",
        default=None,
        help="Path to save output structured file (e.g., outputs/brake_caliper_torque.json)",
    )
    query_parser.add_argument(
        "--index-dir",
        default="data",
        help="Directory where FAISS index was saved (default: 'data')",
    )
    query_parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude extra metadata (page_number, source_excerpt) to output only the 4 core fields",
    )

    # Subcommand: info
    info_parser = subparsers.add_parser("info", help="Display index statistics")
    info_parser.add_argument(
        "--index-dir",
        default="data",
        help="Directory where FAISS index was saved (default: 'data')",
    )

    args = parser.parse_args()

    pipeline = SpecificationPipeline(index_dir=args.index_dir, llm_model=args.model)

    if args.command == "index":
        pipeline.build_index(
            pdf_path=args.pdf,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            max_pages=args.max_pages,
        )
    elif args.command == "query":
        pipeline.run_query(
            query=args.query,
            top_k=args.top_k,
            output_format=args.format,
            output_file=args.output,
            include_metadata=not args.no_metadata,
        )
    elif args.command == "info":
        pipeline.show_info()


if __name__ == "__main__":
    main()
