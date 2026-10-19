"""
Structured vehicle specification extraction module.
Supports:
  1. Hugging Face Inference API (HF_TOKEN) with active models.
  2. Official Mistral AI API (MISTRAL_API_KEY, endpoint https://api.mistral.ai/v1).
  3. Local offline Mistral via Ollama (OPENAI_BASE_URL=http://localhost:11434/v1).
  4. OpenAI-compatible endpoints (OPENAI_API_KEY).
Parses retrieved context into strictly grounded, schema-validated vehicle specifications.
Exports to JSON and CSV formats.
"""

import csv
import io
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.retriever import RetrievalResult

# Automatically load environment variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)

# Default model identifiers for each supported provider
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Fully warm & active on Hugging Face Serverless
DEFAULT_MISTRAL_API_MODEL = "open-mistral-7b"  # Official Mistral AI platform
DEFAULT_OLLAMA_MODEL = "mistral"              # Local Ollama Mistral


class MissingConfigurationError(Exception):
    """Raised when required credentials or model configuration is missing."""
    pass


class VehicleSpecification(BaseModel):
    """
    Schema for an extracted vehicle specification.
    Strictly preserves the 4 core fields from the assignment specification
    along with optional source traceability metadata.
    """
    component: str = Field(
        ...,
        description="Name of the vehicle component, fastener, or fluid item",
    )
    spec_type: str = Field(
        ...,
        description="Specification category (e.g., Torque, Fluid Capacity, Dimension, Wear Limit, Part Number)",
    )
    value: str = Field(
        ...,
        description="Exact value as stated in manual (e.g., '35', '1,420', '32', 'WSS-M6C62-A')",
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (e.g., 'Nm', 'lb-ft', 'lb-in', 'ml', 'mm', or 'N/A')",
    )
    page_number: Optional[int] = Field(
        None,
        description="Source manual page number where this specification was found",
    )
    source_excerpt: Optional[str] = Field(
        None,
        description="Verbatim text excerpt from the manual supporting this specification",
    )

    def to_core_dict(self) -> Dict[str, str]:
        """Return dict with only the 4 assignment-specified fields."""
        return {
            "component": self.component,
            "spec_type": self.spec_type,
            "value": self.value,
            "unit": self.unit,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Return dict including source page and excerpt traceability."""
        data: Dict[str, Any] = self.to_core_dict()
        if self.page_number is not None:
            data["page_number"] = self.page_number
        if self.source_excerpt is not None:
            data["source_excerpt"] = self.source_excerpt
        return data


SYSTEM_PROMPT = """You are an expert automotive specification extractor.
Your job is to extract exact vehicle specifications (e.g., torque values, fluid capacities, dimensions, part numbers) from the provided service manual excerpts.

CRITICAL RULES:
1. ONLY extract specifications explicitly supported by the provided context.
2. NEVER invent, extrapolate, guess, or use external knowledge to fabricate specifications.
3. Preserve the EXACT numeric values and units as written in the text.
4. If a component has multiple torque values in different units (e.g. 35 Nm and 26 lb-ft), you may return separate specification entries for each unit, or prioritize the primary unit (Nm).
5. If the provided context does NOT contain any specification matching the query, return an empty JSON array: []
6. Do NOT output any markdown headers, conversational filler, or explanatory prose. Output ONLY a valid JSON array matching the schema.

OUTPUT SCHEMA:
A JSON array of objects, each with the following structure:
[
  {
    "component": "Component name (e.g., Brake Caliper Flow Bolt)",
    "spec_type": "Specification type (e.g., Torque, Fluid Capacity, Dimension)",
    "value": "Numeric value or alphanumeric code",
    "unit": "Unit of measurement (e.g., Nm, lb-ft, ml, mm, or N/A)",
    "page_number": 636,
    "source_excerpt": "Exact short sentence or table row from text"
  }
]
"""


class SpecificationExtractor:
    """
    Extracts structured vehicle specifications using configured LLM provider.
    Validates output schema and provides JSON/CSV formatting.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: int = 1000,
    ) -> None:
        self.hf_token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.mistral_key = os.getenv("MISTRAL_API_KEY")
        self.openai_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        # Determine active provider and model
        if self.base_url:
            self.provider = "custom_openai"
            self.model_name = model_name or os.getenv("HF_MODEL_NAME", DEFAULT_OLLAMA_MODEL)
        elif self.mistral_key:
            self.provider = "mistral_official"
            self.model_name = model_name or os.getenv("HF_MODEL_NAME", DEFAULT_MISTRAL_API_MODEL)
        elif self.hf_token:
            self.provider = "huggingface"
            self.model_name = model_name or os.getenv("HF_MODEL_NAME", DEFAULT_HF_MODEL)
        elif self.openai_key:
            self.provider = "openai"
            self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        else:
            self.provider = "unconfigured"
            self.model_name = model_name or os.getenv("HF_MODEL_NAME", DEFAULT_HF_MODEL)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Initializes client based on available configuration."""
        if self._client is not None:
            return self._client

        if self.provider == "custom_openai":
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.openai_key or "local_dummy_key")
            return self._client

        if self.provider == "mistral_official":
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.mistral.ai/v1",
                api_key=self.mistral_key,
            )
            return self._client

        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.openai_key)
            return self._client

        if self.provider == "huggingface":
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.hf_token)
            return self._client

        raise MissingConfigurationError(
            "No LLM credentials configured. To extract specifications, set one of the following in your .env:\n"
            "  1. HF_TOKEN=hf_... and HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct (Hugging Face Serverless)\n"
            "  2. MISTRAL_API_KEY=... and HF_MODEL_NAME=open-mistral-7b (Official Mistral AI API)\n"
            "  3. OPENAI_BASE_URL=http://localhost:11434/v1 and HF_MODEL_NAME=mistral (Local Ollama Mistral)\n"
            "  4. OPENAI_API_KEY=sk-... and OPENAI_MODEL_NAME=gpt-4o-mini"
        )

    def build_user_prompt(self, query: str, context_chunks: List[RetrievalResult]) -> str:
        """Constructs prompt containing query and retrieved manual context with metadata."""
        context_parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            context_parts.append(
                f"--- EXCERPT {i} (Page {chunk.page_number} | Section: {chunk.section}) ---\n"
                f"{chunk.text}\n"
            )

        combined_context = "\n".join(context_parts)
        return (
            f"USER QUERY:\n{query}\n\n"
            f"RETRIEVED SERVICE MANUAL CONTEXT:\n{combined_context}\n\n"
            f"Extract all relevant specifications for the query in the required JSON array format:"
        )

    def parse_and_validate_response(
        self, raw_response: str
    ) -> List[VehicleSpecification]:
        """
        Parses raw LLM string into validated VehicleSpecification objects.
        Safely strips markdown code blocks and handles edge cases.
        """
        clean_text = raw_response.strip()

        # Strip markdown code blocks if model wrapped JSON in ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if fence_match:
            clean_text = fence_match.group(1).strip()

        # If text is empty or []
        if not clean_text or clean_text == "[]":
            return []

        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            # Attempt to find the first [ and last ] in case of leading/trailing model tokens
            start_idx = clean_text.find("[")
            end_idx = clean_text.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    data = json.loads(clean_text[start_idx : end_idx + 1])
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON: {clean_text}")
                    raise ValueError(f"LLM did not return valid JSON: {e}. Raw content: {raw_response[:200]}") from e
            else:
                logger.error(f"Failed to parse LLM response as JSON: {clean_text}")
                raise ValueError(f"LLM did not return valid JSON: {e}. Raw content: {raw_response[:200]}") from e

        if not isinstance(data, list):
            if isinstance(data, dict):
                data = [data]
            else:
                raise ValueError(f"Expected JSON array of specifications, got: {type(data)}")

        validated: List[VehicleSpecification] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                spec = VehicleSpecification(**item)
                validated.append(spec)
            except ValidationError as ve:
                logger.warning(f"Skipping invalid specification object: {item} (error: {ve})")

        return validated

    def extract_specifications(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
    ) -> List[VehicleSpecification]:
        """
        Executes extraction against retrieved service manual context.

        Args:
            query: User's specification query.
            retrieved_chunks: Top-k retrieved chunks from vector store.

        Returns:
            List of validated VehicleSpecification objects.
        """
        if not retrieved_chunks:
            logger.info("No context chunks provided for extraction; returning empty list.")
            return []

        client = self._get_client()
        user_prompt = self.build_user_prompt(query, retrieved_chunks)

        logger.info(f"Calling LLM provider '{self.provider}' (model: {self.model_name})")
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw_content = response.choices[0].message.content or "[]"
            return self.parse_and_validate_response(raw_content)
        except Exception as e:
            logger.error(f"LLM API request failed: {e}")
            raise

    # Convenient alias
    extract = extract_specifications

    # -------------------------------------------------------------------------
    # Format and Export Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def to_json(
        specs: List[VehicleSpecification],
        include_metadata: bool = True,
        indent: int = 2,
    ) -> str:
        """Converts specifications list to formatted JSON string."""
        if include_metadata:
            data = [s.to_full_dict() for s in specs]
        else:
            data = [s.to_core_dict() for s in specs]
        return json.dumps(data, indent=indent, ensure_ascii=False)

    @staticmethod
    def to_csv(
        specs: List[VehicleSpecification],
        include_metadata: bool = True,
    ) -> str:
        """Converts specifications list to CSV formatted string."""
        output = io.StringIO()
        if include_metadata:
            fieldnames = ["component", "spec_type", "value", "unit", "page_number", "source_excerpt"]
            data = [s.to_full_dict() for s in specs]
        else:
            fieldnames = ["component", "spec_type", "value", "unit"]
            data = [s.to_core_dict() for s in specs]

        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        return output.getvalue()

    @classmethod
    def save_results(
        cls,
        specs: List[VehicleSpecification],
        filepath: str | os.PathLike,
        fmt: str = "json",
        include_metadata: bool = True,
    ) -> None:
        """Saves extraction results to disk in JSON or CSV format."""
        path = os.fspath(filepath)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if fmt.lower() == "csv":
            content = cls.to_csv(specs, include_metadata=include_metadata)
        else:
            content = cls.to_json(specs, include_metadata=include_metadata)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved {len(specs)} specifications to {path} ({fmt})")
