"""
Unit tests for SpecificationExtractor schema validation, parsing, and export.
"""

import json
from pathlib import Path
import tempfile
import pytest

from src.retriever import RetrievalResult
from src.specification_extractor import (
    DEFAULT_HF_MODEL,
    MissingConfigurationError,
    SpecificationExtractor,
    VehicleSpecification,
)


def test_specification_model_valid():
    spec = VehicleSpecification(
        component="Brake Caliper Bolt",
        spec_type="Torque",
        value="35",
        unit="Nm",
        page_number=636,
        source_excerpt="Brake caliper flow bolt 35 Nm (26 lb-ft)",
    )
    assert spec.component == "Brake Caliper Bolt"
    assert spec.value == "35"
    assert spec.unit == "Nm"

    core_dict = spec.to_core_dict()
    assert set(core_dict.keys()) == {"component", "spec_type", "value", "unit"}

    full_dict = spec.to_full_dict()
    assert full_dict["page_number"] == 636
    assert "35 Nm" in full_dict["source_excerpt"]


def test_parse_clean_json_array():
    extractor = SpecificationExtractor()
    raw = json.dumps([
        {
            "component": "Brake Caliper Bolt",
            "spec_type": "Torque",
            "value": "35",
            "unit": "Nm",
            "page_number": 636,
        }
    ])
    specs = extractor.parse_and_validate_response(raw)
    assert len(specs) == 1
    assert specs[0].component == "Brake Caliper Bolt"
    assert specs[0].value == "35"


def test_parse_markdown_fenced_json():
    extractor = SpecificationExtractor()
    raw = (
        "Here are the extracted specifications:\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "component": "Brake caliper flow bolt",\n'
        '    "spec_type": "Torque",\n'
        '    "value": "35",\n'
        '    "unit": "Nm"\n'
        "  }\n"
        "]\n"
        "```"
    )
    specs = extractor.parse_and_validate_response(raw)
    assert len(specs) == 1
    assert specs[0].component == "Brake caliper flow bolt"


def test_parse_empty_result():
    extractor = SpecificationExtractor()
    assert extractor.parse_and_validate_response("[]") == []
    assert extractor.parse_and_validate_response("") == []


def test_parse_malformed_json_raises():
    extractor = SpecificationExtractor()
    with pytest.raises(ValueError):
        extractor.parse_and_validate_response("This is not JSON at all.")


def test_missing_credentials_raises_configuration_error():
    # Force credentials to None
    extractor = SpecificationExtractor(hf_token=None, api_key=None)
    extractor.hf_token = None
    extractor.mistral_key = None
    extractor.openai_key = None
    extractor.base_url = None
    extractor.provider = "unconfigured"

    dummy_chunk = RetrievalResult(
        rank=1,
        score=0.9,
        chunk_id="c1",
        text="test",
        page_number=1,
        section="test",
        subsections=[],
        metadata={},
    )
    with pytest.raises(MissingConfigurationError) as exc_info:
        extractor.extract_specifications("torque", [dummy_chunk])
    assert "credentials" in str(exc_info.value).lower()


def test_export_json_and_csv():
    extractor = SpecificationExtractor()
    specs = [
        VehicleSpecification(
            component="Brake Caliper Bolt",
            spec_type="Torque",
            value="35",
            unit="Nm",
            page_number=636,
        ),
        VehicleSpecification(
            component="Rear Axle Lubricant",
            spec_type="Fluid Capacity",
            value="2.6",
            unit="L",
            page_number=258,
        ),
    ]

    # Test JSON formatting
    json_str = extractor.to_json(specs, include_metadata=False)
    data = json.loads(json_str)
    assert len(data) == 2
    assert list(data[0].keys()) == ["component", "spec_type", "value", "unit"]

    # Test CSV formatting
    csv_str = extractor.to_csv(specs, include_metadata=True)
    assert "component,spec_type,value,unit,page_number,source_excerpt" in csv_str
    assert "Brake Caliper Bolt,Torque,35,Nm,636" in csv_str

    # Test file saving
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_file = Path(tmp_dir) / "output.json"
        csv_file = Path(tmp_dir) / "output.csv"

        extractor.save_results(specs, json_file, fmt="json")
        extractor.save_results(specs, csv_file, fmt="csv")

        assert json_file.exists()
        assert csv_file.exists()
        assert len(json.loads(json_file.read_text(encoding="utf-8"))) == 2
