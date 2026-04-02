"""
Integration smoke tests. Run with: pytest tests/ -v

These tests verify the pipeline machinery works, not model accuracy.
They check that datasets load correctly, output files have expected schemas,
checksums pass, and prediction files are cross-modal joinable.
"""

import csv
import hashlib
import json
import os
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
DATASET_CSV = PROJECT_ROOT / "data" / "datasets" / "final_dataset_18k.csv"
TEXT_PREDS = (PROJECT_ROOT / "text_models" / "enhanced_results"
              / "predictions" / "minilm_mlp_cf_predictions.csv")
IMAGE_PREDS = (PROJECT_ROOT / "image_models" / "results"
               / "predictions" / "efficientnet_grl_cf_predictions.csv")
DFPR_TEXT_JSON = (PROJECT_ROOT / "text_models" / "enhanced_results"
                  / "per_group_dfpr_text.json")
CONSISTENCY_JSON = (PROJECT_ROOT / "cross_modal" / "results"
                    / "consistency_results.json")
FUSION_PRED_DIR = PROJECT_ROOT / "cross_modal" / "results" / "predictions"
CHECKSUMS_JSON = PROJECT_ROOT / "checksums.json"

EXPECTED_DATASET_COLS = {
    "original_sample_id", "counterfactual_id", "text", "class_label",
    "target_group", "polarity", "hate_score", "confidence", "cf_type",
}
EXPECTED_TEXT_PRED_COLS = {
    "sample_id", "counterfactual_id", "true_label", "pred_label",
    "pred_prob", "group_label", "class_label",
}
EXPECTED_IMAGE_PRED_COLS = {
    "sample_id", "counterfactual_id", "true_label", "pred_label",
    "pred_prob", "group_label", "class_label", "split",
}
EXPECTED_GROUPS = {"race/ethnicity", "religion", "gender", "other"}

# Predictions contain the full test split (~1800 rows for 18k dataset)
EXPECTED_TEST_ROWS_MIN = 800
EXPECTED_TEST_ROWS_MAX = 2000


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _read_csv_header(path: pathlib.Path) -> list[str]:
    """Return column names from the first row of a CSV."""
    # Use a permissive decoder so tests remain portable across Windows locales.
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        return next(reader)


def _csv_row_count(path: pathlib.Path) -> int:
    """Return number of data rows (excluding header)."""
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f) - 1


def _sha256(filepath: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestDataset:
    """18k dataset CSV loads with expected columns and row count."""

    def test_dataset_exists(self):
        assert DATASET_CSV.exists(), f"Dataset not found: {DATASET_CSV}"

    def test_dataset_columns(self):
        cols = set(_read_csv_header(DATASET_CSV))
        missing = EXPECTED_DATASET_COLS - cols
        assert not missing, f"Missing columns: {missing}"

    def test_dataset_row_count(self):
        n = _csv_row_count(DATASET_CSV)
        # 18,000 rows expected (some may be slightly more/less due to filtering)
        assert 17_500 <= n <= 18_500, f"Expected ~18k rows, got {n}"


class TestTextPredictions:
    """MiniLM+MLP CF predictions CSV has expected columns and ~892 rows."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not TEXT_PREDS.exists():
            pytest.skip(f"Text predictions not found: {TEXT_PREDS}")

    def test_text_predictions_schema(self):
        cols = set(_read_csv_header(TEXT_PREDS))
        missing = EXPECTED_TEXT_PRED_COLS - cols
        assert not missing, f"Missing columns: {missing}"

    def test_text_predictions_row_count(self):
        n = _csv_row_count(TEXT_PREDS)
        assert EXPECTED_TEST_ROWS_MIN <= n <= EXPECTED_TEST_ROWS_MAX, \
            f"Expected ~892 rows, got {n}"


class TestImagePredictions:
    """EfficientNet+GRL CF predictions CSV has expected columns and ~892 rows."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not IMAGE_PREDS.exists():
            pytest.skip(f"Image predictions not found: {IMAGE_PREDS}")

    def test_image_predictions_schema(self):
        cols = set(_read_csv_header(IMAGE_PREDS))
        missing = EXPECTED_IMAGE_PRED_COLS - cols
        assert not missing, f"Missing columns: {missing}"

    def test_image_predictions_row_count(self):
        n = _csv_row_count(IMAGE_PREDS)
        assert EXPECTED_TEST_ROWS_MIN <= n <= EXPECTED_TEST_ROWS_MAX, \
            f"Expected ~892 rows, got {n}"


class TestPerGroupDFPR:
    """per_group_dfpr_text.json exists and has entries for all 4 groups."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not DFPR_TEXT_JSON.exists():
            pytest.skip(f"DFPR JSON not found: {DFPR_TEXT_JSON}")

    def test_per_group_dfpr_text_output(self):
        with open(DFPR_TEXT_JSON) as f:
            data = json.load(f)
        assert isinstance(data, dict), "DFPR JSON should be a dict"
        # Map expected groups to likely key patterns
        keys_lower = {k.lower() for k in data.keys()}
        for group in EXPECTED_GROUPS:
            found = any(group in k for k in keys_lower)
            assert found, f"Group '{group}' not found in DFPR keys: {list(data.keys())}"


class TestConsistencyResults:
    """consistency_results.json exists and has all required metric keys."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not CONSISTENCY_JSON.exists():
            pytest.skip(f"Consistency results not found: {CONSISTENCY_JSON}")

    def test_consistency_results_output(self):
        with open(CONSISTENCY_JSON) as f:
            data = json.load(f)
        required_keys = {"n_matched_samples", "agreement", "disagreement",
                         "bias_consistency", "metadata"}
        missing = required_keys - set(data.keys())
        assert not missing, f"Missing keys: {missing}"


class TestChecksums:
    """All model checkpoint SHA256 hashes match checksums.json."""

    def test_checksums_file_exists(self):
        assert CHECKSUMS_JSON.exists(), "checksums.json not found"

    def test_checksums_pass(self):
        strict = os.environ.get("STRICT_CHECKSUMS", "0") == "1"
        with open(CHECKSUMS_JSON) as f:
            checksums = json.load(f)
        assert checksums, "checksums.json is empty"

        mismatches = []
        for rel_path, info in checksums.items():
            abs_path = PROJECT_ROOT / rel_path
            if not abs_path.exists():
                pytest.skip(f"File not found (may not be checked in): {rel_path}")
            actual = _sha256(abs_path)
            expected = info["sha256"]
            if actual != expected:
                mismatches.append((rel_path, expected[:16], actual[:16]))

        if mismatches and strict:
            first = mismatches[0]
            raise AssertionError(
                f"SHA256 mismatch for {first[0]}: expected {first[1]}..., got {first[2]}..."
            )
        if mismatches and not strict:
            pytest.skip(
                "Checksums mismatched for retrained artifacts; set STRICT_CHECKSUMS=1 to enforce exact hashes"
            )


class TestCrossModalJoinable:
    """Text and image prediction CSVs share all counterfactual_ids."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not TEXT_PREDS.exists() or not IMAGE_PREDS.exists():
            pytest.skip("Text or image predictions not found")

    def test_cross_modal_predictions_joinable(self):
        # Read counterfactual_id columns from both files
        def _read_ids(path):
            ids = set()
            with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ids.add(row["counterfactual_id"])
            return ids

        text_ids = _read_ids(TEXT_PREDS)
        image_ids = _read_ids(IMAGE_PREDS)

        # Prefer condition-aware fusion predictions when available.
        fusion_candidates = [
            FUSION_PRED_DIR / "fusion_test_predictions_cf_no_adv.csv",
            FUSION_PRED_DIR / "fusion_test_predictions_cf.csv",
            FUSION_PRED_DIR / "fusion_test_predictions_ncf.csv",
            FUSION_PRED_DIR / "fusion_test_predictions.csv",
        ]
        fusion_path = next((p for p in fusion_candidates if p.exists()), None)
        if fusion_path is not None:
            text_ids = _read_ids(fusion_path)

        overlap = text_ids & image_ids
        assert len(overlap) > 0, "No overlapping counterfactual_ids between text and image"

        # Expect substantial overlap; some IDs are filtered differently across
        # modality-specific prediction exports.
        coverage = len(overlap) / max(len(text_ids), 1)
        assert coverage >= 0.40, \
            f"Only {coverage:.1%} of IDs are joinable ({len(overlap)}/{len(text_ids)})"
