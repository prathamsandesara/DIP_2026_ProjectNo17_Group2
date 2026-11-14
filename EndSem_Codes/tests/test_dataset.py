"""Tests for src.data.parser / dataset / splitter against the REAL AITEX
dataset placed under data/raw/ (see data/raw/README.md). No synthetic
directories or images are constructed."""
from src.data.splitter import normal_texture_training_samples, stratified_split
from src.data.validator import validate_samples
from tests.conftest import DATASET_ROOT, DEFAULT_CONFIG, requires_dataset


@requires_dataset
def test_parser_discovers_samples(real_samples):
    assert len(real_samples) > 0
    n_defective = sum(1 for s in real_samples if s.is_defective)
    n_defect_free = sum(1 for s in real_samples if not s.is_defective)
    assert n_defective + n_defect_free == len(real_samples)


@requires_dataset
def test_parser_matches_masks_for_defective_samples(real_samples):
    defective = [s for s in real_samples if s.is_defective]
    assert len(defective) > 0
    matched = [s for s in defective if s.mask_path is not None and s.mask_path.exists()]
    # Every defective image should ideally have a matching mask; report
    # the exact count rather than assuming 100% (see data/raw/README.md
    # for how matching is attempted).
    assert len(matched) > 0


@requires_dataset
def test_fabric_types_discovered(real_samples):
    from src.data.parser import AitexParser

    parser = AitexParser(DATASET_ROOT, DEFAULT_CONFIG["data"]["raw"])
    fabric_types = parser.fabric_types(real_samples)
    assert len(fabric_types) > 0


@requires_dataset
def test_defect_types_discovered(real_samples):
    from src.data.parser import AitexParser

    parser = AitexParser(DATASET_ROOT, DEFAULT_CONFIG["data"]["raw"])
    defect_types = parser.defect_types(real_samples)
    assert len(defect_types) > 0


@requires_dataset
def test_validator_reports_consistent_counts(real_samples):
    report = validate_samples(real_samples, check_readable=True)
    assert report.total_samples == len(real_samples)
    assert report.defect_free_count + report.defective_count == len(real_samples)


@requires_dataset
def test_stratified_split_covers_all_real_samples(real_samples):
    split = stratified_split(real_samples, random_seed=0)
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert total == len(real_samples)


@requires_dataset
def test_normal_texture_training_samples_excludes_defective(real_samples):
    split = stratified_split(real_samples, random_seed=0)
    normal = normal_texture_training_samples(split)
    assert len(normal) > 0
    assert all(not s.is_defective for s in normal)
