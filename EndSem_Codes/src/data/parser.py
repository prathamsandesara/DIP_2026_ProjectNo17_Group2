"""
AITEX Fabric Image Database (AFID) directory/filename parser.

The AITEX dataset is organized as:

    NODefect_images/<fabric_roll_folder>/<image_name>.png   (defect-free images)
    Defect_images/<image_name>.png                            (defective images)
    Mask_images/<mask_name>.png                               (ground-truth masks)

Filename convention (confirmed empirically against the actual distributed
dataset -- see `data/raw/README.md` for how this was verified):

    <sequential_index>_<defect_code>_<fabric_type>.png        (both defect-free and defective)
    <sequential_index>_<defect_code>_<fabric_type>_mask[N].png (mask; N for multi-mask defects)

e.g. `0072_010_03.png` -> sequential_index=0072 (a running counter with no
semantic meaning, discarded), defect_code=010, fabric_type=03. For
defect-free images the defect_code is always the sentinel `000`.

**This differs from some other public AITEX mirrors**, which instead use
`<fabric_id>_<roll>_<position>[_<defect_code>]`. We verified the actual
convention for this dataset directly:

  - Every image inside one `NODefect_images/<subfolder>/` shares an
    identical TRAILING filename token (e.g. all files in
    `NODefect_images/2306881-210020u/` end in `_05`), and there are
    exactly 7 such subfolders with 7 distinct trailing tokens (00-06) --
    matching AITEX's documented 7 fabric types exactly.
  - `Defect_images/*.png` filenames' trailing token takes the same
    00-06 values (plus two singleton outliers, 05 and 08, out of 106
    images -- treated as legitimate rare/edge values, not stripped).
  - `Defect_images/*.png` filenames' MIDDLE token takes exactly 12
    distinct numeric values -- matching AITEX's documented 12 defect
    types exactly. `NODefect_images` files' middle token is always the
    `000` sentinel.

So: **fabric type = trailing token**, **defect code = middle token**.

Because AITEX's public documentation for defect-code -> defect-name
mapping is not bundled with every mirror of this dataset, this parser
does NOT invent that mapping. Defect samples are labeled `defect_<code>`
(e.g. `defect_019`) unless you populate `configs/default.yaml`'s
`data.raw.defect_code_map` (e.g. `{"019": "thin_bar"}`) with the real
correspondence for your copy -- see `data/raw/README.md`.

Rather than hard-code this exact layout, this parser:

  1. Reads directory layout and filename patterns from `configs/default.yaml`
     (section `data.raw`), so they can be adjusted after you inspect the
     actual downloaded dataset with `scripts/inspect_dataset.py`.
  2. Falls back to permissive glob-based discovery (any *.png/*.jpg under
     the configured image/mask roots) plus token-based metadata
     extraction, so a slightly different real-world layout does not
     silently produce zero samples.

If your downloaded copy uses a different structure, edit the `data.raw`
section of the config (glob patterns, defect-code map) rather than this
file's logic -- that is the intended adaptation point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_DEFECT_FREE_GLOB = "NODefect_images/**/*.png"
DEFAULT_DEFECT_GLOB = "Defect_images/**/*.png"
DEFAULT_MASK_GLOB = "Mask_images/**/*.png"

# Sentinel defect-code used by defect-free images (no defect present).
NO_DEFECT_CODE = "000"

# Known AITEX defect-type name vocabulary (12 classes as documented for
# AFID), used only when a filename spells the defect out by name rather
# than by numeric code (some AITEX mirrors do this instead).
AITEX_DEFECT_TYPES = [
    "broken_end", "broken_pick", "weft_crack", "warp_crack", "knot",
    "netting_multiple", "thick_bar", "thin_bar", "float", "nep",
    "hole", "stain",
]


@dataclass
class Sample:
    """One dataset sample: an image, optional mask, and parsed metadata."""

    image_path: Path
    mask_path: Optional[Path]
    fabric_id: str
    defect_type: Optional[str]  # None for defect-free samples
    is_defective: bool
    sample_id: str
    extra: Dict = field(default_factory=dict)


def _tokenize(stem: str) -> List[str]:
    return stem.split("_")


def _extract_fabric_id(stem: str) -> str:
    """Fabric type = the TRAILING underscore-separated token of the
    filename (confirmed convention; see module docstring). Falls back to
    "unknown" for filenames with fewer than 2 tokens."""
    tokens = _tokenize(stem)
    if len(tokens) < 2:
        return "unknown"
    return tokens[-1]


def _extract_defect_code(stem: str) -> Optional[str]:
    """Defect code = the MIDDLE underscore-separated token of the
    filename (confirmed convention; see module docstring). Returns None
    if the filename has fewer than 3 tokens."""
    tokens = _tokenize(stem)
    if len(tokens) < 3:
        return None
    return tokens[1]


def _extract_defect_type(stem: str, defect_code_map: Optional[Dict[str, str]] = None) -> str:
    """Resolve a human-readable defect-type label for a defective sample.

    Tries, in order: (1) a named AITEX defect type appearing literally in
    the filename (for mirrors that spell defects out by name), (2) the
    numeric defect code mapped through the user-supplied
    `data.raw.defect_code_map`, (3) a generic `defect_<code>` label so no
    information is silently discarded when the real code->name mapping is
    not available.
    """
    lowered = stem.lower()
    for defect in AITEX_DEFECT_TYPES:
        if defect in lowered:
            return defect

    code = _extract_defect_code(stem)
    if code is not None:
        if defect_code_map and code in defect_code_map:
            return defect_code_map[code]
        return f"defect_{code}"

    return "unknown_defect"


def _find_mask_for_image(image_path: Path, mask_paths_by_stem: Dict[str, Path]) -> Optional[Path]:
    """Match an image to its mask by exact/suffixed stem lookup.

    AITEX masks are typically named identically to the defective image
    (sometimes with a `_mask` / `_mask1` / `_mask2` suffix, the latter for
    images with more than one disjoint defect region). We try exact stem,
    then `stem_mask`, `stem_mask1`, then a longest-common-prefix fallback.
    Note: a defect with multiple mask files (`_mask1`, `_mask2`, ...) is
    matched to only the first found here; `AitexParser.parse` does not
    currently merge multi-mask defects into one sample -- see
    `data/raw/README.md` if your evaluation needs to account for this.
    """
    stem = image_path.stem
    if stem in mask_paths_by_stem:
        return mask_paths_by_stem[stem]
    for suffix in ("_mask", "_mask1", "_mask2"):
        if f"{stem}{suffix}" in mask_paths_by_stem:
            return mask_paths_by_stem[f"{stem}{suffix}"]
    # prefix match: mask stem starts with image stem or vice versa
    for mask_stem, mask_path in mask_paths_by_stem.items():
        if mask_stem.startswith(stem) or stem.startswith(mask_stem):
            return mask_path
    return None


class AitexParser:
    """Parses the AITEX dataset directory into a list of `Sample` objects.

    Parameters
    ----------
    root : Path
        Root directory containing NODefect_images/, Defect_images/,
        Mask_images/ (see data/raw/README.md).
    config : dict, optional
        Sub-dict `data.raw` from the YAML config, allowing glob patterns
        and a defect-code map to be overridden without touching this file.
    """

    def __init__(self, root: Path, config: Optional[Dict] = None):
        self.root = Path(root)
        config = config or {}
        self.defect_free_glob = config.get("defect_free_glob", DEFAULT_DEFECT_FREE_GLOB)
        self.defect_glob = config.get("defect_glob", DEFAULT_DEFECT_GLOB)
        self.mask_glob = config.get("mask_glob", DEFAULT_MASK_GLOB)
        self.defect_code_map = config.get("defect_code_map", {})
        self.image_extensions = tuple(config.get("image_extensions", [".png", ".jpg", ".jpeg", ".bmp"]))

    def _glob(self, pattern: str) -> List[Path]:
        return sorted(p for p in self.root.glob(pattern) if p.suffix.lower() in self.image_extensions)

    def parse(self) -> List[Sample]:
        """Discover all samples under `self.root`. Never raises on a
        missing sub-directory (returns fewer samples instead) so that
        `inspect_dataset.py` can report exactly what is missing."""
        samples: List[Sample] = []

        mask_paths = self._glob(self.mask_glob)
        mask_paths_by_stem = {p.stem: p for p in mask_paths}

        # Defect-free images: no mask expected (or an all-zero mask if present).
        for img_path in self._glob(self.defect_free_glob):
            fabric_id = _extract_fabric_id(img_path.stem)
            samples.append(
                Sample(
                    image_path=img_path,
                    mask_path=_find_mask_for_image(img_path, mask_paths_by_stem),
                    fabric_id=fabric_id,
                    defect_type=None,
                    is_defective=False,
                    sample_id=img_path.stem,
                )
            )

        # Defective images: mask required for pixel-level evaluation.
        for img_path in self._glob(self.defect_glob):
            fabric_id = _extract_fabric_id(img_path.stem)
            defect_type = _extract_defect_type(img_path.stem, self.defect_code_map)
            samples.append(
                Sample(
                    image_path=img_path,
                    mask_path=_find_mask_for_image(img_path, mask_paths_by_stem),
                    fabric_id=fabric_id,
                    defect_type=defect_type,
                    is_defective=True,
                    sample_id=img_path.stem,
                )
            )

        return samples

    def fabric_types(self, samples: List[Sample]) -> List[str]:
        return sorted({s.fabric_id for s in samples})

    def defect_types(self, samples: List[Sample]) -> List[str]:
        return sorted({s.defect_type for s in samples if s.defect_type is not None})
