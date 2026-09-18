"""Shared data-integrity helpers for dataset loaders.

Corrupt/unreadable images are **quarantined**: excluded from the dataset
and recorded (path + reason) in the loader's ``quarantined`` list, with a
log line per file. They are never silently replaced by a placeholder
image while keeping their original label — that masks data corruption
and poisons both the label distribution and the reported dataset size.

Loaders call :func:`quarantine_corrupt` (or the pair-level variant for
image+mask datasets) on their full sample list BEFORE computing the
deterministic split, so split ratios stay in sync with what is actually
iterable.
"""

import logging
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

from PIL import Image

logger = logging.getLogger("crop_ssl.data")


class Quarantined(NamedTuple):
    """A sample excluded from the dataset, with the reason."""

    path: str
    reason: str


def probe_image(path) -> Optional[str]:
    """Open and fully decode an image.

    Returns:
        None if the image opens and decodes cleanly, otherwise a short
        failure reason string. ``Image.load()`` forces the full decode,
        which catches truncated files that ``Image.open`` alone would
        accept (it only reads the header).
    """
    try:
        with Image.open(path) as img:
            img.load()
        return None
    except Exception as e:  # noqa: BLE001 — any decode failure quarantines
        return f"{type(e).__name__}: {e}"


def quarantine_corrupt(
    samples: Sequence,
    probe=probe_image,
) -> Tuple[list, List[Quarantined]]:
    """Exclude samples whose image (first tuple element) fails to decode.

    Args:
        samples: Sample tuples whose FIRST element is the image path.
        probe: Readability probe (overridable for tests).

    Returns:
        (kept, quarantined): the filtered sample list and the list of
        :class:`Quarantined` exclusions. Every exclusion is logged.
    """
    kept, bad = [], []
    for sample in samples:
        reason = probe(sample[0])
        if reason is None:
            kept.append(sample)
        else:
            bad.append(Quarantined(str(sample[0]), reason))
    if bad:
        logger.warning(
            "Quarantined %d unreadable image(s); inspect `.quarantined`:",
            len(bad),
        )
        for q in bad:
            logger.warning("  excluded corrupt file: %s (%s)", q.path, q.reason)
    return kept, bad


def quarantine_pairs(
    samples: Sequence,
    probe=probe_image,
) -> Tuple[list, List[Quarantined]]:
    """Quarantine (image, mask, ...) sample PAIRS where either file fails.

    Used by image+mask datasets (e.g. PlantSeg): an image whose mask is
    unreadable — or a mask whose image is unreadable — is unusable in
    segmentation mode, so the pair is excluded together.
    """
    kept, bad = [], []
    for sample in samples:
        img_reason = probe(sample[0])
        mask_reason = probe(sample[1])
        if img_reason is None and mask_reason is None:
            kept.append(sample)
        else:
            reason = img_reason or mask_reason
            bad.append(Quarantined(f"{sample[0]} + {sample[1]}", str(reason)))
    if bad:
        logger.warning(
            "Quarantined %d unusable image/mask pair(s); inspect `.quarantined`:",
            len(bad),
        )
        for q in bad:
            logger.warning("  excluded pair: %s (%s)", q.path, q.reason)
    return kept, bad
