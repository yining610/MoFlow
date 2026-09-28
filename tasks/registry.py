"""Registry of real HuggingFace benchmarks the CLI can build on.
"""

from .aime import AIME_2026
from .gpqa import GPQA_DIAMOND
from .hf_dataset import HFDatasetSpec
from .hotpotqa import HOTPOT_HARD
from .math import MATH_L5
from .mbpp import MBPP_SANITIZED
from .swe import SWE_LITE

DATASETS: dict[str, HFDatasetSpec] = {
    "aime": AIME_2026,
    "math": MATH_L5,
    "mbpp": MBPP_SANITIZED,
    "swe": SWE_LITE,
    "gpqa": GPQA_DIAMOND,
    "hotpotqa": HOTPOT_HARD,
}
