# TODO: add the context for testing if neccessary
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jestr.data.datasets as datasets
