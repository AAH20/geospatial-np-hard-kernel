#!/usr/bin/env python3
"""Root executable CLI entrypoint for Geospatial NP-Hard Kernel."""
import sys
from pathlib import Path

# Ensure root package is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from geospatial_np_hard_kernel.cli import main

if __name__ == "__main__":
    main()
