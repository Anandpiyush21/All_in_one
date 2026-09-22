#!/usr/bin/env python3
"""Entry point.

    python main.py                 # guided prompts
    python main.py --help          # every option
    python main.py --all-models    # benchmark the whole registry
"""

import sys

from omniclf.cli import main

if __name__ == "__main__":
    sys.exit(main())
