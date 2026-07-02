"""Module entry point so ``python -m star_reacher`` behaves identically to
the installed ``star`` console script: both dispatch through ``cli.main``."""

import sys

from star_reacher.cli import main

if __name__ == "__main__":
    sys.exit(main())
