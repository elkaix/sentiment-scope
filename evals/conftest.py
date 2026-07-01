"""Make ``run_eval`` importable from the eval test suite.

The eval CLI lives at ``evals/run_eval.py`` (repo-root relative), so we add
this directory to ``sys.path`` explicitly rather than relying on pytest's
import-mode inference — an explicit insert never silently fails collection.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
