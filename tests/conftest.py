"""Keep test jobs out of the developer's data directory."""

import os
import tempfile

os.environ.setdefault("PARTCAPTURE_DATA", tempfile.mkdtemp(prefix="trueshape-test-"))
