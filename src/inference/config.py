"""Single knob for the local inference engine: which checkpoint to load.

Overridable via the MODEL_NAME environment variable; defaults to a
tiny model so the engine is usable for experimentation without a GPU.
"""

import os

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
