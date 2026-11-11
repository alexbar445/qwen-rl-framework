"""ARC tool-agent DAPO runtime."""
import sys as _sys

# An installed Learner bundle treats its source closure as immutable. Preserve
# that boundary for direct host CLI calls as well as the systemd unit.
if len(_sys.argv) > 1 and _sys.argv[1] == "learner":
    _sys.dont_write_bytecode = True

del _sys

__version__ = "0.1.0"
