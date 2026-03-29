# sitecustomize.py
import os
if os.environ.get("PYHOOK_ENABLE", "0") == "1":
    import pyhook_agent
    pyhook_agent.bootstrap()
