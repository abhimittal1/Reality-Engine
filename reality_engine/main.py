import os
import tempfile
import sys

# Suppress Matplotlib cache warnings before importing any CV/ML libraries
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Suppress MediaPipe/GL warnings by setting quiet logging
os.environ["GLOG_minloglevel"] = "2" 

from reality_engine.core.engine import RealityEngine

def main():
    engine = RealityEngine()
    try:
        engine.start()
    except KeyboardInterrupt:
        print("\nShutting down Reality Engine...")
    finally:
        engine.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()
