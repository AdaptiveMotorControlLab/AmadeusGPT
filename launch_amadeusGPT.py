# launch_amadeusGPT.py
from amadeusgpt.app import main

if __name__ == "__main__":
    import subprocess
    import sys

    cmd = ["streamlit", "run", "app.py"]
    subprocess.run(cmd + sys.argv[1:])