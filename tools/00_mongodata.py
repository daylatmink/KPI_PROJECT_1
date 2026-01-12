import runpy
from pathlib import Path


def main():
    script_path = Path(__file__).resolve().parent / "mongodata3.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
