import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def main():
    parser = argparse.ArgumentParser(
        description="Build missing EDIP artifacts and optionally launch the Streamlit app."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Build missing artifacts and launch the Streamlit dashboard.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding all artifacts before serving.",
    )
    args = parser.parse_args()

    if not args.serve:
        parser.print_help()
        return

    build_cmd = [PYTHON, str(ROOT / "build_all.py")]
    if args.force:
        build_cmd.append("--force")

    result = subprocess.run(build_cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

    streamlit_cmd = [PYTHON, "-m", "streamlit", "run", "app.py"]
    subprocess.run(streamlit_cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
