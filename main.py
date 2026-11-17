
import argparse
import sys
import os

from dotenv import load_dotenv

# Load environment variables before importing components
load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cricket DRS - Decision Review System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
       
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input video file (default: from .env or v16_crop.mp4)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for output video (default: from .env or output_video.mp4)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Roboflow model ID (default: from .env or drs-5gvgi/3)",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show a live preview window during processing",
    )
    return parser.parse_args()


def check_environment():
    """Verify that required environment variables and files are set."""
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        print("[DRS] WARNING: ROBOFLOW_API_KEY not set in .env file")
        print("[DRS] Please add your API key to the .env file:")
        print('       ROBOFLOW_API_KEY=your_actual_key')
        print()


def main():
    """Entry point for the DRS system."""
    args = parse_args()
    check_environment()

    # Import here to avoid slow imports when just checking --help
    from drs_engine import DRSEngine

    print()
    print("=" * 60)
    print("    CRICKET DRS - Decision Review System")
    print("    Ball Tracking | LBW Analysis | Hawk-Eye")
    print("=" * 60)
    print()

    engine = DRSEngine(
        input_path=args.input,
        output_path=args.output,
        model_id=args.model,
        show_preview=args.show_preview,
    )

    try:
        engine.process()
    except FileNotFoundError as e:
        print(f"[DRS] ERROR: File not found — {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[DRS] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[DRS] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()