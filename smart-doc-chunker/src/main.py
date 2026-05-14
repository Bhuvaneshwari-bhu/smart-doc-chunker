import argparse
import os
import sys

import pipeline

# ---------------------------------------------------------------------------
# Method routing table
#
# Maps every user-facing CLI method name to the internal chunker method that
# backs it.  "fixed" and "sentence" are valid user choices but are routed to
# the nearest implemented strategy rather than raising an error:
#
#   fixed    → sliding (overlap=0 produces fixed-size, non-overlapping windows)
#   sentence → recursive (recursive already splits at sentence boundaries)
#
# When the mapping is lossy a notice is printed so the user is never silently
# surprised by the substitution.
# ---------------------------------------------------------------------------

_METHOD_MAP: dict[str, str] = {
    "recursive": "recursive",
    "sliding":   "sliding",
    "fixed":     "sliding",     # fixed-size = sliding with overlap forced to 0
    "sentence":  "recursive",   # recursive uses sentence-level splitting
}

_SUPPORTED_METHODS = sorted(_METHOD_MAP.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Document Chunking Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input document file (.pdf or .txt)",
    )
    parser.add_argument(
        "--method",
        default="recursive",
        choices=_SUPPORTED_METHODS,
        metavar="METHOD",
        help=(
            "Chunking strategy (default: recursive)\n"
            "  recursive — paragraph → sentence hierarchy with greedy merging\n"
            "  sliding   — overlapping token windows (uses --overlap)\n"
            "  fixed     — fixed-size windows, no overlap (alias: sliding + overlap=0)\n"
            "  sentence  — sentence-boundary splitting (alias: recursive)\n"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Maximum tokens per chunk (default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap between consecutive chunks; used by sliding (default: 50)",
    )
    return parser.parse_args()


def _resolve_method(method: str, overlap: int) -> tuple[str, int]:
    """
    Return the (internal_method, effective_overlap) pair for a user-facing
    method name.  Prints a notice when the mapping changes visible behaviour.
    """
    internal = _METHOD_MAP[method]

    if method == "fixed":
        print(
            f"[main] method 'fixed' → routed to 'sliding' with overlap forced to 0"
        )
        return internal, 0

    if method == "sentence":
        print(
            f"[main] method 'sentence' → routed to 'recursive' "
            f"(recursive splits at sentence boundaries)"
        )
        return internal, overlap

    return internal, overlap


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.overlap < 0:
        print("Error: --overlap must be >= 0", file=sys.stderr)
        sys.exit(1)

    internal_method, effective_overlap = _resolve_method(args.method, args.overlap)

    config = {
        "input":      args.input,
        "method":     internal_method,
        "chunk_size": args.chunk_size,
        "overlap":    effective_overlap,
    }

    print("=== Smart Doc Chunker ===")
    print(f"  input      : {config['input']}")
    print(f"  method     : {args.method}"
          + (f" → {internal_method}" if internal_method != args.method else ""))
    print(f"  chunk_size : {config['chunk_size']}")
    print(f"  overlap    : {config['overlap']}"
          + (" (forced to 0 for fixed)" if args.method == "fixed" else ""))
    print()

    pipeline.run(config)


if __name__ == "__main__":
    main()
