"""Command-line entry point: ``clinsum``."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__


def _prepare(args: argparse.Namespace) -> None:
    from .data.loaders import build_pairs, load_notes, load_targets

    pairs = build_pairs(load_notes(args.notes), load_targets(args.targets),
                        remove_reference=not args.keep_reference, expand_abbrev=args.expand_abbreviations,
                        seed=args.seed)
    if args.split != "all":
        pairs = pairs[pairs["split"] == args.split]
    if args.limit:
        pairs = pairs.sort_values("note_id").head(args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out, index=False) if out.suffix == ".parquet" else pairs.to_csv(out, index=False)
    removed = pairs["bhc_removed"].mean() if len(pairs) else 0.0
    print(f"wrote {len(pairs):,} pairs to {out}  (BHC section removed from {removed:.1%} of inputs)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="clinsum", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="build note_id-aligned (input, target) pairs with the BHC removed")
    prep.add_argument("--notes", required=True)
    prep.add_argument("--targets", required=True)
    prep.add_argument("--out", required=True, help=".csv or .parquet")
    prep.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    prep.add_argument("--limit", type=int, default=0)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--expand-abbreviations", action="store_true")
    prep.add_argument("--keep-reference", action="store_true", help="do NOT remove the BHC (for leakage ablations)")
    prep.set_defaults(func=_prepare)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
