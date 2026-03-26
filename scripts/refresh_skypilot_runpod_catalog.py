"""Refresh SkyPilot's RunPod catalog with all zones from the RunPod API.

SkyPilot's built-in RunPod fetcher has a hardcoded REGION_ZONES dict that
misses many datacenters. This script:
  1. Queries the RunPod dataCenters API for all available zones
  2. Patches the REGION_ZONES in SkyPilot's fetch_runpod module
  3. Runs the fetcher to regenerate ~/.sky/catalogs/v8/runpod/vms.csv
  4. Removes the MD5 checksum so SkyPilot won't overwrite our catalog

Requires: RUNPOD_API_KEY env var, runpod + pandas packages (from skypilot).

Usage:
    python scripts/refresh_skypilot_runpod_catalog.py
    python scripts/refresh_skypilot_runpod_catalog.py --no-cpu  # skip CPU instances (faster)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CATALOG_DIR = Path.home() / ".sky" / "catalogs" / "v8" / "runpod"
CATALOG_CSV = CATALOG_DIR / "vms.csv"
CATALOG_MD5 = Path.home() / ".sky" / "catalogs" / "v8" / ".meta" / "runpod" / "vms.csv.md5"


def fetch_datacenters() -> list[dict]:
    import runpod
    from runpod.api import graphql

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    result = graphql.run_graphql_query("{ dataCenters { id name location } }")
    if "errors" in result:
        raise RuntimeError(f"GraphQL errors: {result['errors']}")
    return result["data"]["dataCenters"]


def build_region_zones(datacenters: list[dict]) -> dict[str, list[str]]:
    regions: dict[str, list[str]] = {}
    for dc in datacenters:
        zone_id = dc["id"]
        # EUR-IS-1 -> EUR, US-MO-2 -> US, CA-MTL-1 -> CA, etc.
        key = zone_id.split("-")[0]
        # Normalize EUR -> EU for consistency
        if key == "EUR":
            key = "EU"
        regions.setdefault(key, []).append(zone_id)
    for key in regions:
        regions[key].sort()
    return dict(sorted(regions.items()))


def main():
    parser = argparse.ArgumentParser(description="Refresh SkyPilot RunPod catalog")
    parser.add_argument("--no-cpu", action="store_true", help="Skip CPU instances (faster)")
    parser.add_argument("--dry-run", action="store_true", help="Show zones but don't regenerate")
    parser.add_argument(
        "--prefer-regions",
        default="US,CA,EU",
        help="Comma-separated region preference order (default: US,CA,EU)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    print("Fetching datacenters from RunPod API...")
    datacenters = fetch_datacenters()
    region_zones = build_region_zones(datacenters)

    total_zones = sum(len(zones) for zones in region_zones.values())
    print(f"Found {total_zones} zones across {len(region_zones)} regions:")
    for region, zones in region_zones.items():
        print(f"  {region}: {', '.join(zones)}")

    if args.dry_run:
        return

    # Patch SkyPilot's fetch_runpod module
    print("\nPatching SkyPilot's REGION_ZONES...")
    from sky.catalog.data_fetchers import fetch_runpod

    fetch_runpod.REGION_ZONES = region_zones

    # Run the fetcher
    print("Regenerating catalog (this takes a few minutes)...")
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    fetcher_args = ["--output-dir", str(CATALOG_DIR)]
    if args.no_cpu:
        fetcher_args.append("--no-cpu")
    old_argv = sys.argv
    try:
        sys.argv = ["fetch_runpod"] + fetcher_args
        fetch_runpod.main()
    finally:
        sys.argv = old_argv

    # Reorder catalog rows to prefer specified regions
    if args.prefer_regions:
        import pandas as pd

        preferred = [r.strip() for r in args.prefer_regions.split(",")]
        df = pd.read_csv(CATALOG_CSV)
        region_rank = {r: i for i, r in enumerate(preferred)}
        default_rank = len(preferred)
        df["_rank"] = df["Region"].map(lambda r: region_rank.get(r, default_rank))
        df.sort_values(
            ["AcceleratorName", "AcceleratorCount", "_rank", "AvailabilityZone"],
            inplace=True,
        )
        df.drop(columns=["_rank"], inplace=True)
        df.to_csv(CATALOG_CSV, index=False)
        print(f"Reordered catalog with region preference: {', '.join(preferred)}")

    # Remove MD5 so SkyPilot won't overwrite our catalog on auto-refresh
    if CATALOG_MD5.exists():
        CATALOG_MD5.unlink()
        print(f"Removed {CATALOG_MD5} to prevent auto-overwrite")

    print(f"\nCatalog written to {CATALOG_CSV}")
    if CATALOG_CSV.exists():
        import csv

        with open(CATALOG_CSV) as f:
            row_count = sum(1 for _ in csv.reader(f)) - 1
        print(f"Total entries: {row_count}")


if __name__ == "__main__":
    main()
