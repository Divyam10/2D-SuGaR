#!/usr/bin/env python3
"""
Script to compute average metrics from DTU evaluation results
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def read_results_json(filepath):
    """Read a single results.json file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    base_dir = Path("results/dtu")
    

    # Find all results.json files
    results_files = list(base_dir.glob("*/results.json"))
    # results_files = list(base_dir.glob("*/*/results.json"))

    if not results_files:
        print("No results.json files found!")
        return

    print(f"Found {len(results_files)} results files")
    print("=" * 80)

    # Store all metrics
    all_metrics = defaultdict(list)
    scene_results = {}

    # Read all results
    for results_file in sorted(results_files):
        scene_name = results_file.parent.name
        # scene_name = results_file.parent.parent.name
        data = read_results_json(results_file)

        if data is None:
            continue

        scene_results[scene_name] = data

        # Collect metrics
        for key, value in data.items():
            if isinstance(value, (int, float)):
                all_metrics[key].append(value)

    # Print individual scene results
    print("\nIndividual Scene Results:")
    print("-" * 80)

    for scene_name in sorted(scene_results.keys()):
        print(f"\n{scene_name}:")
        for key, value in scene_results[scene_name].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

    # Compute and print averages
    print("\n" + "=" * 80)
    print("AVERAGE METRICS ACROSS ALL SCENES:")
    print("=" * 80)

    avg_metrics = {}
    for key, values in sorted(all_metrics.items()):
        avg = sum(values) / len(values)
        avg_metrics[key] = avg
        print(f"{key}: {avg:.4f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE:")
    print("=" * 80)

    # Get all metric keys
    metric_keys = sorted(all_metrics.keys())

    # Print header
    print(f"{'Scene':<15}", end="")
    for key in metric_keys:
        print(f"{key:<12}", end="")
    print()
    print("-" * (15 + 12 * len(metric_keys)))

    # Print each scene
    for scene_name in sorted(scene_results.keys()):
        print(f"{scene_name:<15}", end="")
        for key in metric_keys:
            value = scene_results[scene_name].get(key, 0)
            print(f"{value:<12.4f}", end="")
        print()

    # Print average row
    print("-" * (15 + 12 * len(metric_keys)))
    print(f"{'AVERAGE':<15}", end="")
    for key in metric_keys:
        avg = avg_metrics.get(key, 0)
        print(f"{avg:<12.4f}", end="")
    print()

    print("\n" + "=" * 80)
    print(f"Total scenes processed: {len(scene_results)}")
    print("=" * 80)

    # Save summary to file
    summary_file = base_dir / "summary_metrics.json"
    summary_data = {
        "averages": avg_metrics,
        "scenes": scene_results,
        "num_scenes": len(scene_results)
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()
