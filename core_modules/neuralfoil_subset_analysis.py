"""
Subset NeuralFoil Analysis - Quick Test Version
================================================
Runs NeuralFoil on a representative sample from each category.
Much faster than full analysis while still providing meaningful results.

This version analyzes ~50 birds (10 per category) instead of all 10,975.
Estimated time: ~1 hour instead of 1,200 hours.
"""

import sys
from pathlib import Path
import pandas as pd

# Import from main analysis script
from neuralfoil_analysis import (
    analyze_all_dat_files, 
    AIRFOIL_DIR, 
    PROJECT_ROOT,
    REYNOLDS_NUMBERS,
    ALPHAS
)

# Configuration for subset analysis
SUBSET_SIZE_PER_CATEGORY = 10  # How many birds per category to analyze
OUTPUT_DIR = PROJECT_ROOT / "RESULTS" / "neuralfoil_subset_test"

def get_representative_sample(airfoil_dir: Path, n_per_category: int = 10):
    """
    Get a stratified sample of airfoils - N from each category.
    
    Selection strategy:
    - Evenly spaced indices from each category (start, middle, end samples)
    - Ensures representation across morphological diversity
    """
    from pathlib import Path
    import numpy as np
    
    # Get all .dat files organized by category
    categories = {}
    all_files = list(airfoil_dir.rglob("*.dat"))
    
    for f in all_files:
        cat = f.parent.name
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(f)
    
    # Sample evenly from each category
    selected = []
    for cat, files in categories.items():
        if len(files) <= n_per_category:
            selected.extend(files)
        else:
            # Evenly spaced indices
            indices = np.linspace(0, len(files)-1, n_per_category, dtype=int)
            selected.extend([files[i] for i in indices])
    
    return sorted(selected)


if __name__ == "__main__":
    print("=" * 72)
    print("NEURALFOIL SUBSET ANALYSIS - QUICK TEST")
    print("=" * 72)
    print()
    print(f"Configuration:")
    print(f"  Sample size per category : {SUBSET_SIZE_PER_CATEGORY}")
    print(f"  Reynolds numbers        : {len(REYNOLDS_NUMBERS)}")
    print(f"  Alpha values            : {len(ALPHAS)}")
    print(f"  Conditions per airfoil  : {len(ALPHAS) * len(REYNOLDS_NUMBERS)}")
    print()
    
    # Get category counts
    categories = {}
    all_files = list(AIRFOIL_DIR.rglob("*.dat"))
    for f in all_files:
        cat = f.parent.name
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"Available birds by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:15s}: {count:5d} birds")
    
    # Calculate expected subset size
    n_categories = len(categories)
    expected_total = min(n_categories * SUBSET_SIZE_PER_CATEGORY, len(all_files))
    expected_sims = expected_total * len(ALPHAS) * len(REYNOLDS_NUMBERS)
    expected_time_hrs = expected_sims * 5 / 3600  # 5 sec per sim
    
    print()
    print(f"Subset analysis will include:")
    print(f"  Total airfoils   : ~{expected_total}")
    print(f"  Total simulations: {expected_sims:,}")
    print(f"  Est. time        : ~{expected_time_hrs:.1f} hours")
    print()
    
    # Confirm
    response = input("Proceed with subset analysis? [y/N]: ").strip().lower()
    if response != 'y':
        print("Analysis cancelled.")
        sys.exit(0)
    
    print()
    print("Getting representative sample...")
    sample_files = get_representative_sample(AIRFOIL_DIR, SUBSET_SIZE_PER_CATEGORY)
    
    print(f"Sample breakdown:")
    sample_cats = {}
    for f in sample_files:
        cat = f.parent.name
        sample_cats[cat] = sample_cats.get(cat, 0) + 1
    for cat, count in sorted(sample_cats.items()):
        print(f"  {cat:15s}: {count:3d} airfoils")
    print()
    
    # Create temporary directory with symlinks to sample files
    print("Setting up temporary analysis directory...")
    temp_dir = PROJECT_ROOT / "TEMP_subset_airfoils"
    temp_dir.mkdir(exist_ok=True)
    
    for f in sample_files:
        cat_dir = temp_dir / f.parent.name
        cat_dir.mkdir(exist_ok=True)
        link_path = cat_dir / f.name
        if not link_path.exists():
            try:
                link_path.symlink_to(f)
            except:
                # Fallback: copy file if symlink fails
                import shutil
                shutil.copy2(f, link_path)
    
    print(f"Temporary directory: {temp_dir}")
    print()
    print("=" * 72)
    print("Starting NeuralFoil Analysis")
    print("=" * 72)
    print()
    
    # Run analysis on subset
    detailed_df, summary_df, category_leaders = analyze_all_dat_files(
        temp_dir,
        OUTPUT_DIR
    )
    
    # Cleanup
    print()
    print("Cleaning up temporary directory...")
    import shutil
    shutil.rmtree(temp_dir)
    
    # Print summary
    if not summary_df.empty:
        print()
        print("=" * 72)
        print("SUBSET ANALYSIS COMPLETE")
        print("=" * 72)
        print(f"Results saved to: {OUTPUT_DIR}")
        print()
        print("Key findings (subset only - not representative of full dataset):")
        print(f"  Best L/D      : {summary_df['max_LD'].max():.2f}")
        print(f"  Mean L/D      : {summary_df['max_LD'].mean():.2f}")
        print(f"  Best CLmax    : {summary_df['CL_max'].max():.3f}")
        print(f"  Mean CLmax    : {summary_df['CL_max'].mean():.3f}")
        print()
        print("Top 5 performers in subset:")
        top5 = summary_df.nlargest(5, 'max_LD')[['bird_name', 'category', 'max_LD', 'CL_max']]
        for i, row in top5.iterrows():
            print(f"  {row['bird_name'][:30]:30s} ({row['category']:12s}) "
                  f"L/D={row['max_LD']:6.2f}  CLmax={row['CL_max']:.3f}")
        print()
        print("NOTE: This is a subset analysis. For publication-quality results,")
        print("      run the full analysis with: python neuralfoil_analysis.py")
        print("=" * 72)
    else:
        print()
        print("✗ Analysis failed - check logs above for errors")
