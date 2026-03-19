"""
Quick Data Explorer - View categorised bird data
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "OUTPUT"

def explore_categories():
    """Quick exploration of bird categories"""
    
    print("\n" + "="*70)
    print("AVONET BIRD CATEGORIES - QUICK EXPLORER")
    print("="*70)
    
    # Load main dataset
    df = pd.read_csv(OUTPUT_DIR / "birds_with_categories.csv")
    
    print(f"\nTotal birds: {len(df):,}")
    print(f"Total categories: {df['flight_category'].nunique()}")
    
    print(f"\n{'='*70}")
    print("CATEGORY BREAKDOWN")
    print("="*70)
    
    for category in sorted(df['flight_category'].unique()):
        cat_df = df[df['flight_category'] == category]
        count = len(cat_df)
        pct = count / len(df) * 100
        
        print(f"\n{category.upper()} ({count:,} birds, {pct:.1f}%)")
        print("-" * 70)
        
        # Stats
        print(f"  Avg Wing Length:    {cat_df['Wing.Length'].mean():8.1f} mm")
        print(f"  Avg Aspect Ratio:   {cat_df['aspect_ratio'].mean():8.2f}")
        print(f"  Avg Pointedness:    {cat_df['pointedness_index'].mean():8.3f}")
        print(f"  Avg Efficiency:     {cat_df['efficiency_index'].mean():8.3f}")
        
        # Sample species
        sample_species = cat_df['species'].head(5).tolist()
        print(f"\n  Sample species:")
        for species in sample_species:
            print(f"    - {species.replace('_', ' ')}")
    
    print(f"\n{'='*70}")
    print("TOP 10 MOST EFFICIENT BIRDS")
    print("="*70)
    
    top_efficient = df.nlargest(10, 'efficiency_index')[['species', 'flight_category', 
                                                          'efficiency_index', 'Wing.Length']]
    for idx, row in top_efficient.iterrows():
        print(f"{row['species'].replace('_', ' '):40s} | {row['flight_category']:15s} | "
              f"Efficiency: {row['efficiency_index']:.3f} | Wing: {row['Wing.Length']:.1f}mm")
    
    print(f"\n{'='*70}")
    print("FILES AVAILABLE FOR USE")
    print("="*70)
    
    files = [
        ("birds_with_categories.csv", "Main dataset with all categories"),
        ("normalized_data.csv", "MinMax scaled (0-1) for ML"),
        ("clean_data_with_indices.csv", "Raw data with calculated indices"),
        ("standardized_data.csv", "Z-score standardized"),
        ("summary_statistics.json", "JSON summary of categories"),
    ]
    
    for filename, description in files:
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename:35s} - {description:40s} ({size_mb:.2f} MB)")
    
    # Category files
    category_dir = OUTPUT_DIR / "categories"
    if category_dir.exists():
        print(f"\n  Category-specific files:")
        for cat_file in sorted(category_dir.glob("*.csv")):
            cat_name = cat_file.stem.replace('category_', '')
            cat_count = len(pd.read_csv(cat_file))
            print(f"    ✓ {cat_file.name:30s} - {cat_count:5,} birds")
    
    print(f"\n{'='*70}\n")


def search_species(search_term):
    """Search for specific bird species"""
    
    df = pd.read_csv(OUTPUT_DIR / "birds_with_categories.csv")
    results = df[df['species'].str.contains(search_term, case=False, na=False)]
    
    if len(results) == 0:
        print(f"\nNo species found matching '{search_term}'")
        return
    
    print(f"\nFound {len(results)} species matching '{search_term}':")
    print("="*70)
    
    for idx, row in results.iterrows():
        print(f"\n{row['species'].replace('_', ' ')}")
        print(f"  Category:        {row['flight_category'].upper()}")
        print(f"  Wing Length:     {row['Wing.Length']:.1f} mm")
        print(f"  Aspect Ratio:    {row['aspect_ratio']:.2f}")
        print(f"  Pointedness:     {row['pointedness_index']:.3f}")
        print(f"  Efficiency:      {row['efficiency_index']:.3f}")


def compare_categories(cat1, cat2):
    """Compare two flight categories"""
    
    df = pd.read_csv(OUTPUT_DIR / "birds_with_categories.csv")
    
    df1 = df[df['flight_category'] == cat1]
    df2 = df[df['flight_category'] == cat2]
    
    if len(df1) == 0:
        print(f"Category '{cat1}' not found")
        return
    if len(df2) == 0:
        print(f"Category '{cat2}' not found")
        return
    
    print(f"\nComparing {cat1.upper()} vs {cat2.upper()}")
    print("="*70)
    
    metrics = {
        'Wing.Length': 'Wing Length (mm)',
        'aspect_ratio': 'Aspect Ratio',
        'pointedness_index': 'Pointedness Index',
        'efficiency_index': 'Efficiency Index',
        'wing_loading_proxy': 'Wing Loading Proxy'
    }
    
    print(f"\n{'Metric':<25s} | {cat1.upper():>15s} | {cat2.upper():>15s} | Difference")
    print("-" * 70)
    
    for metric, label in metrics.items():
        val1 = df1[metric].mean()
        val2 = df2[metric].mean()
        diff = val2 - val1
        diff_pct = (diff / val1 * 100) if val1 != 0 else 0
        
        print(f"{label:<25s} | {val1:15.3f} | {val2:15.3f} | {diff:+.3f} ({diff_pct:+.1f}%)")
    
    print(f"\nSample size: {len(df1)} vs {len(df2)} birds")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "search" and len(sys.argv) > 2:
            search_species(sys.argv[2])
        
        elif command == "compare" and len(sys.argv) > 3:
            compare_categories(sys.argv[2], sys.argv[3])
        
        else:
            print("Usage:")
            print("  python explore_data.py                    - Show all categories")
            print("  python explore_data.py search <term>      - Search for species")
            print("  python explore_data.py compare cat1 cat2  - Compare categories")
    
    else:
        # Default: show all categories
        explore_categories()
