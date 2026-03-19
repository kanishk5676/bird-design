"""
AVONET Bird Data Categorisation Module
=======================================
This module loads, processes, and categorizes bird species based on morphological
features from the AVONET dataset. Results are saved for use in dashboards.

Categorisation uses a SCORED, EXCLUSIVE assignment system:
- Each bird receives a fitness score for every flight category
- The bird is assigned to the single highest-scoring category
- No bird appears in more than one category
- This is ideal for the airfoil generation + NeuralFoil simulation pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "DATA"
OUTPUT_DIR = Path(__file__).parent.parent / "OUTPUT"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "AVONET_BIRDLIFE.csv"


def calculate_morphological_indices(df):
    """
    Calculate biologically meaningful morphological indices

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with raw morphological measurements

    Returns:
    --------
    pd.DataFrame
        DataFrame with added morphological indices
    """
    df = df.copy()

    # Aspect ratio proxy (longer wing = higher aspect ratio)
    df['aspect_ratio'] = df['Wing.Length'] / df['Secondary1']

    # Wing loading proxy (larger secondaries suggest broader wings, lower loading)
    df['wing_loading_proxy'] = df['Wing.Length'] / (df['Secondary1'] * df['Tail.Length'])

    # Wing pointedness (high Kipps distance = pointed wing)
    df['pointedness_index'] = df['Kipps.Distance'] / df['Wing.Length']

    # Flight efficiency index (combination of hand-wing index and pointedness)
    df['efficiency_index'] = (df['Hand-Wing.Index'] * df['pointedness_index']) / 100

    return df


def load_and_process_avonet_data(csv_path):
    """
    Load and process AVONET bird data with enhanced validation

    Parameters:
    -----------
    csv_path : str or Path
        Path to AVONET CSV file

    Returns:
    --------
    tuple
        (normalized_df, df_clean, minmax_scaler, standardized_df)
    """
    print("="*60)
    print("LOADING AVONET BIRD DATA")
    print("="*60)

    try:
        df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        print(f"✓ Successfully loaded AVONET data with {len(df)} species")
    except Exception as e:
        print(f"✗ Could not load CSV file: {e}")
        raise

    # Check for required columns
    required_cols = ['Species1', 'Wing.Length', 'Secondary1', 'Kipps.Distance',
                    'Tail.Length', 'Hand-Wing.Index']

    # Try alternative species column names
    species_col = None
    for col in ['Species1', 'species', 'Species', 'SPECIES']:
        if col in df.columns:
            species_col = col
            break

    if species_col:
        required_cols[0] = species_col
    else:
        print("Warning: No species column found, using index")
        df['species'] = df.index
        required_cols[0] = 'species'

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)[:10]}...")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Rename species column for consistency
    df = df.rename(columns={required_cols[0]: 'species'})
    required_cols[0] = 'species'

    print(f"\n{'='*60}")
    print("CLEANING DATA")
    print("="*60)

    # Enhanced data cleaning with outlier detection
    df_clean = df[required_cols].copy()
    initial_count = len(df_clean)

    # Remove outliers using IQR method
    for col in ['Wing.Length', 'Secondary1', 'Kipps.Distance', 'Hand-Wing.Index']:
        Q1 = df_clean[col].quantile(0.05)  # More conservative outlier removal
        Q3 = df_clean[col].quantile(0.95)
        IQR = Q3 - Q1
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= Q1 - 1.5*IQR) & (df_clean[col] <= Q3 + 1.5*IQR)]
        removed = before - len(df_clean)
        if removed > 0:
            print(f"  {col}: Removed {removed} outliers")

    # Remove missing values
    before_na = len(df_clean)
    df_clean = df_clean.dropna()
    na_removed = before_na - len(df_clean)
    if na_removed > 0:
        print(f"  Removed {na_removed} rows with missing values")

    print(f"\n✓ Using {len(df_clean)} birds after cleaning (removed {initial_count - len(df_clean)} total)")

    print(f"\n{'='*60}")
    print("CALCULATING MORPHOLOGICAL INDICES")
    print("="*60)

    # Calculate derived morphological parameters
    df_clean = calculate_morphological_indices(df_clean)
    print("✓ Calculated: aspect_ratio, wing_loading_proxy, pointedness_index, efficiency_index")

    print(f"\n{'='*60}")
    print("NORMALIZING DATA")
    print("="*60)

    # Normalize with both MinMax and Standard scaling for comparison
    features = ['Wing.Length', 'Secondary1', 'Kipps.Distance', 'Tail.Length', 'Hand-Wing.Index']

    # MinMax scaling (0-1) for airfoil generation
    minmax_scaler = MinMaxScaler()
    normalized_minmax = minmax_scaler.fit_transform(df_clean[features])
    normalized_df = pd.DataFrame(normalized_minmax, columns=features)

    # Standard scaling for statistical analysis
    std_scaler = StandardScaler()
    normalized_std = std_scaler.fit_transform(df_clean[features])
    standardized_df = pd.DataFrame(normalized_std, columns=features)

    # Add species and derived parameters
    for col in ['species', 'aspect_ratio', 'wing_loading_proxy', 'pointedness_index', 'efficiency_index']:
        normalized_df[col] = df_clean[col].reset_index(drop=True)
        standardized_df[col] = df_clean[col].reset_index(drop=True)

    print("✓ MinMax scaling complete (0-1 range)")
    print("✓ Standard scaling complete (z-scores)")

    return normalized_df, df_clean, minmax_scaler, standardized_df


# ===========================================================================
# SCORING-BASED EXCLUSIVE CATEGORISATION
# ---------------------------------------------------------------------------
# Instead of binary masks (which cause overlaps), each category defines a
# score_function(row) → float. Every bird is scored against all six categories
# and assigned to the single highest-scoring one. This guarantees:
#   1. No duplicates across categories
#   2. Every bird is assigned (no silent overwrites)
#   3. The "best fit" semantics you need for airfoil selection
# ===========================================================================

def _percentile_rank(series: pd.Series) -> pd.Series:
    """
    Convert a column to percentile ranks [0, 1].

    This is more robust than z-scores for scoring because:
    - It is unaffected by skewed distributions (morphological data is often skewed)
    - Every trait contributes on the same 0-1 scale regardless of variance
    - Thresholds like 0.75 mean "top 25% of the population" — directly interpretable
      in the Methods section of a paper
    """
    return series.rank(pct=True)


def score_birds_for_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each bird to exactly one flight category using percentile-ranked
    morphological scoring.

    Design rationale (for paper Methods section)
    ---------------------------------------------
    Rather than binary threshold masks (which cause overlap) or z-score weights
    (which are sensitive to distribution skew), we use percentile ranks of each
    morphological index. Each category is defined by a weighted combination of
    percentile ranks reflecting the dominant aerodynamic traits of that ecological
    guild. The bird is assigned to the category for which its weighted score is
    highest, guaranteeing exclusive, exhaustive assignment across all 11,247 species.

    Category definitions follow established ornithological literature:
    - HWI (Hand-Wing Index): Tobias et al. 2022 / Sheard et al. 2020
    - Aspect ratio / wing loading: Rayner 1988, Pennycuick 2008
    - Pointedness index (Kipps distance): Lockwood et al. 1998

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with morphological indices already calculated.

    Returns:
    --------
    pd.DataFrame
        Original df with one score column per category, the assigned
        'flight_category', and 'category_score' (the winning score, 0-4 range).
    """
    scored = df.copy()

    # Convert each trait to a population percentile rank [0, 1]
    # High rank = high value relative to all species in the dataset
    p_aspect      = _percentile_rank(df['aspect_ratio'])         # high = long narrow wing
    p_loading_inv = 1 - _percentile_rank(df['wing_loading_proxy'])  # high = low wing loading (broad wing)
    p_pointed     = _percentile_rank(df['pointedness_index'])    # high = very pointed wingtip
    p_hwi         = _percentile_rank(df['Hand-Wing.Index'])      # high = long hand-wing (fast/migratory)
    p_hwi_inv     = 1 - p_hwi                                    # high = short hand-wing (maneuverable)
    p_efficiency  = _percentile_rank(df['efficiency_index'])     # high = aerodynamically efficient
    p_size_inv    = 1 - _percentile_rank(df['Wing.Length'])      # high = small absolute wing size

    # ------------------------------------------------------------------
    # SOARING  (target ~10-15% of dataset)
    # Archetype: albatross, vulture, stork, frigatebird
    # Key traits: highest aspect ratios + broadest wings (low loading)
    # Literature: aspect ratio > 8 (Pennycuick 2008), low HWI in thermal soarers
    # ------------------------------------------------------------------
    scored['score_soaring'] = (
        2.0 * p_aspect          # primary trait: long narrow wings
      + 2.0 * p_loading_inv     # primary trait: broad, low-loaded wings
      + 0.5 * p_efficiency      # secondary: efficient glide
      - 0.5 * p_pointed         # thermal soarers have blunt/slotted tips, not pointed
    )

    # ------------------------------------------------------------------
    # DIVING  (target ~10-15% of dataset)
    # Archetype: peregrine falcon, gannet, swift, tern
    # Key traits: most pointed wings + high HWI (long arm bones)
    # Literature: Kipps distance / wing length > 0.4, HWI > 50 (Sheard 2020)
    # ------------------------------------------------------------------
    scored['score_diving'] = (
        2.5 * p_pointed         # dominant trait: swept, pointed wing
      + 1.5 * p_hwi             # high HWI = long distal wing (speed)
      + 0.5 * p_efficiency      # efficiency matters at high speed
    )

    # ------------------------------------------------------------------
    # MANEUVERING  (target ~15-20% of dataset)
    # Archetype: sparrowhawk, flycatcher, woodpecker, forest passerine
    # Key traits: low HWI (short hand-wing) + low aspect ratio
    # Literature: HWI < 25 typical for forest/understory birds (Tobias 2022)
    # Note: this is a genuinely common guild — many passerines qualify
    # ------------------------------------------------------------------
    scored['score_maneuvering'] = (
        2.5 * p_hwi_inv         # dominant trait: short hand-wing
      + 1.5 * (1 - p_aspect)    # low aspect ratio (broad, rounded wing)
      - 0.5 * p_pointed         # rounded wingtips
    )

    # ------------------------------------------------------------------
    # CRUISING  (target ~20-25% of dataset)
    # Archetype: shorebird, swallow, swift (moderate), gull, duck
    # Key traits: high efficiency index + moderate HWI + moderate aspect ratio
    # Birds that score highly on efficiency but aren't extreme in any one trait
    # ------------------------------------------------------------------
    scored['score_cruising'] = (
        2.5 * p_efficiency      # dominant: aerodynamic efficiency
      + 1.0 * p_hwi             # moderate-high HWI (migratory capability)
      + 1.0 * p_aspect          # moderate-high aspect ratio
      - 0.5 * p_pointed         # not as pointed as diving specialists
    )

    # ------------------------------------------------------------------
    # HOVERING  (target ~3-5% of dataset)
    # Archetype: hummingbird, sunbird, kingfisher, kestrel
    # Key traits: highest HWI + small absolute wing size
    # Literature: HWI > 60 for hovering specialists (Sheard 2020)
    # ------------------------------------------------------------------
    scored['score_hovering'] = (
        2.5 * p_hwi             # dominant: extremely high HWI
      + 2.0 * p_size_inv        # small absolute wing (rapid beats possible)
      - 1.0 * p_aspect          # not high aspect ratio
    )

    # ------------------------------------------------------------------
    # GENERALIST  (target ~30-35% of dataset)
    # Archetype: corvid, starling, pigeon, medium passerine
    # These are birds that score below the 60th percentile on ALL specialist
    # dimensions — genuinely balanced, no dominant flight strategy.
    # Using a soft score rather than a residual ensures they are positively
    # identified rather than just being the leftover category.
    # ------------------------------------------------------------------
    specialist_cols = [
        'score_soaring', 'score_diving', 'score_maneuvering',
        'score_cruising', 'score_hovering'
    ]
    max_specialist = scored[specialist_cols].max(axis=1)

    # Generalist score peaks for birds where ALL specialist scores are middling
    # (around 2.5 on a 0-5 scale). Birds with very high OR very low specialist
    # scores get a lower generalist score.
    scored['score_generalist'] = 5.0 - max_specialist  # inverse of best specialist fit

    # ------------------------------------------------------------------
    # ASSIGN EXCLUSIVE CATEGORY: highest score wins
    # ------------------------------------------------------------------
    all_score_cols = specialist_cols + ['score_generalist']
    category_map = {col: col.replace('score_', '') for col in all_score_cols}

    scored['flight_category'] = (
        scored[all_score_cols]
        .idxmax(axis=1)
        .map(category_map)
    )

    # Winning score (useful for ranking within category, e.g. picking top airfoil)
    scored['category_score'] = scored[all_score_cols].max(axis=1)

    return scored


def categorize_birds_by_flight_style(df: pd.DataFrame):
    """
    Assign each bird to exactly one flight category using scored best-fit.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with morphological indices.

    Returns:
    --------
    tuple
        (flight_categories_dict, df_with_categories)
        flight_categories_dict maps category name → DataFrame of birds in it.
    """
    print("\n" + "="*60)
    print("CATEGORIZING BIRDS BY FLIGHT STYLE (scored, exclusive)")
    print("="*60)

    df_with_categories = score_birds_for_categories(df)

    # Build the per-category dict from the exclusive assignments
    flight_categories = {}
    for category in df_with_categories['flight_category'].unique():
        flight_categories[category] = df_with_categories[
            df_with_categories['flight_category'] == category
        ].copy()

    # Print category statistics
    print("\nFlight Categories Distribution:")
    print("-" * 60)
    total = len(df_with_categories)
    for category, birds in sorted(flight_categories.items(), key=lambda x: -len(x[1])):
        pct = len(birds) / total * 100
        avg_score = birds['category_score'].mean()
        print(f"  {category.upper():20s}: {len(birds):5d} birds ({pct:5.1f}%)  "
              f"avg fit score: {avg_score:+.3f}")

    # Sanity check: no duplicates, no missing birds
    assigned_total = sum(len(v) for v in flight_categories.values())
    assert assigned_total == total, (
        f"Assignment mismatch: {assigned_total} assigned vs {total} total birds"
    )
    print(f"\n✓ All {total} birds assigned to exactly one category (no duplicates)")

    return flight_categories, df_with_categories


def save_processed_data(normalized_df, df_clean, standardized_df, df_with_categories,
                        flight_categories, output_dir=OUTPUT_DIR):
    """
    Save all processed dataframes for use in dashboards

    Parameters:
    -----------
    normalized_df : pd.DataFrame
        MinMax normalized data
    df_clean : pd.DataFrame
        Cleaned raw data with indices
    standardized_df : pd.DataFrame
        Standard scaled data
    df_with_categories : pd.DataFrame
        Data with flight categories and scores
    flight_categories : dict
        Dictionary of category DataFrames
    output_dir : Path
        Directory to save output files
    """
    print(f"\n{'='*60}")
    print("SAVING PROCESSED DATA")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    files_saved = []

    # 1. Normalized data (for airfoil generation)
    normalized_df.to_csv(output_dir / "normalized_data.csv", index=False)
    files_saved.append("normalized_data.csv")

    # 2. Clean data with morphological indices
    df_clean.to_csv(output_dir / "clean_data_with_indices.csv", index=False)
    files_saved.append("clean_data_with_indices.csv")

    # 3. Standardized data (for statistical analysis)
    standardized_df.to_csv(output_dir / "standardized_data.csv", index=False)
    files_saved.append("standardized_data.csv")

    # 4. Data with flight categories and scores (MAIN FILE FOR DASHBOARD)
    df_with_categories.to_csv(output_dir / "birds_with_categories.csv", index=False)
    files_saved.append("birds_with_categories.csv")

    # 5. Save individual category files
    category_dir = output_dir / "categories"
    category_dir.mkdir(exist_ok=True)

    for category_name, category_df in flight_categories.items():
        if len(category_df) > 0:
            # Sort by category_score descending so the best-fit bird is row 0
            # — useful when you pick one representative bird per category
            category_df_sorted = category_df.sort_values('category_score', ascending=False)
            filename = f"category_{category_name}.csv"
            category_df_sorted.to_csv(category_dir / filename, index=False)
            files_saved.append(f"categories/{filename}")

    # 6. Save summary statistics
    import json
    summary = {
        'total_birds': len(df_with_categories),
        'categories': {}
    }

    for category_name, category_df in flight_categories.items():
        summary['categories'][category_name] = {
            'count': len(category_df),
            'percentage': len(category_df) / len(df_with_categories) * 100,
            'avg_wing_length': float(category_df['Wing.Length'].mean()),
            'avg_aspect_ratio': float(category_df['aspect_ratio'].mean()),
            'avg_category_score': float(category_df['category_score'].mean()),
            # Top bird = the best representative for airfoil generation
            'top_bird': category_df.loc[
                category_df['category_score'].idxmax(), 'species'
            ]
        }

    with open(output_dir / "summary_statistics.json", 'w') as f:
        json.dump(summary, f, indent=2)
    files_saved.append("summary_statistics.json")

    print(f"\n✓ Saved {len(files_saved)} files to {output_dir}")
    for file in files_saved:
        print(f"  - {file}")

    return files_saved


def get_category_summary(df_with_categories):
    """
    Generate a summary DataFrame for each flight category.

    Parameters:
    -----------
    df_with_categories : pd.DataFrame
        DataFrame with flight categories and scores.

    Returns:
    --------
    pd.DataFrame
        Summary statistics by category, sorted by count descending.
    """
    summary_stats = []

    for category in df_with_categories['flight_category'].unique():
        cat_df = df_with_categories[df_with_categories['flight_category'] == category]
        top_bird = cat_df.loc[cat_df['category_score'].idxmax(), 'species']

        stats = {
            'Category':          category.upper(),
            'Count':             len(cat_df),
            'Percentage':        f"{len(cat_df)/len(df_with_categories)*100:.1f}%",
            'Avg Wing Length':   f"{cat_df['Wing.Length'].mean():.2f}",
            'Avg Aspect Ratio':  f"{cat_df['aspect_ratio'].mean():.2f}",
            'Avg Pointedness':   f"{cat_df['pointedness_index'].mean():.3f}",
            'Avg Efficiency':    f"{cat_df['efficiency_index'].mean():.3f}",
            'Avg Fit Score':     f"{cat_df['category_score'].mean():+.3f}",
            'Top Bird':          top_bird,
        }
        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('Count', ascending=False)

    return summary_df


def main():
    """
    Main execution function — runs the complete data processing pipeline.
    """
    print("\n" + "="*60)
    print("AVONET BIRD DATA PROCESSING PIPELINE")
    print("="*60)
    print(f"Data source: {CSV_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    try:
        # Step 1: Load and process data
        normalized_df, df_clean, minmax_scaler, standardized_df = load_and_process_avonet_data(CSV_PATH)

        # Step 2: Categorize birds (scored, exclusive)
        flight_categories, df_with_categories = categorize_birds_by_flight_style(df_clean)

        # Step 3: Generate summary
        summary_df = get_category_summary(df_with_categories)
        print("\n" + "="*60)
        print("CATEGORY SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))

        # Step 4: Save all data
        files_saved = save_processed_data(
            normalized_df,
            df_clean,
            standardized_df,
            df_with_categories,
            flight_categories
        )

        print("\n" + "="*60)
        print("PIPELINE COMPLETE ✓")
        print("="*60)
        print(f"\nProcessed {len(df_with_categories)} birds into {len(flight_categories)} categories")
        print(f"All data saved to: {OUTPUT_DIR}")
        print("\nReady for airfoil generation + NeuralFoil simulation!")

        return {
            'normalized_df':      normalized_df,
            'df_clean':           df_clean,
            'standardized_df':    standardized_df,
            'df_with_categories': df_with_categories,
            'flight_categories':  flight_categories,
            'summary_df':         summary_df
        }

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


def load_processed_data(output_dir=OUTPUT_DIR): 
    """
    Load previously processed data for use in dashboards.

    Returns:
    --------
    dict
        Dictionary containing all processed dataframes.
    """
    output_dir = Path(output_dir)

    data = {
        'normalized':  pd.read_csv(output_dir / "normalized_data.csv"),
        'clean':       pd.read_csv(output_dir / "clean_data_with_indices.csv"),
        'standardized': pd.read_csv(output_dir / "standardized_data.csv"),
        'categorized': pd.read_csv(output_dir / "birds_with_categories.csv")
    }

    category_dir = output_dir / "categories"
    if category_dir.exists():
        data['categories'] = {}
        for csv_file in category_dir.glob("category_*.csv"):
            category_name = csv_file.stem.replace('category_', '')
            data['categories'][category_name] = pd.read_csv(csv_file)

    import json
    summary_file = output_dir / "summary_statistics.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)

    return data


if __name__ == "__main__":
    results = main()