# NeuralFoil Aerodynamic Analysis for Bird-Inspired Airfoils

## Overview

This module performs comprehensive aerodynamic analysis on all 10,975 bird-inspired airfoils using NeuralFoil, a physics-informed machine learning tool that provides XFOIL-quality predictions at high speed.

## What's Been Created

1. **neuralfoil_analysis.py** - Main analysis script
2. **test_neuralfoil_setup.py** - Testing script to verify setup

## Analysis Configuration

### Test Matrix
- **Reynolds Numbers**: 1×10⁴, 5×10⁴, 1×10⁵, 2×10⁵, 5×10⁵, 1×10⁶
- **Angles of Attack**: -4° to 20° in 2° steps (13 values)
- **Total Conditions**: 78 per airfoil
- **Total Simulations**: 856,050 (10,975 airfoils × 78 conditions)

### Why These Parameters?

**Reynolds Number Range**:
- 1×10⁴: Hummingbirds, small insects
- 5×10⁴: Small passerines  
- 1×10⁵: Medium birds (pigeons)
- 2×10⁵: Large birds (herons, eagles)
- 5×10⁵: Very large birds (storks, pelicans)
- 1×10⁶: Albatross-scale

**Angle of Attack Range**:
- **-4° to 0°**: Descent, negative AoA performance
- **0° to 2°**: Cruise regime
- **4° to 8°**: Design/climb regime
- **8° to 12°**: High climb/maneuver
- **14° to 20°**: Post-stall characterization

## Key Features

### 1. Automatic Chord Normalization
Your airfoils are generated with chord = 2.0. The script automatically normalizes all coordinates to chord = 1.0 before analysis, ensuring:
- Consistent reference areas across all birds
- Valid L/D comparisons
- Correct coefficient scaling

### 2. Reynolds Number Extraction
The script extracts estimated Reynolds numbers from your .dat file headers based on:
- Wing morphology (wingspan, chord)
- Estimated flight speed
- Falls back to morphology-based estimation if not in header

### 3. Comprehensive Metrics

**Per-Bird Metrics**:
- Maximum L/D (overall best)
- Cruise L/D (α = 0-2°)
- Climb L/D (α = 8-12°)
- High-alpha L/D (α ≥ 14°)
- Negative-alpha performance
- CLmax and stall angle
- Minimum CD
- Design-point performance at estimated Re
- L/D coefficient of variation (consistency metric)

**Category Analysis**:
- Top 3 performers by overall L/D
- Top 3 by cruise efficiency
- Top 3 by CLmax/stall performance
- Top 3 by consistency
- Category statistics
- Natural-language performance explanations

### 4. Stall Detection
Automatically detects:
- CLmax (maximum lift coefficient)
- Stall angle (first 5% drop from CLmax)
- Post-stall behavior

### 5. Output Files

**CSV Files**:
- `detailed_analysis.csv` - All 856,050 data points (bird, α, Re, CL, CD, CM, L/D, confidence)
- `summary_analysis.csv` - Per-bird summary statistics

**Pickle Files**:
- `detailed_results.pkl` - Python-readable detailed results
- `summary_results.pkl` - Python-readable summaries
- `category_leaders.pkl` - Category champion data with explanations

**Report**:
- `comprehensive_report.txt` - Human-readable analysis with:
  - Overall statistics
  - Category-by-category breakdown
  - Champion airfoils with performance reasons
  - Paper-ready summary table

**Visualizations** (7 PNG files @ 300 DPI):
1. `ld_distribution_by_category.png` - Box plots of L/D by category
2. `clmax_by_category.png` - CLmax distributions
3. `stall_angle_by_category.png` - Stall angle distributions
4. `top_performers.png` - Top 15 airfoils bar chart
5. `ld_vs_reynolds.png` - L/D trends across Re
6. `correlation_matrix.png` - Performance metric correlations
7. `cl_alpha_polars.png` - CL-α curves for category leaders

## Usage

### Quick Test (30 seconds)
```bash
python test_neuralfoil_setup.py
```
This verifies:
- All .dat files can be loaded
- NeuralFoil is working
- Category distribution
- One sample simulation

### Full Analysis

**Option 1: Foreground** (not recommended - takes ~1200 hours)
```bash
python neuralfoil_analysis.py
```

**Option 2: Background with logging** (recommended)
```bash
nohup python neuralfoil_analysis.py > neuralfoil_analysis.log 2>&1 &
```

Monitor progress:
```bash
tail -f neuralfoil_analysis.log
```

**Option 3: Subset Test** (recommended first)
Modify the script to test on a subset:
```python
# In neuralfoil_analysis.py, line ~372
dat_files = sorted(directory_path.rglob("*.dat"))[:100]  # Only first 100 birds
```

### Check Running Process
```bash
ps aux | grep neuralfoil_analysis
```

### Stop Running Analysis
```bash
kill -9 $(ps aux | grep neuralfoil_analysis | awk '{print $2}')
```

## Performance Optimization

### Computational Cost
- **Per simulation**: ~5 seconds on average
- **Total time estimate**: ~1,189 hours (~50 days) on single core
- **Memory usage**: ~2-4 GB RAM

### Speed-Up Options

1. **Reduce Model Size**: Change `MODEL_SIZE = "large"` instead of "xxlarge"
   - Trade-off: ~3x faster, slightly less accurate

2. **Parallel Processing**: Modify script to use multiprocessing
   ```python
   from multiprocessing import Pool
   ```

3. **Reduce Test Matrix**:
   - Fewer Reynolds numbers (e.g., only 1e4, 1e5, 1e6)
   - Smaller alpha range (e.g., 0° to 16°)
   - Skip negative alphas if not needed

4. **Category Sampling**: Run analysis on stratified sample:
   - Top 100 airfoils per category by morphological efficiency
   - Representative sample across aspect ratios

## File Organization

```
final itration/
├── neuralfoil_analysis.py          ← Main analysis script
├── test_neuralfoil_setup.py        ← Setup verification
├── NEURALFOIL_README.md            ← This file
│
├── OUTPUT/
│   └── airfoils/                   ← Input: 10,975 .dat files
│       ├── hovering/
│       ├── diving/
│       ├── maneuvering/
│       ├── soaring/
│       └── generalist/
│
└── RESULTS/
    └── neuralfoil_analysis/        ← Output directory
        ├── detailed_analysis.csv
        ├── summary_analysis.csv
        ├── comprehensive_report.txt
        ├── category_leaders.pkl
        └── *.png (7 plots)
```

## Code Adaptations Made

### Original Script Issues Fixed:

1. **✅ Path Configuration**
   - Changed from hardcoded `./OUTPUT/airfoils` to `Path(__file__).parent / "OUTPUT" / "airfoils"`
   - Works from any directory

2. **✅ Reynolds Number Handling**
   - Extracts from header if available
   - Falls back to morphology-based estimation
   - Handles multiple header formats

3. **✅ Category Structure**
   - Works with your subdirectory organization (hovering/, diving/, etc.)
   - Automatically extracts category from directory name

4. **✅ Numpy Compatibility**
   - Fixed `safe_float()` to handle numpy 0-d arrays properly
   - Uses `.item()` method for numpy scalars
   - Avoids deprecation warnings

5. **✅ Coordinate Normalization**
   - Handles chord=2.0 airfoils correctly
   - Normalizes to chord=1.0 before NeuralFoil
   - Guarantees consistent reference areas

6. **✅ Species Name Extraction**
   - Parses from filename if not in header
   - Handles underscore-separated names
   - Robust to different header formats

## Interpreting Results

### L/D Ratios
- **Excellent**: L/D > 30
- **Very Good**: L/D 20-30
- **Good**: L/D 15-20
- **Moderate**: L/D 10-15
- **Poor**: L/D < 10

Bird airfoils typically range 5-25 due to:
- Low Reynolds numbers
- Thickness constraints
- Multi-point design requirements

### CLmax Values
- **High**: CLmax > 1.5 (aggressive camber, high-lift)
- **Moderate**: CLmax 1.0-1.5 (balanced)
- **Low**: CLmax < 1.0 (low camber, speed-oriented)

### Stall Angles
- **Gentle**: Stall > 16° (broad usable range)
- **Typical**: Stall 12-16°
- **Sharp**: Stall < 12° (narrow operating envelope)

### Analysis Confidence
NeuralFoil provides confidence scores:
- **High**: > 0.7 (trustworthy)
- **Medium**: 0.5-0.7 (use with caution)
- **Low**: < 0.5 (flagged, may be inaccurate)

The script flags low-confidence results but keeps them for completeness.

## Citation & References

If you use this analysis in your research:

**NeuralFoil**:
```
Sharpe, P. (2023). NeuralFoil: Physics-informed machine learning for 
aerodynamic analysis. GitHub repository: 
https://github.com/peterdsharpe/NeuralFoil
```

**XFOIL** (NeuralFoil is trained on XFOIL):
```
Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds 
number airfoils. In Low Reynolds Number Aerodynamics (pp. 1-12). Springer.
```

## Troubleshooting

### "No .dat files found"
- Check `AIRFOIL_DIR` path in script
- Verify files are in `OUTPUT/airfoils/*/` subdirectories

### "NeuralFoil error"
- Install: `pip install neuralfoil`
- Update: `pip install --upgrade neuralfoil`

### "Memory error"
- Reduce model size to "large" or "medium"
- Process in batches (add file slicing)

### "Analysis taking too long"
- Expected! 856,050 simulations takes ~50 days
- Consider running subset first
- Use background processing with nohup

### "Unusual L/D values"
- Check coordinate normalization (script handles automatically)
- Verify Reynolds number extraction
- Check analysis_confidence values

## Next Steps

1. **✅ Test Setup**: Run `python test_neuralfoil_setup.py`
2. **Subset Test**: Modify script to run 100 birds first
3. **Full Analysis**: Run in background with nohup
4. **Results Analysis**: Use summary_analysis.csv for statistics
5. **Paper Figures**: Use generated PNG plots (300 DPI, publication-ready)

## Questions?

Check the script docstrings for detailed explanations of each function.
All algorithms are documented with rationale.
