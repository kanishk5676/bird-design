# Bird-Inspired Airfoil Generation System

## 🎯 Overview

This system generates aerodynamically valid airfoils from bird morphological data using biologically-grounded Bézier curves.

## 📁 Generated Files

```
OUTPUT/
├── airfoils/                          # Airfoil coordinate files (.dat)
│   ├── cruising/                      # 339 cruising bird airfoils
│   ├── diving/                        # 110 diving bird airfoils
│   ├── generalist/                    # 819 generalist bird airfoils
│   ├── hovering/                      # 388 hovering bird airfoils
│   ├── maneuvering/                   # 5,069 maneuvering bird airfoils
│   ├── complete_airfoil_database.csv  # Summary of all airfoils
│   └── category_statistics.csv        # Statistics by category
│
└── control_points/                    # Bézier control points (.csv)
    ├── cruising/
    ├── diving/
    ├── generalist/
    ├── hovering/
    ├── maneuvering/
    └── control_points_summary.csv     # Summary of all control points
```

## 📊 Current Status

✅ **Generated:** 6,715 airfoils (61% complete)  
📂 **Airfoil files:** `.dat` format (CFD-ready)  
📂 **Control points:** `.csv` format (parametric design)

## 🚀 Quick Start

### 1. View Airfoils in Streamlit

```bash
streamlit run airfoil_visualizer.py
```

**Features:**
- 🔍 Search by species name
- 📂 Browse by flight category
- 🎲 Random selection
- 🏆 View top performers
- 📥 Download airfoil coordinates
- 📊 Real-time generation for any bird
- ✅ Quality validation

### 2. Complete Airfoil Generation

To generate the remaining airfoils:

```bash
python airfoil_generation.py
```

This will resume and generate all 10,975 airfoils.

### 3. Generate for Specific Birds

```python
from airfoil_generation import generate_and_validate_airfoil
import pandas as pd

# Load bird data
df = pd.read_csv('OUTPUT/birds_with_categories.csv')

# Select a bird
bird = df.iloc[0]

# Generate airfoil
upper_cp, lower_cp, metadata, is_valid = generate_and_validate_airfoil(bird)
```

## 📖 File Formats

### Airfoil DAT Files

Standard format for CFD software (XFOIL, ANSYS, etc.):

```
# Bird-inspired airfoil generated from AVONET morphological data
# Airfoil ID: 00001
# Species: Accipiter_albogularis
# Flight Category: maneuvering
# ...metadata...
# X/C    Y/C
0.99999  0.00001
0.99801  0.00089
...
```

**Usage:**
- Import directly into XFOIL
- Use in CFD simulations
- 200 points for smooth resolution
- Closed loop (TE to TE)

### Control Points CSV

Bézier control points for parametric design:

```csv
Surface,Point_Index,X,Y
UPPER,0,0.00000000,0.00000000
UPPER,1,0.01200000,0.07200000
...
LOWER,0,0.00000000,0.00000000
LOWER,1,0.01200000,-0.05200000
...
```

**Usage:**
- Regenerate airfoils with different resolution
- Modify geometry parametrically
- CAD software import

## 🔧 Key Parameters

The airfoil generation uses these morphological features:

| Input Parameter | Source | Effect on Airfoil |
|-----------------|--------|-------------------|
| Wing Length | Direct measurement | Reynolds number estimation |
| Secondary1 | Direct measurement | Thickness ratio |
| Kipps Distance | Direct measurement | Camber |
| Hand-Wing Index | Direct measurement | Pointedness |
| Tail Length | Direct measurement | Trailing edge, reflex |
| Aspect Ratio | Calculated | Overall thickness |
| Pointedness Index | Calculated | Leading edge sharpness |

## 📊 Airfoil Properties

Generated airfoils have these characteristics:

- **Thickness:** 4-15% chord (biologically constrained)
- **Camber:** 1-8% chord
- **Leading Edge Radius:** 0.5-4% chord
- **Reynolds Number:** 10^4 to 10^6 (size-dependent)
- **Validation:** Automatic quality checks

## 🎨 Visualization Examples

The Streamlit visualizer shows:

1. **Airfoil Profile** - Upper/lower surfaces
2. **Control Points** - Bézier curve controls
3. **Properties** - Geometric characteristics
4. **Comparison** - vs category averages
5. **Export** - Download coordinates/control points

## 📈 Use Cases

### 1. CFD Analysis

```bash
# Export airfoil for specific bird
python -c "
from categorisation import load_processed_data
from airfoil_generation import generate_and_validate_airfoil, bezier_curve
import numpy as np

data = load_processed_data()
df = data['categorized']

# Get a high-efficiency bird
bird = df.nlargest(1, 'efficiency_index').iloc[0]

# Generate
upper_cp, lower_cp, metadata, valid = generate_and_validate_airfoil(bird)

# Export for CFD
upper = bezier_curve(upper_cp, 200)
lower = bezier_curve(lower_cp, 200)
coords = np.vstack([upper[::-1], lower[1:]])
np.savetxt('my_airfoil.dat', coords, fmt='%.8f')
print(f'Exported: {bird[\"species\"]}')
"
```

### 2. Compare Flight Styles

```python
import pandas as pd
import matplotlib.pyplot as plt
from airfoil_generation import biologically_enhanced_airfoil_generation, bezier_curve

df = pd.read_csv('OUTPUT/birds_with_categories.csv')

# Get representative birds from each category
categories = ['diving', 'soaring', 'hovering', 'maneuvering']

plt.figure(figsize=(12, 8))
for i, cat in enumerate(categories):
    bird = df[df['flight_category'] == cat].iloc[0]
    
    upper_cp, lower_cp, _ = biologically_enhanced_airfoil_generation(
        bird['Wing.Length'], bird['Secondary1'],
        bird['Kipps.Distance'], bird['Hand-Wing.Index'],
        bird['Tail.Length'], bird['aspect_ratio'],
        bird['pointedness_index']
    )
    
    upper = bezier_curve(upper_cp, 100)
    lower = bezier_curve(lower_cp, 100)
    
    plt.subplot(2, 2, i+1)
    plt.plot(upper[:, 0], upper[:, 1], 'b-', label='Upper')
    plt.plot(lower[:, 0], lower[:, 1], 'r-', label='Lower')
    plt.title(f'{cat.upper()}: {bird["species"]}')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.savefig('category_comparison.png', dpi=300)
```

### 3. Batch Export

```python
from pathlib import Path
import pandas as pd
from airfoil_generation import export_all_airfoil_data

# Load data
df = pd.read_csv('OUTPUT/birds_with_categories.csv')

# Export specific category
diving_birds = df[df['flight_category'] == 'diving']
summary = export_all_airfoil_data(diving_birds)

print(f"Exported {len(summary)} diving bird airfoils")
```

## 🔍 Quality Validation

Each airfoil is automatically validated for:

- ✅ No self-intersections
- ✅ Smooth curvature (no kinks)
- ✅ Reasonable thickness (4-20%)
- ✅ Appropriate leading edge radius

Invalid airfoils are flagged in the output.

## 📚 Database Summary

### By Flight Category

| Category | Count | Avg Thickness | Avg Camber | Characteristics |
|----------|-------|---------------|------------|-----------------|
| Maneuvering | 5,069 | ~12% | ~3% | Thick, moderate camber |
| Generalist | 819 | ~10% | ~4% | Balanced |
| Diving | 110 | ~6% | ~6% | Thin, high camber |
| Cruising | 339 | ~8% | ~5% | Moderate |
| Hovering | 388 | ~7% | ~7% | Thin, very high camber |

## 🆘 Troubleshooting

**Q: Airfoil looks wrong?**
- Check validation messages
- Verify bird data is complete
- Some extreme morphologies may produce unusual shapes

**Q: Want to modify generation?**
- Edit `airfoil_generation.py`
- Adjust parameters in `biologically_enhanced_airfoil_generation()`
- Re-run generation

**Q: Need different resolution?**
- Modify `num_points` in `bezier_curve()` function
- Default is 200 points (good for CFD)

**Q: How to use in XFOIL?**
```bash
xfoil
LOAD my_airfoil.dat
OPER
```

## 📊 Statistics

**Total Birds:** 10,975  
**Generated Airfoils:** 6,715 (61%)  
**File Size:** ~100 MB (airfoils) + ~80 MB (control points)  
**Categories:** 5 flight styles  
**Resolution:** 200 points per airfoil  
**Control Points:** 12 per surface  

## 🔄 Next Steps

1. ✅ **Complete generation** - Run `python airfoil_generation.py`
2. ✅ **Explore visually** - Run `streamlit run airfoil_visualizer.py`
3. ⚙️ **CFD analysis** - Import `.dat` files into your CFD software
4. 📊 **Statistical analysis** - Analyze `complete_airfoil_database.csv`
5. 🎨 **Custom design** - Modify control points for hybrid designs

## 💡 Tips

- Use **normalized data** for ML applications
- Use **control points** for parametric design
- Use **DAT files** for CFD simulation
- **Streamlit visualizer** generates on-demand (no pre-generation needed)

---

**Generated:** March 8, 2026  
**Database:** AVONET  
**Algorithm:** Bio-inspired Bézier interpolation  
**Validation:** Automated quality checks
