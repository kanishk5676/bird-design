# 🔧 Core Modules

This folder contains the essential Python modules for bird data processing and airfoil generation.

## 📁 Files

### 1. categorisation.py
**Main bird data processing and categorization module**

**Functions:**
- `load_and_process_avonet_data()` - Load and clean bird data
- `calculate_morphological_indices()` - Calculate derived parameters
- `categorize_birds_by_flight_style()` - Assign flight categories
- `save_processed_data()` - Export processed data
- `load_processed_data()` - Load previously processed data

**Usage:**
```python
from categorisation import load_processed_data

data = load_processed_data()
df = data['categorized']  # Main dataset with categories
```

**Run as script:**
```bash
python categorisation.py
```
This processes all bird data and saves to `../OUTPUT/`

---

### 2. airfoil_generation.py
**Airfoil generation from bird morphology**

**Functions:**
- `biologically_enhanced_airfoil_generation()` - Generate airfoil from bird data
- `bezier_curve()` - Create smooth Bézier curves
- `validate_airfoil_quality()` - Check aerodynamic validity
- `generate_and_validate_airfoil()` - Complete generation pipeline
- `export_all_airfoil_data()` - Batch export all airfoils

**Usage:**
```python
from airfoil_generation import generate_and_validate_airfoil

upper_cp, lower_cp, metadata, is_valid = generate_and_validate_airfoil(bird_data)
```

**Run as script:**
```bash
python airfoil_generation.py
```
This generates airfoils for all birds and saves to `../OUTPUT/airfoils/`

---

### 3. explore_data.py
**Command-line data exploration tool**

**Functions:**
- `explore_categories()` - Display all categories
- `search_species()` - Search for specific birds
- `compare_categories()` - Compare two flight categories

**Usage:**
```bash
# View all categories
python explore_data.py

# Search for species
python explore_data.py search "swift"

# Compare categories
python explore_data.py compare diving hovering
```

---

## 🔄 Data Flow

```
AVONET_BIRDLIFE.csv
        ↓
categorisation.py
        ↓
birds_with_categories.csv
        ↓
airfoil_generation.py
        ↓
airfoils/*.dat + control_points/*.csv
```

---

## 📊 Output Structure

All modules save to `../OUTPUT/`:

```
OUTPUT/
├── birds_with_categories.csv       (categorisation.py)
├── normalized_data.csv              (categorisation.py)
├── clean_data_with_indices.csv      (categorisation.py)
├── standardized_data.csv            (categorisation.py)
├── summary_statistics.json          (categorisation.py)
├── categories/                      (categorisation.py)
├── airfoils/                        (airfoil_generation.py)
└── control_points/                  (airfoil_generation.py)
```

---

## 🚀 Quick Start

### First Time Setup
```bash
# 1. Process bird data
python categorisation.py

# 2. Generate airfoils (optional - takes time)
python airfoil_generation.py

# 3. Explore data
python explore_data.py
```

### Regular Use
```python
# Import in your scripts
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "core_modules"))

from categorisation import load_processed_data
from airfoil_generation import generate_and_validate_airfoil

# Load data
data = load_processed_data()
df = data['categorized']

# Generate airfoil for a bird
bird = df.iloc[0]
upper_cp, lower_cp, metadata, valid = generate_and_validate_airfoil(bird)
```

---

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn scipy
```

**Required:**
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Data scaling
- scipy - Bézier curves

---

## 🔧 Configuration

All modules use these paths:
- `DATA_DIR` - `../DATA/` (input data)
- `OUTPUT_DIR` - `../OUTPUT/` (generated files)

Paths are automatically configured relative to module location.

---

## ⚙️ Advanced Usage

### Custom Category Criteria

Edit `categorisation.py`, function `categorize_birds_by_flight_style()`:

```python
# Modify thresholds
soaring_mask = (df['aspect_ratio'] > 2.8) & ...  # Change from 2.5
diving_mask = (df['pointedness_index'] > 0.5) & ...  # Change from 0.4
```

### Custom Airfoil Parameters

Edit `airfoil_generation.py`, function `biologically_enhanced_airfoil_generation()`:

```python
# Modify thickness range
max_thickness = np.clip(max_thickness, 0.05, 0.20)  # Change limits

# Modify camber
max_camber = base_camber * 1.2  # Increase camber
```

---

## 🧪 Testing

```bash
# Test categorisation
python -c "from categorisation import main; main()"

# Test airfoil generation for one bird
python -c "
from categorisation import load_processed_data
from airfoil_generation import generate_and_validate_airfoil
data = load_processed_data()
bird = data['categorized'].iloc[0]
upper, lower, meta, valid = generate_and_validate_airfoil(bird)
print(f'Valid: {valid}, Thickness: {meta[\"max_thickness\"]:.3f}')
"
```

---

## 📚 Documentation

See `../docs/` for complete documentation:
- README.md - Project overview
- AIRFOIL_README.md - Airfoil system guide
- COMPLETE_SYSTEM_SUMMARY.md - Full system reference
