# AVONET Bird Categorisation System

## 📁 Project Structure

```
final itration/
├── DATA/
│   └── AVONET_BIRDLIFE.csv          # Original dataset
├── OUTPUT/                           # Generated processed data
│   ├── normalized_data.csv           # MinMax scaled (0-1) for ML
│   ├── clean_data_with_indices.csv   # Clean data with morphological indices
│   ├── standardized_data.csv         # Z-score standardized data
│   ├── birds_with_categories.csv     # ⭐ MAIN FILE - All birds with categories
│   ├── summary_statistics.json       # Quick stats summary
│   └── categories/                   # Individual category files
│       ├── category_soaring.csv
│       ├── category_diving.csv
│       ├── category_maneuvering.csv
│       ├── category_cruising.csv
│       ├── category_hovering.csv
│       └── category_generalist.csv
├── categorisation.py                 # Main processing script
└── dashboard_example.py              # Streamlit dashboard example
```

## 🚀 Quick Start

### 1. Process the Data

Run the categorisation script to process AVONET data:

```bash
python categorisation.py
```

This will:
- Load 11,009 bird species from AVONET dataset
- Clean and remove outliers
- Calculate morphological indices (aspect ratio, pointedness, efficiency, etc.)
- Categorize birds into 6 flight performance groups
- Save processed data to OUTPUT/ directory

### 2. Launch the Dashboard

```bash
streamlit run dashboard_example.py
```

## 📊 Flight Categories

The system categorizes birds into 6 groups based on morphological features:

| Category | Description | Criteria |
|----------|-------------|----------|
| **Soaring** | Thermal soaring specialists | High aspect ratio + low wing loading |
| **Diving** | High-speed diving birds | High pointedness + moderate-high aspect ratio |
| **Maneuvering** | Tight-turn specialists | Low hand-wing index + broad wings |
| **Cruising** | Long-distance flyers | Balanced efficiency + moderate aspect ratio |
| **Hovering** | Hummingbird-like | Small size + high hand-wing index |
| **Generalist** | Balanced performers | Everything else |

## 📈 Results Summary

From your data (10,975 birds after cleaning):

- **Maneuvering**: 8,695 birds (79.2%)
- **Generalist**: 1,291 birds (11.8%)
- **Cruising**: 828 birds (7.5%)
- **Hovering**: 390 birds (3.6%)
- **Diving**: 987 birds (9.0%)
- **Soaring**: 15 birds (0.1%)

## 🔧 Using the Processed Data

### In Python

```python
from categorisation import load_processed_data

# Load all processed data
data = load_processed_data()

# Access different datasets
df_categorized = data['categorized']      # Main dataset with categories
df_normalized = data['normalized']        # For ML/airfoil generation
df_clean = data['clean']                  # Raw measurements + indices
df_standardized = data['standardized']    # For statistical analysis

# Access individual categories
soaring_birds = data['categories']['soaring']
diving_birds = data['categories']['diving']
```

### In Streamlit

```python
import streamlit as st
import pandas as pd
from categorisation import load_processed_data

@st.cache_data
def load_data():
    return load_processed_data()

data = load_data()
df = data['categorized']

# Now use df in your dashboard
st.dataframe(df)
```

### Direct CSV Access

```python
import pandas as pd

# Load the main categorized dataset
df = pd.read_csv('OUTPUT/birds_with_categories.csv')

# Filter by category
soaring = df[df['flight_category'] == 'soaring']
```

## 📋 Available Columns

### Basic Measurements
- `species`: Bird species name
- `Wing.Length`: Wing length (mm)
- `Secondary1`: Secondary feather length (mm)
- `Kipps.Distance`: Distance from wing tip to secondary (mm)
- `Tail.Length`: Tail length (mm)
- `Hand-Wing.Index`: Hand-wing ratio

### Calculated Indices
- `aspect_ratio`: Wing length / Secondary1
- `wing_loading_proxy`: Wing length / (Secondary1 × Tail length)
- `pointedness_index`: Kipps distance / Wing length
- `efficiency_index`: (Hand-wing index × Pointedness) / 100

### Category
- `flight_category`: Assigned flight performance group

## 🎯 Common Use Cases

### 1. Find birds in a specific category

```python
import pandas as pd

df = pd.read_csv('OUTPUT/birds_with_categories.csv')
diving_specialists = df[df['flight_category'] == 'diving']
print(f"Found {len(diving_specialists)} diving specialists")
```

### 2. Compare morphology across categories

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('OUTPUT/birds_with_categories.csv')

# Average wing length by category
avg_wing = df.groupby('flight_category')['Wing.Length'].mean()
avg_wing.plot(kind='bar')
plt.ylabel('Average Wing Length (mm)')
plt.title('Wing Length by Flight Category')
plt.show()
```

### 3. Select birds for airfoil generation

```python
import pandas as pd

df_normalized = pd.read_csv('OUTPUT/normalized_data.csv')

# Get high-efficiency birds
efficient_birds = df_normalized[df_normalized['efficiency_index'] > 0.5]

# Use normalized features for airfoil parameters
features = efficient_birds[['Wing.Length', 'Secondary1', 'Kipps.Distance']]
```

## 🔄 Reprocessing Data

To reprocess with different parameters, edit `categorisation.py` and modify:

1. **Outlier thresholds** (line ~100):
   ```python
   Q1 = df_clean[col].quantile(0.05)  # Change 0.05
   Q3 = df_clean[col].quantile(0.95)  # Change 0.95
   ```

2. **Category criteria** (lines ~200-240):
   ```python
   soaring_mask = (df['aspect_ratio'] > 2.5) & ...  # Adjust thresholds
   ```

Then run `python categorisation.py` again.

## 📦 Dependencies

```bash
pip install pandas numpy scikit-learn streamlit plotly
```

## 🎨 Dashboard Features

The example dashboard includes:
- ✅ Interactive category filtering
- ✅ Distribution visualizations (pie, bar, scatter, box plots)
- ✅ Morphological analysis across categories
- ✅ Species search and detailed view
- ✅ CSV export of filtered data
- ✅ Real-time statistics

## 🆘 Troubleshooting

**Error: "Could not load CSV file"**
- Check that `DATA/AVONET_BIRDLIFE.csv` exists
- Verify file path in `categorisation.py` (line ~15)

**Error: "Missing columns"**
- AVONET dataset structure may vary
- Check column names match expected format

**Dashboard shows no data**
- Run `python categorisation.py` first to generate OUTPUT files
- Check that OUTPUT directory contains CSV files

## 📝 Notes

- Processing removes ~0.3% outliers (34 of 11,009 birds)
- Categories are mutually exclusive (each bird in one category)
- Normalized data is scaled 0-1 for machine learning
- Standardized data uses z-scores for statistical analysis

## 📧 Integration Tips

To integrate into your existing Streamlit app:

```python
# At the top of your main app
from categorisation import load_processed_data

# In your main function
@st.cache_data
def load_bird_data():
    return load_processed_data()

data = load_bird_data()
df = data['categorized']

# Now use df throughout your app
```

---

**Generated**: March 8, 2026  
**Dataset**: AVONET Bird Functional Trait Database  
**Total Species Processed**: 10,975
