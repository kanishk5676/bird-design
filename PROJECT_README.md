# 🦅 Bird-Inspired Airfoil Generation System

> Generate aerodynamically valid airfoils from 10,975 bird species using morphological data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This system processes AVONET bird morphological data to:
1. **Categorize** 10,975 bird species into 6 flight performance groups
2. **Generate** bio-inspired airfoils using Bézier curve interpolation
3. **Visualize** data through interactive Streamlit dashboards
4. **Export** CFD-ready airfoil coordinates

---

## 📁 Project Structure

```
final itration/
├── 🔧 core_modules/              # Core Python modules
│   ├── categorisation.py         # Bird data processing
│   ├── airfoil_generation.py     # Airfoil generation
│   ├── explore_data.py           # CLI explorer
│   └── README.md                 # Module documentation
│
├── 🎨 streamlit_apps/            # Interactive dashboards
│   ├── dashboard_example.py      # Main bird analysis
│   ├── airfoil_visualizer.py     # Airfoil generator
│   ├── streamlit_examples.py     # Code examples
│   └── README.md                 # Dashboard guide
│
├── 📚 docs/                      # Documentation
│   ├── README.md                 # Main documentation
│   ├── AIRFOIL_README.md         # Airfoil guide
│   ├── SETUP_COMPLETE.md         # Quick setup
│   ├── COMPLETE_SYSTEM_SUMMARY.md
│   └── SYSTEM_READY.txt          # Quick reference
│
├── 💾 DATA/                      # Input data
│   └── AVONET_BIRDLIFE.csv       # Original dataset
│
├── 📊 OUTPUT/                    # Generated data
│   ├── birds_with_categories.csv # Categorized birds
│   ├── categories/               # Category-specific CSVs
│   ├── airfoils/                 # CFD-ready .dat files
│   └── control_points/           # Bézier control points
│
└── 🚀 launch_dashboards.py       # Start both dashboards
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn scipy streamlit plotly
```

### 2. Process Bird Data
```bash
python core_modules/categorisation.py
```

### 3. Launch Dashboards
```bash
python launch_dashboards.py
```

**Access:**
- Main Dashboard: http://localhost:8501
- Airfoil Visualizer: http://localhost:8502

---

## 🎨 Interactive Dashboards

### Main Bird Analysis Dashboard
![Dashboard](https://img.shields.io/badge/Port-8501-blue)

- Category distribution charts
- Morphological scatter plots
- Species search & filtering
- Statistical comparisons
- Data export

### Airfoil Visualizer
![Visualizer](https://img.shields.io/badge/Port-8502-red)

- Real-time airfoil generation
- Interactive plots with control points
- 10,975 bird species available
- Quality validation
- Download coordinates & control points

---

## 📊 Data Summary

| Component | Count | Description |
|-----------|-------|-------------|
| **Total Birds** | 10,975 | Categorized species |
| **Categories** | 6 | Flight performance groups |
| **Airfoils** | 6,715 | Generated (61% complete) |
| **Features** | 10+ | Morphological parameters |

### Flight Categories

| Category | Count | % | Characteristics |
|----------|-------|---|-----------------|
| **Maneuvering** | 8,695 | 79.2% | Small, agile, short wings |
| **Generalist** | 1,291 | 11.8% | Balanced all-around |
| **Diving** | 987 | 9.0% | High-speed, pointed wings |
| **Cruising** | 828 | 7.5% | Long-distance efficient |
| **Hovering** | 390 | 3.6% | Tiny, hummingbird-like |
| **Soaring** | 15 | 0.1% | Large thermal soarers |

---

## 🔧 Core Modules

### categorisation.py
Process and categorize bird data
```python
from core_modules.categorisation import load_processed_data

data = load_processed_data()
df = data['categorized']
```

### airfoil_generation.py
Generate airfoils from bird morphology
```python
from core_modules.airfoil_generation import generate_and_validate_airfoil

upper_cp, lower_cp, metadata, is_valid = generate_and_validate_airfoil(bird_data)
```

### explore_data.py
Command-line data explorer
```bash
python core_modules/explore_data.py search "swift"
python core_modules/explore_data.py compare diving hovering
```

---

## 📥 Output Files

### Bird Categories
- `OUTPUT/birds_with_categories.csv` - Main dataset
- `OUTPUT/normalized_data.csv` - 0-1 scaled for ML
- `OUTPUT/categories/*.csv` - Individual categories

### Airfoils
- `OUTPUT/airfoils/*.dat` - CFD-ready coordinates
- `OUTPUT/control_points/*.csv` - Bézier parameters
- Organized by flight category

---

## 🎯 Use Cases

### Research & Analysis
```python
import pandas as pd

df = pd.read_csv('OUTPUT/birds_with_categories.csv')
diving_birds = df[df['flight_category'] == 'diving']
print(f"Found {len(diving_birds)} diving specialists")
```

### CFD Analysis
```python
# Export airfoil for specific bird
from core_modules.airfoil_generation import generate_and_validate_airfoil, bezier_curve
import numpy as np

upper_cp, lower_cp, meta, valid = generate_and_validate_airfoil(bird)
upper = bezier_curve(upper_cp, 200)
lower = bezier_curve(lower_cp, 200)
coords = np.vstack([upper[::-1], lower[1:]])
np.savetxt('my_airfoil.dat', coords, fmt='%.8f')
```

### Visualization
Use Streamlit dashboards for interactive exploration

---

## 🏆 Top Performers

1. **Cypsiurus parvus** (African Palm Swift) - Efficiency: 0.552
2. **Streptoprocne phelpsi** (Tepui Swift) - Efficiency: 0.544
3. **Apus apus** (Common Swift) - Efficiency: 0.534

---

## 📖 Documentation

Comprehensive documentation in `docs/`:

- **Quick Setup:** `docs/SETUP_COMPLETE.md`
- **Airfoil Guide:** `docs/AIRFOIL_README.md`
- **Full Reference:** `docs/COMPLETE_SYSTEM_SUMMARY.md`
- **Quick Commands:** `docs/SYSTEM_READY.txt`

---

## 🔄 Workflow Examples

### Generate More Airfoils
```bash
python core_modules/airfoil_generation.py
```

### Reprocess Bird Data
```bash
python core_modules/categorisation.py
```

### Launch Single Dashboard
```bash
streamlit run streamlit_apps/dashboard_example.py
```

---

## 🛠️ Customization

### Modify Category Criteria
Edit `core_modules/categorisation.py`:
```python
# Function: categorize_birds_by_flight_style()
soaring_mask = (df['aspect_ratio'] > 2.8) & ...  # Adjust thresholds
```

### Adjust Airfoil Parameters
Edit `core_modules/airfoil_generation.py`:
```python
# Function: biologically_enhanced_airfoil_generation()
max_thickness = np.clip(max_thickness, 0.05, 0.20)  # Modify range
```

---

## 📊 System Status

| Component | Status | Details |
|-----------|--------|---------|
| Bird Processing | ✅ Complete | 10,975 birds categorized |
| Airfoil Generation | 🔄 61% | 6,715 / 10,975 generated |
| Main Dashboard | ✅ Ready | Port 8501 |
| Airfoil Visualizer | ✅ Ready | Port 8502 |
| Documentation | ✅ Complete | 5 guides |

---

## 🆘 Troubleshooting

**Dashboard won't load?**
```bash
# Check if port is in use
lsof -ti:8501 | xargs kill -9
```

**Import errors?**
```bash
# Ensure you're in project directory
cd "/path/to/final itration"
python launch_dashboards.py
```

**Missing data?**
```bash
# Reprocess bird data
python core_modules/categorisation.py
```

---

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
scipy>=1.7.0
streamlit>=1.0.0
plotly>=5.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 📞 Quick Commands

```bash
# View categories
python core_modules/explore_data.py

# Search species
python core_modules/explore_data.py search "eagle"

# Compare categories
python core_modules/explore_data.py compare diving hovering

# Launch dashboards
python launch_dashboards.py

# Generate airfoils
python core_modules/airfoil_generation.py
```

---

## 🎓 Citation

If you use this system in your research, please cite:

```
AVONET Bird Functional Trait Database
Tobias et al. (2022). AVONET: Morphological, ecological and 
geographical data for all birds. Ecology Letters, 25(3), 581-597.
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request

---

## 📧 Contact

For questions or issues, please open an issue or contact the maintainers.

---

**Built with ❤️ using Python, Streamlit, and bird data**

*Last Updated: March 8, 2026*
