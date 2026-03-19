# 🎨 Streamlit Applications

This folder contains interactive Streamlit dashboards for bird data analysis and airfoil visualization.

## 📊 Applications

### 1. Main Bird Analysis Dashboard
**File:** `dashboard_example.py`  
**Port:** 8501

**Features:**
- Category distribution analysis (pie, bar charts)
- Morphological scatter plots
- Species search and filtering
- Statistical comparisons
- Data export (CSV)
- Box plots by flight category

**Launch:**
```bash
streamlit run dashboard_example.py
# or from project root:
streamlit run streamlit_apps/dashboard_example.py
```

---

### 2. Airfoil Visualizer
**File:** `airfoil_visualizer.py`  
**Port:** 8502

**Features:**
- Real-time airfoil generation for any bird
- Interactive airfoil plots
- Control point visualization
- Species search (10,975 birds)
- Browse by flight category
- Random selection & top performers
- Download coordinates & control points
- Quality validation
- Geometric properties display

**Launch:**
```bash
streamlit run airfoil_visualizer.py --server.port 8502
# or from project root:
streamlit run streamlit_apps/airfoil_visualizer.py --server.port 8502
```

---

### 3. Streamlit Code Examples
**File:** `streamlit_examples.py`

Reference file with code patterns and examples for integrating bird data into Streamlit applications.

---

## 🚀 Quick Start

### Launch Single Dashboard
```bash
cd streamlit_apps
streamlit run dashboard_example.py
```

### Launch Both Dashboards
From project root:
```bash
python launch_dashboards.py
```

This will start:
- Main Dashboard at http://localhost:8501
- Airfoil Visualizer at http://localhost:8502

---

## 📦 Dependencies

All dashboards require:
- streamlit
- pandas
- numpy
- plotly
- pathlib (built-in)

Install:
```bash
pip install streamlit pandas numpy plotly
```

---

## 🔧 Configuration

Both dashboards automatically:
- Import from `../core_modules/` directory
- Load data from `../OUTPUT/` directory
- Work with categorized bird data
- Support real-time airfoil generation

---

## 📖 Usage Tips

1. **First Time:** Run `python ../core_modules/categorisation.py` to generate bird categories
2. **Airfoils:** Airfoil visualizer generates on-demand (no pre-generation needed)
3. **Export:** Use download buttons in dashboards for CSV export
4. **Search:** Use species search for quick access to specific birds
5. **Categories:** Filter by flight category to analyze specific groups

---

## 🎯 Common Tasks

**View all categories:**
- Open Main Dashboard → Category distribution section

**Generate airfoil for specific bird:**
- Open Airfoil Visualizer → Search → Enter species name

**Compare flight styles:**
- Open Airfoil Visualizer → Browse by Category → Select category

**Export data:**
- Both dashboards have download buttons for filtered data

---

## 🔗 Related Files

- Core modules: `../core_modules/`
- Documentation: `../docs/`
- Data: `../OUTPUT/`
- Launcher: `../launch_dashboards.py`
