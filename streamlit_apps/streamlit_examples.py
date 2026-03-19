"""
QUICK REFERENCE - Using Bird Categories in Your Streamlit Dashboard
=====================================================================
"""

# ============================================================================
# METHOD 1: Simple - Load CSV directly
# ============================================================================

import streamlit as st
import pandas as pd

# Load the main categorized dataset
df = pd.read_csv('OUTPUT/birds_with_categories.csv')

# Show basic info
st.title("Bird Flight Categories")
st.write(f"Total birds: {len(df):,}")

# Filter by category
category = st.selectbox("Select Category", df['flight_category'].unique())
filtered = df[df['flight_category'] == category]

st.dataframe(filtered)


# ============================================================================
# METHOD 2: Using the load function (recommended)
# ============================================================================

from categorisation import load_processed_data

@st.cache_data
def get_bird_data():
    return load_processed_data()

data = get_bird_data()

# Access different datasets
df_main = data['categorized']           # Main dataset with categories
df_normalized = data['normalized']      # For ML/airfoil generation
df_clean = data['clean']                # Raw measurements + indices
df_standardized = data['standardized']  # For statistics

# Access individual categories
soaring_birds = data['categories']['soaring']
diving_birds = data['categories']['diving']


# ============================================================================
# METHOD 3: Direct category files
# ============================================================================

# Load specific category directly
import pandas as pd

diving_birds = pd.read_csv('OUTPUT/categories/category_diving.csv')
hovering_birds = pd.read_csv('OUTPUT/categories/category_hovering.csv')


# ============================================================================
# COMMON STREAMLIT PATTERNS
# ============================================================================

# Pattern 1: Category statistics
st.header("Category Statistics")

stats = df.groupby('flight_category').agg({
    'Wing.Length': 'mean',
    'aspect_ratio': 'mean',
    'efficiency_index': 'mean',
    'species': 'count'
}).round(2)

st.dataframe(stats)


# Pattern 2: Interactive filtering
st.sidebar.header("Filters")

# Multi-select for categories
selected_categories = st.sidebar.multiselect(
    "Flight Categories",
    options=df['flight_category'].unique(),
    default=df['flight_category'].unique()
)

# Slider for wing length
wing_range = st.sidebar.slider(
    "Wing Length (mm)",
    min_value=float(df['Wing.Length'].min()),
    max_value=float(df['Wing.Length'].max()),
    value=(float(df['Wing.Length'].min()), float(df['Wing.Length'].max()))
)

# Apply filters
filtered_df = df[
    (df['flight_category'].isin(selected_categories)) &
    (df['Wing.Length'].between(wing_range[0], wing_range[1]))
]


# Pattern 3: Visualizations with Plotly
import plotly.express as px

# Scatter plot
fig = px.scatter(
    df,
    x='Wing.Length',
    y='aspect_ratio',
    color='flight_category',
    hover_data=['species'],
    title='Aspect Ratio vs Wing Length'
)
st.plotly_chart(fig)

# Box plot
fig = px.box(
    df,
    x='flight_category',
    y='efficiency_index',
    color='flight_category',
    title='Efficiency by Category'
)
st.plotly_chart(fig)


# Pattern 4: Species search
st.header("Search Species")
search = st.text_input("Enter species name")

if search:
    results = df[df['species'].str.contains(search, case=False, na=False)]
    if len(results) > 0:
        st.success(f"Found {len(results)} species")
        st.dataframe(results)
    else:
        st.warning("No species found")


# Pattern 5: Download filtered data
st.header("Export Data")

csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_birds.csv",
    mime="text/csv"
)


# Pattern 6: Metrics display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Birds", f"{len(df):,}")

with col2:
    st.metric("Categories", df['flight_category'].nunique())

with col3:
    avg_wing = df['Wing.Length'].mean()
    st.metric("Avg Wing Length", f"{avg_wing:.1f} mm")

with col4:
    most_common = df['flight_category'].mode()[0]
    st.metric("Most Common", most_common.title())


# ============================================================================
# ACCESSING SPECIFIC DATA
# ============================================================================

# Get all diving birds
diving = df[df['flight_category'] == 'diving']

# Get top 10 most efficient
top_efficient = df.nlargest(10, 'efficiency_index')

# Get birds with high aspect ratio
high_aspect = df[df['aspect_ratio'] > 2.5]

# Get small birds (potential hoverers)
small_birds = df[df['Wing.Length'] < 100]

# Get birds in specific range
medium_wings = df[df['Wing.Length'].between(100, 200)]


# ============================================================================
# NORMALIZED DATA FOR AIRFOIL GENERATION
# ============================================================================

# Use normalized data (0-1 scale) for generating airfoil parameters
df_norm = pd.read_csv('OUTPUT/normalized_data.csv')

# Select a bird
bird_idx = 100
bird = df_norm.iloc[bird_idx]

# Extract normalized features (ready for airfoil generation)
wing_params = {
    'wing_length': bird['Wing.Length'],        # 0-1 normalized
    'secondary': bird['Secondary1'],           # 0-1 normalized
    'kipps': bird['Kipps.Distance'],          # 0-1 normalized
    'tail': bird['Tail.Length'],              # 0-1 normalized
    'hand_wing': bird['Hand-Wing.Index'],     # 0-1 normalized
}

# Map to airfoil parameters
airfoil_params = {
    'camber': wing_params['kipps'] * 0.1,                    # 0-0.1
    'thickness': wing_params['secondary'] * 0.3,             # 0-0.3
    'chord_length': wing_params['wing_length'] * 200 + 50,   # 50-250mm
}


# ============================================================================
# CATEGORY-BASED RECOMMENDATIONS
# ============================================================================

def recommend_airfoil_type(category):
    """Recommend airfoil characteristics based on flight category"""
    
    recommendations = {
        'soaring': {
            'type': 'High lift, low drag',
            'camber': 'High (3-5%)',
            'thickness': 'Moderate (12-15%)',
            'use_case': 'Long-distance gliding'
        },
        'diving': {
            'type': 'Low drag, high speed',
            'camber': 'Low (1-2%)',
            'thickness': 'Thin (8-10%)',
            'use_case': 'Fast diving, minimal drag'
        },
        'maneuvering': {
            'type': 'High lift, maneuverability',
            'camber': 'Moderate (2-3%)',
            'thickness': 'Thick (15-18%)',
            'use_case': 'Tight turns, slow flight'
        },
        'cruising': {
            'type': 'Balanced efficiency',
            'camber': 'Moderate (2-4%)',
            'thickness': 'Moderate (12-14%)',
            'use_case': 'Long-distance cruising'
        },
        'hovering': {
            'type': 'Maximum lift',
            'camber': 'Very high (5-7%)',
            'thickness': 'Thin (8-12%)',
            'use_case': 'Hovering, rapid acceleration'
        },
        'generalist': {
            'type': 'All-around performance',
            'camber': 'Moderate (2-3%)',
            'thickness': 'Moderate (12-15%)',
            'use_case': 'General purpose flight'
        }
    }
    
    return recommendations.get(category, recommendations['generalist'])

# Use in Streamlit
selected_category = st.selectbox("Select flight category", df['flight_category'].unique())
recommendation = recommend_airfoil_type(selected_category)

st.subheader(f"Airfoil Recommendation for {selected_category.title()}")
st.write(f"**Type:** {recommendation['type']}")
st.write(f"**Camber:** {recommendation['camber']}")
st.write(f"**Thickness:** {recommendation['thickness']}")
st.write(f"**Use Case:** {recommendation['use_case']}")


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

import json

# Load summary JSON
with open('OUTPUT/summary_statistics.json', 'r') as f:
    summary = json.load(f)

st.write(f"Total birds processed: {summary['total_birds']:,}")

for category, stats in summary['categories'].items():
    st.write(f"""
    **{category.upper()}**
    - Count: {stats['count']:,}
    - Percentage: {stats['percentage']:.1f}%
    - Avg Wing Length: {stats.get('avg_wing_length', 'N/A')} mm
    - Avg Aspect Ratio: {stats.get('avg_aspect_ratio', 'N/A')}
    """)


# ============================================================================
# NOTES
# ============================================================================

"""
AVAILABLE FILES:
- birds_with_categories.csv      (main file - use this)
- normalized_data.csv             (0-1 scaled for ML)
- clean_data_with_indices.csv     (raw + indices)
- standardized_data.csv           (z-scores)
- summary_statistics.json         (quick stats)
- categories/category_*.csv       (individual category files)

COLUMNS AVAILABLE:
- species                         (bird name)
- Wing.Length                     (mm)
- Secondary1                      (mm)
- Kipps.Distance                  (mm)
- Tail.Length                     (mm)
- Hand-Wing.Index                 (ratio)
- aspect_ratio                    (calculated)
- wing_loading_proxy              (calculated)
- pointedness_index               (calculated)
- efficiency_index                (calculated)
- flight_category                 (assigned category)

CATEGORIES:
- soaring       (15 birds, 0.1%)
- diving        (987 birds, 9.0%)
- maneuvering   (8,695 birds, 79.2%)
- cruising      (828 birds, 7.5%)
- hovering      (390 birds, 3.6%)
- generalist    (1,291 birds, 11.8%)
"""
