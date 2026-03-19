"""
Streamlit Dashboard Example for AVONET Bird Categorisation
===========================================================
This dashboard demonstrates how to use the processed bird data from categorisation.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Add core_modules to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core_modules"))

# Import the data loading function
from categorisation import load_processed_data, OUTPUT_DIR

# Page configuration
st.set_page_config(
    page_title="AVONET Bird Flight Analysis",
    page_icon="🦅",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all processed data (cached for performance)"""
    return load_processed_data(OUTPUT_DIR)

def main():
    # Header
    st.markdown('<div class="main-header">🦅 AVONET Bird Flight Analysis</div>', unsafe_allow_html=True)
    
    # Quick links
    col1, col2 = st.columns(2)
    with col1:
        st.info("✈️ **New!** [Launch Airfoil Visualizer](http://localhost:8502) - Generate bird-inspired airfoils")
    with col2:
        st.info("📊 View 10,975 categorized birds with morphological analysis")
    
    st.markdown("---")
    
    # Load data
    try:
        data = load_data()
        df = data['categorized']
        summary = data.get('summary', {})
        
        # Sidebar - Category selection
        st.sidebar.header("🔍 Filters")
        
        categories = sorted(df['flight_category'].unique())
        selected_categories = st.sidebar.multiselect(
            "Select Flight Categories",
            categories,
            default=categories
        )
        
        # Filter data
        filtered_df = df[df['flight_category'].isin(selected_categories)]
        
        # Main Dashboard
        # Row 1: Key Metrics
        st.header("📊 Overview Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Birds", f"{len(df):,}")
        with col2:
            st.metric("Filtered Birds", f"{len(filtered_df):,}")
        with col3:
            st.metric("Categories", len(selected_categories))
        with col4:
            avg_wing = filtered_df['Wing.Length'].mean()
            st.metric("Avg Wing Length", f"{avg_wing:.1f} mm")
        
        st.markdown("---")
        
        # Row 2: Distribution Charts
        st.header("📈 Flight Category Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of categories
            category_counts = filtered_df['flight_category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Birds by Flight Category",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart of categories
            fig_bar = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                labels={'x': 'Flight Category', 'y': 'Number of Species'},
                title="Species Count by Category",
                color=category_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # Row 3: Morphological Analysis
        st.header("🔬 Morphological Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: Aspect Ratio vs Wing Length
            fig_scatter1 = px.scatter(
                filtered_df,
                x='Wing.Length',
                y='aspect_ratio',
                color='flight_category',
                title="Aspect Ratio vs Wing Length",
                labels={'Wing.Length': 'Wing Length (mm)', 'aspect_ratio': 'Aspect Ratio'},
                hover_data=['species'],
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_scatter1, use_container_width=True)
        
        with col2:
            # Scatter plot: Pointedness vs Efficiency
            fig_scatter2 = px.scatter(
                filtered_df,
                x='pointedness_index',
                y='efficiency_index',
                color='flight_category',
                title="Wing Pointedness vs Flight Efficiency",
                labels={'pointedness_index': 'Pointedness Index', 'efficiency_index': 'Efficiency Index'},
                hover_data=['species'],
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)
        
        st.markdown("---")
        
        # Row 4: Box plots for each morphological feature
        st.header("📦 Distribution of Morphological Features")
        
        features = ['Wing.Length', 'aspect_ratio', 'pointedness_index', 'efficiency_index']
        feature_labels = {
            'Wing.Length': 'Wing Length (mm)',
            'aspect_ratio': 'Aspect Ratio',
            'pointedness_index': 'Pointedness Index',
            'efficiency_index': 'Efficiency Index'
        }
        
        selected_feature = st.selectbox("Select Feature to Analyze", features, format_func=lambda x: feature_labels[x])
        
        fig_box = px.box(
            filtered_df,
            x='flight_category',
            y=selected_feature,
            color='flight_category',
            title=f"{feature_labels[selected_feature]} Distribution by Flight Category",
            labels={'flight_category': 'Flight Category', selected_feature: feature_labels[selected_feature]},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("---")
        
        # Row 5: Category Details Table
        st.header("📋 Category Statistics")
        
        # Calculate statistics for each category
        category_stats = []
        for cat in selected_categories:
            cat_df = filtered_df[filtered_df['flight_category'] == cat]
            stats = {
                'Category': cat.upper(),
                'Count': len(cat_df),
                'Percentage': f"{len(cat_df)/len(filtered_df)*100:.1f}%",
                'Avg Wing Length': f"{cat_df['Wing.Length'].mean():.1f}",
                'Avg Aspect Ratio': f"{cat_df['aspect_ratio'].mean():.2f}",
                'Avg Pointedness': f"{cat_df['pointedness_index'].mean():.3f}",
                'Avg Efficiency': f"{cat_df['efficiency_index'].mean():.3f}"
            }
            category_stats.append(stats)
        
        stats_df = pd.DataFrame(category_stats)
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")
        
        # Row 6: Individual Bird Explorer
        st.header("🔎 Individual Bird Explorer")
        
        # Search by species name
        search_term = st.text_input("Search for a bird species", "")
        
        if search_term:
            search_results = filtered_df[filtered_df['species'].str.contains(search_term, case=False, na=False)]
            
            if len(search_results) > 0:
                st.success(f"Found {len(search_results)} matching species")
                
                # Display results
                display_cols = ['species', 'flight_category', 'Wing.Length', 'aspect_ratio', 
                               'pointedness_index', 'efficiency_index']
                st.dataframe(search_results[display_cols], use_container_width=True)
                
                # Detailed view of first match
                if st.checkbox("Show detailed view of first match"):
                    bird = search_results.iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Basic Info")
                        st.write(f"**Species:** {bird['species']}")
                        st.write(f"**Category:** {bird['flight_category'].upper()}")
                    
                    with col2:
                        st.subheader("Measurements")
                        st.write(f"**Wing Length:** {bird['Wing.Length']:.1f} mm")
                        st.write(f"**Secondary1:** {bird['Secondary1']:.1f} mm")
                        st.write(f"**Tail Length:** {bird['Tail.Length']:.1f} mm")
                        st.write(f"**Kipps Distance:** {bird['Kipps.Distance']:.1f} mm")
                    
                    with col3:
                        st.subheader("Indices")
                        st.write(f"**Aspect Ratio:** {bird['aspect_ratio']:.2f}")
                        st.write(f"**Pointedness:** {bird['pointedness_index']:.3f}")
                        st.write(f"**Efficiency:** {bird['efficiency_index']:.3f}")
                        st.write(f"**Wing Loading:** {bird['wing_loading_proxy']:.6f}")
            else:
                st.warning("No species found matching your search")
        
        st.markdown("---")
        
        # Row 7: Data Export
        st.header("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name="filtered_bird_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download category statistics
            stats_csv = stats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Category Stats (CSV)",
                data=stats_csv,
                file_name="category_statistics.csv",
                mime="text/csv"
            )
        
        with col3:
            st.info(f"Total records: {len(filtered_df)}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>Data source: AVONET Bird Database</p>
            <p>Processed using categorisation.py</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you've run categorisation.py first to generate the processed data files.")


if __name__ == "__main__":
    main()
