"""
Streamlit Airfoil Visualizer Dashboard
=======================================
Interactive visualization of bird-inspired airfoils
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add core_modules directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core_modules"))

from airfoil_generation import (
    biologically_enhanced_airfoil_generation,
    bezier_curve,
    validate_airfoil_quality
)
from categorisation import load_processed_data, OUTPUT_DIR

# Page configuration
st.set_page_config(
    page_title="Bird Airfoil Generator",
    page_icon="✈️",
    layout="wide"
)

# Directories
AIRFOIL_DIR = Path(OUTPUT_DIR) / "airfoils"
CP_DIR = Path(OUTPUT_DIR) / "control_points"

@st.cache_data
def load_bird_data():
    """Load categorized bird data"""
    return load_processed_data(OUTPUT_DIR)

@st.cache_data
def load_airfoil_summary():
    """Load airfoil database summary"""
    summary_file = AIRFOIL_DIR / "complete_airfoil_database.csv"
    if summary_file.exists():
        return pd.read_csv(summary_file)
    return None

def generate_airfoil_for_bird(bird_data):
    """Generate airfoil for a specific bird"""
    upper_cp, lower_cp, metadata = biologically_enhanced_airfoil_generation(
        bird_data['Wing.Length'],
        bird_data['Secondary1'],
        bird_data['Kipps.Distance'],
        bird_data['Hand-Wing.Index'],
        bird_data['Tail.Length'],
        bird_data['aspect_ratio'],
        bird_data['pointedness_index'],
        bird_data.get('wing_loading_proxy', 0)
    )
    
    # Generate curves
    upper_curve = bezier_curve(upper_cp, 200)
    lower_curve = bezier_curve(lower_cp, 200)
    
    # Validate
    is_valid, issues = validate_airfoil_quality(upper_cp, lower_cp)
    
    return upper_cp, lower_cp, upper_curve, lower_curve, metadata, is_valid, issues

def plot_airfoil(upper_curve, lower_curve, upper_cp, lower_cp, title="", show_cp=True):
    """Create plotly figure of airfoil"""
    fig = go.Figure()
    
    # Add airfoil surfaces
    fig.add_trace(go.Scatter(
        x=upper_curve[:, 0], y=upper_curve[:, 1],
        mode='lines',
        name='Upper Surface',
        line=dict(color='#2E86AB', width=3),
        hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=lower_curve[:, 0], y=lower_curve[:, 1],
        mode='lines',
        name='Lower Surface',
        line=dict(color='#A23B72', width=3),
        hovertemplate='x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
    ))
    
    # Add control points if requested
    if show_cp:
        fig.add_trace(go.Scatter(
            x=upper_cp[:, 0], y=upper_cp[:, 1],
            mode='markers',
            name='Upper Control Points',
            marker=dict(color='#2E86AB', size=8, symbol='circle-open', line=dict(width=2)),
            hovertemplate='CP %{pointNumber}<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=lower_cp[:, 0], y=lower_cp[:, 1],
            mode='markers',
            name='Lower Control Points',
            marker=dict(color='#A23B72', size=8, symbol='circle-open', line=dict(width=2)),
            hovertemplate='CP %{pointNumber}<br>x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>'
        ))
    
    # Styling
    fig.update_layout(
        title=title,
        xaxis_title="x/c (Chord Position)",
        yaxis_title="y/c (Thickness)",
        hovermode='closest',
        template='plotly_white',
        height=500,
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 style="text-align: center;">✈️ Bird-Inspired Airfoil Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em;">Generate aerodynamically valid airfoils from avian morphology</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    try:
        data = load_bird_data()
        df = data['categorized']
    except Exception as e:
        st.error(f"Error loading bird data: {e}")
        st.info("Please run categorisation.py first")
        return
    
    # Sidebar - Selection method
    st.sidebar.header("🔧 Airfoil Selection")
    
    selection_mode = st.sidebar.radio(
        "Selection Mode",
        ["Search by Species", "Browse by Category", "Random Selection", "Top Performers"]
    )
    
    selected_bird = None
    
    # === SELECTION MODES ===
    
    if selection_mode == "Search by Species":
        search_term = st.sidebar.text_input("Search for species name")
        
        if search_term:
            matches = df[df['species'].str.contains(search_term, case=False, na=False)]
            
            if len(matches) > 0:
                st.sidebar.success(f"Found {len(matches)} matches")
                species_list = matches['species'].tolist()
                selected_species = st.sidebar.selectbox("Select species", species_list)
                selected_bird = matches[matches['species'] == selected_species].iloc[0]
            else:
                st.sidebar.warning("No matches found")
    
    elif selection_mode == "Browse by Category":
        category = st.sidebar.selectbox("Flight Category", sorted(df['flight_category'].unique()))
        category_birds = df[df['flight_category'] == category]
        
        # Sort by efficiency
        category_birds = category_birds.sort_values('efficiency_index', ascending=False)
        
        species_list = category_birds['species'].tolist()
        selected_species = st.sidebar.selectbox(f"Select from {len(species_list)} {category} birds", species_list)
        selected_bird = category_birds[category_birds['species'] == selected_species].iloc[0]
    
    elif selection_mode == "Random Selection":
        if st.sidebar.button("🎲 Generate Random Bird"):
            selected_bird = df.sample(1).iloc[0]
            st.sidebar.info(f"Selected: {selected_bird['species']}")
        else:
            st.sidebar.info("Click button to select random bird")
    
    elif selection_mode == "Top Performers":
        metric = st.sidebar.selectbox(
            "Sort by",
            ["efficiency_index", "aspect_ratio", "pointedness_index", "Wing.Length"]
        )
        top_n = st.sidebar.slider("Show top", 5, 50, 10)
        
        top_birds = df.nlargest(top_n, metric)
        species_list = top_birds['species'].tolist()
        selected_species = st.sidebar.selectbox(f"Top {top_n} by {metric}", species_list)
        selected_bird = top_birds[top_birds['species'] == selected_species].iloc[0]
    
    # === MAIN DISPLAY ===
    
    if selected_bird is not None:
        st.header(f"🦅 {selected_bird['species'].replace('_', ' ')}")
        
        # Bird info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Flight Category", selected_bird['flight_category'].upper())
        with col2:
            st.metric("Aspect Ratio", f"{selected_bird['aspect_ratio']:.2f}")
        with col3:
            st.metric("Efficiency Index", f"{selected_bird['efficiency_index']:.3f}")
        with col4:
            st.metric("Wing Length", f"{selected_bird['Wing.Length']:.1f} mm")
        
        st.markdown("---")
        
        # Generate airfoil
        with st.spinner("Generating airfoil..."):
            upper_cp, lower_cp, upper_curve, lower_curve, metadata, is_valid, issues = generate_airfoil_for_bird(selected_bird)
        
        # Validation status
        if is_valid:
            st.success("✅ Airfoil passed all quality checks")
        else:
            st.warning("⚠️ Airfoil has quality issues:")
            for issue in issues:
                st.write(f"  - {issue}")
        
        # Visualization controls
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Display Options")
            show_control_points = st.checkbox("Show Control Points", value=True)
            show_grid = st.checkbox("Show Grid", value=False)
            
            st.markdown("---")
            st.subheader("Export")
            
            # Export coordinates
            combined_coords = np.vstack([upper_curve[::-1], lower_curve[1:]])
            coords_csv = pd.DataFrame(combined_coords, columns=['X', 'Y']).to_csv(index=False)
            
            st.download_button(
                "📥 Download Coordinates (.csv)",
                coords_csv,
                file_name=f"airfoil_{selected_bird['species']}.csv",
                mime="text/csv"
            )
            
            # Export control points
            cp_combined = pd.DataFrame({
                'Surface': ['UPPER']*len(upper_cp) + ['LOWER']*len(lower_cp),
                'Index': list(range(len(upper_cp))) + list(range(len(lower_cp))),
                'X': np.concatenate([upper_cp[:, 0], lower_cp[:, 0]]),
                'Y': np.concatenate([upper_cp[:, 1], lower_cp[:, 1]])
            })
            cp_csv = cp_combined.to_csv(index=False)
            
            st.download_button(
                "📥 Download Control Points (.csv)",
                cp_csv,
                file_name=f"cp_{selected_bird['species']}.csv",
                mime="text/csv"
            )
        
        with col1:
            # Main airfoil plot
            fig = plot_airfoil(
                upper_curve, lower_curve, upper_cp, lower_cp,
                title=f"Airfoil Profile: {selected_bird['species'].replace('_', ' ')}",
                show_cp=show_control_points
            )
            
            if show_grid:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed properties
        st.header("📊 Detailed Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Morphological Data")
            st.write(f"**Wing Length:** {selected_bird['Wing.Length']:.2f} mm")
            st.write(f"**Secondary1:** {selected_bird['Secondary1']:.2f} mm")
            st.write(f"**Kipps Distance:** {selected_bird['Kipps.Distance']:.2f} mm")
            st.write(f"**Hand-Wing Index:** {selected_bird['Hand-Wing.Index']:.2f}")
            st.write(f"**Tail Length:** {selected_bird['Tail.Length']:.2f} mm")
        
        with col2:
            st.subheader("Derived Indices")
            st.write(f"**Aspect Ratio:** {selected_bird['aspect_ratio']:.3f}")
            st.write(f"**Pointedness:** {selected_bird['pointedness_index']:.4f}")
            st.write(f"**Efficiency:** {selected_bird['efficiency_index']:.4f}")
            st.write(f"**Wing Loading:** {selected_bird.get('wing_loading_proxy', 0):.6f}")
        
        with col3:
            st.subheader("Airfoil Geometry")
            st.write(f"**Max Thickness:** {metadata['max_thickness']:.4f}")
            st.write(f"**Thickness Position:** {metadata['max_thickness_pos']:.4f}")
            st.write(f"**Max Camber:** {metadata['max_camber']:.4f}")
            st.write(f"**Camber Position:** {metadata['max_camber_pos']:.4f}")
            st.write(f"**LE Radius:** {metadata['leading_edge_radius']:.4f}")
            st.write(f"**TE Thickness:** {metadata['trailing_edge_thickness']:.4f}")
            st.write(f"**Reynolds #:** {metadata['estimated_reynolds']:.2e}")
        
        st.markdown("---")
        
        # Comparison with category average
        st.header("📈 Comparison with Category Average")
        
        category = selected_bird['flight_category']
        category_birds = df[df['flight_category'] == category]
        
        comparison_metrics = ['aspect_ratio', 'pointedness_index', 'efficiency_index', 'Wing.Length']
        comparison_data = []
        
        for metric in comparison_metrics:
            cat_avg = category_birds[metric].mean()
            bird_val = selected_bird[metric]
            diff_pct = ((bird_val - cat_avg) / cat_avg * 100) if cat_avg != 0 else 0
            
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'This Bird': f"{bird_val:.3f}",
                'Category Avg': f"{cat_avg:.3f}",
                'Difference': f"{diff_pct:+.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    else:
        # Welcome screen
        st.info("👈 Select a bird from the sidebar to generate its airfoil")
        
        # Show statistics
        st.header("📊 Database Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Birds", f"{len(df):,}")
        with col2:
            st.metric("Categories", df['flight_category'].nunique())
        with col3:
            avg_ar = df['aspect_ratio'].mean()
            st.metric("Avg Aspect Ratio", f"{avg_ar:.2f}")
        with col4:
            avg_eff = df['efficiency_index'].mean()
            st.metric("Avg Efficiency", f"{avg_eff:.3f}")
        
        # Category distribution
        st.subheader("Flight Category Distribution")
        
        category_counts = df['flight_category'].value_counts()
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Birds by Flight Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Check if airfoils have been generated
        if AIRFOIL_DIR.exists() and (AIRFOIL_DIR / "complete_airfoil_database.csv").exists():
            airfoil_summary = load_airfoil_summary()
            st.success(f"✅ {len(airfoil_summary)} airfoils have been pre-generated")
            st.info(f"📁 Airfoils location: {AIRFOIL_DIR}")
            st.info(f"📁 Control points location: {CP_DIR}")
        else:
            st.warning("⚠️ Airfoils not yet generated. Run airfoil_generation.py first.")


if __name__ == "__main__":
    main()
