"""
Results & Metrics Page
Visualization and quantitative analysis of segmentation results
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from visualize import plot_overlay, plot_multiplane, create_3d_volume, create_interactive_slice_viewer, plot_probability_maps
from utils import calculate_all_metrics, get_tumor_center, get_tumor_extent, format_metric_for_display
import matplotlib.pyplot as plt

st.set_page_config(page_title="Results & Metrics", page_icon="📊", layout="wide")

# Initialize metrics in session state
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# Header
st.markdown("""
    <div style="background: linear-gradient(135deg, #ec4899 0%, #f59e0b 100%); 
                padding: 2rem; border-radius: 10px; color: white; text-align: center;">
        <h1> Results & Metrics</h1>
        <p style="font-size: 1.1rem;">Comprehensive visualization and quantitative analysis</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Check if segmentation is done
if st.session_state.get('segmentation_result') is None:
    st.warning(" No segmentation results available.")
    
    st.info("""
        ###  Required Steps:
        1. Upload and preprocess an MRI scan
        2. Run the segmentation model
        3. Return here to view results
    """)
    
    st.stop()

# Get data
mri_data = st.session_state.mri_data
seg_data = st.session_state.segmentation_result

# Calculate metrics if not already done
if not st.session_state.metrics:
    with st.spinner("Calculating metrics..."):
        st.session_state.metrics = calculate_all_metrics(seg_data)

metrics = st.session_state.metrics

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    " Key Metrics",
    " 2D Visualization",
    " 3D Visualization",
    " Detailed Analysis"
])

# ============ TAB 1: KEY METRICS ============
with tab1:
    st.markdown("### Quantitative Measurements")
    
    # Tumor volume metrics
    st.markdown("####  Tumor Volume Analysis")
    
    vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
    
    with vol_col1:
        total_vol = metrics.get('total_tumor_volume_cm3', 0)
        st.metric(
            "Total Tumor Volume",
            f"{total_vol:.2f} cm³",
            delta=f"{total_vol * 1000:.0f} mm³"
        )
    
    with vol_col2:
        necrotic_vol = metrics.get('volume_class_1_cm3', 0)
        st.metric(
            "Necrotic Core",
            f"{necrotic_vol:.2f} cm³",
            delta=f"{necrotic_vol / total_vol * 100:.1f}%" if total_vol > 0 else "0%"
        )
    
    with vol_col3:
        edema_vol = metrics.get('volume_class_2_cm3', 0)
        st.metric(
            "Edema",
            f"{edema_vol:.2f} cm³",
            delta=f"{edema_vol / total_vol * 100:.1f}%" if total_vol > 0 else "0%"
        )
    
    with vol_col4:
        enhancing_vol = metrics.get('volume_class_3_cm3', 0)
        st.metric(
            "Enhancing Tumor",
            f"{enhancing_vol:.2f} cm³",
            delta=f"{enhancing_vol / total_vol * 100:.1f}%" if total_vol > 0 else "0%"
        )
    
    # Volume distribution chart
    st.markdown("---")
    st.markdown("####  Volume Distribution")
    
    if total_vol > 0:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Necrotic Core', 'Edema', 'Enhancing Tumor'],
            values=[necrotic_vol, edema_vol, enhancing_vol],
            marker=dict(colors=['#ef4444', '#22c55e', '#eab308']),
            hole=0.4,
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        
        fig_pie.update_layout(
            title="Tumor Composition",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No tumor detected in the scan.")
    
    # Spatial characteristics
    st.markdown("---")
    st.markdown("####  Spatial Characteristics")
    
    spatial_col1, spatial_col2 = st.columns(2)
    
    with spatial_col1:
        # Center of mass
        center = get_tumor_center(seg_data)
        
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4> Tumor Center of Mass</h4>
                <p style="font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;">
                    ({:.1f}, {:.1f}, {:.1f})
                </p>
                <p style="color: #64748b; font-size: 0.9rem;">
                    Coordinates in voxel space
                </p>
            </div>
        """.format(center[0], center[1], center[2]), unsafe_allow_html=True)
    
    with spatial_col2:
        # Bounding box
        extent = get_tumor_extent(seg_data)
        
        if extent:
            size_x = extent[0][1] - extent[0][0]
            size_y = extent[1][1] - extent[1][0]
            size_z = extent[2][1] - extent[2][0]
            
            st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4> Tumor Extent (Bounding Box)</h4>
                    <p style="font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;">
                        {size_x} × {size_y} × {size_z} voxels
                    </p>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        X: [{extent[0][0]}, {extent[0][1]}]<br>
                        Y: [{extent[1][0]}, {extent[1][1]}]<br>
                        Z: [{extent[2][0]}, {extent[2][1]}]
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # Surface area
    st.markdown("---")
    surface_area = metrics.get('surface_area_mm2', 0)
    
    st.metric(
        "Tumor Surface Area",
        f"{surface_area:.2f} mm²",
        delta=f"{surface_area / 100:.2f} cm²"
    )

# ============ TAB 2: 2D VISUALIZATION ============
with tab2:
    st.markdown("###  2D Slice Viewer")
    
    # Slice controls
    viz_col1, viz_col2, viz_col3 = st.columns([2, 1, 1])
    
    with viz_col1:
        slice_idx = st.slider(
            "Slice Index",
            min_value=0,
            max_value=mri_data.shape[2] - 1,
            value=int(get_tumor_center(seg_data)[2]),
            help="Navigate through slices"
        )
    
    with viz_col2:
        axis = st.selectbox(
            "View Plane",
            [("Axial", 2), ("Sagittal", 0), ("Coronal", 1)],
            format_func=lambda x: x[0]
        )[1]
    
    with viz_col3:
        alpha = st.slider(
            "Overlay Alpha",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1
        )
    
    # Display overlay
    st.markdown("---")
    try:
        fig = plot_overlay(
            mri_data,
            seg_data,
            slice_idx=slice_idx,
            axis=axis,
            alpha=alpha,
            figsize=(15, 5)
        )
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error displaying overlay: {str(e)}")
    
    # Multi-plane view
    st.markdown("---")
    st.markdown("###  Multi-Plane View")
    
    if st.checkbox("Show orthogonal views", value=True):
        center = get_tumor_center(seg_data)
        slice_indices = (int(center[0]), int(center[1]), int(center[2]))
        
        try:
            fig_multi = plot_multiplane(
                mri_data,
                seg_data,
                slice_indices=slice_indices,
                alpha=alpha,
                figsize=(18, 6)
            )
            st.pyplot(fig_multi)
            plt.close()
        except Exception as e:
            st.error(f"Error displaying multi-plane view: {str(e)}")
    
    # Interactive slice viewer
    st.markdown("---")
    st.markdown("###  Interactive Slice Viewer")
    
    if st.checkbox("Show interactive viewer", value=False):
        with st.spinner("Creating interactive viewer..."):
            try:
                fig_interactive = create_interactive_slice_viewer(mri_data, seg_data)
                st.plotly_chart(fig_interactive, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating interactive viewer: {str(e)}")

# ============ TAB 3: 3D VISUALIZATION ============
with tab3:
    st.markdown("###  3D Volume Rendering")
    
    st.info("""
        **Interactive 3D Visualization**
        - Use mouse to rotate the volume
        - Scroll to zoom in/out
        - Click and drag to pan
    """)
    
    # Rendering options
    render_col1, render_col2 = st.columns([1, 3])
    
    with render_col1:
        downsample = st.select_slider(
            "Resolution",
            options=[1, 2, 3, 4],
            value=2,
            help="Lower = higher quality but slower"
        )
        
        show_legend = st.checkbox("Show Legend", value=True)
    
    with render_col2:
        if st.button(" Generate 3D Visualization", type="primary"):
            with st.spinner("Creating 3D visualization... This may take a moment."):
                try:
                    fig_3d = create_3d_volume(seg_data, downsample=downsample)
                    
                    if not show_legend:
                        fig_3d.update_layout(showlegend=False)
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    st.success(" 3D visualization created!")
                    
                except Exception as e:
                    st.error(f"Error creating 3D visualization: {str(e)}")
        else:
            st.info(" Click the button above to generate 3D visualization")
    
    st.markdown("---")
    st.markdown("###  3D Visualization Tips")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
            **Navigation:**
            -  Left-click + drag: Rotate
            -  Right-click + drag: Pan
            -  Scroll: Zoom in/out
            -  Double-click: Reset view
        """)
    
    with tips_col2:
        st.markdown("""
            **Performance:**
            - Higher downsample = faster rendering
            - Lower downsample = more detail
            - Recommended: 2x for balance
            - Use 1x only for small tumors
        """)

# ============ TAB 4: DETAILED ANALYSIS ============
with tab4:
    st.markdown("###  Detailed Statistical Analysis")
    
    # Probability maps if available
    if st.session_state.get('segmentation_probabilities') is not None:
        st.markdown("####  Class Probability Maps")
        
        probs = st.session_state.segmentation_probabilities
        prob_slice = st.slider(
            "Probability Map Slice",
            min_value=0,
            max_value=probs.shape[-1] - 1,
            value=probs.shape[-1] // 2
        )
        
        try:
            fig_probs = plot_probability_maps(
                probs,
                slice_idx=prob_slice,
                axis=2,
                figsize=(16, 4)
            )
            st.pyplot(fig_probs)
            plt.close()
        except Exception as e:
            st.error(f"Error displaying probability maps: {str(e)}")
    
    st.markdown("---")
    
    # Detailed metrics table
    st.markdown("####  Complete Metrics Report")
    
    metrics_data = []
    
    # Volume metrics
    metrics_data.append({
        "Category": "Volume",
        "Metric": "Total Tumor Volume",
        "Value": format_metric_for_display(metrics.get('total_tumor_volume_cm3'), 'volume')
    })
    
    for i, name in enumerate(["Necrotic Core", "Edema", "Enhancing Tumor"], 1):
        metrics_data.append({
            "Category": "Volume",
            "Metric": name,
            "Value": format_metric_for_display(metrics.get(f'volume_class_{i}_cm3'), 'volume')
        })
    
    # Spatial metrics
    metrics_data.append({
        "Category": "Spatial",
        "Metric": "Surface Area",
        "Value": format_metric_for_display(metrics.get('surface_area_mm2'), 'area')
    })
    
    center = get_tumor_center(seg_data)
    metrics_data.append({
        "Category": "Spatial",
        "Metric": "Center of Mass",
        "Value": f"({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
    })
    
    st.table(metrics_data)
    
    # Export options
    st.markdown("---")
    st.markdown("####  Export Options")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button(" Export Metrics as JSON", use_container_width=True):
            import json
            
            metrics_json = json.dumps(st.session_state.metrics, indent=2)
            st.download_button(
                label="Download JSON",
                data=metrics_json,
                file_name="segmentation_metrics.json",
                mime="application/json"
            )
    
    with export_col2:
        if st.button(" Export as CSV", use_container_width=True):
            import pandas as pd
            
            df = pd.DataFrame(metrics_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="segmentation_metrics.csv",
                mime="text/csv"
            )

# Summary card at bottom
st.markdown("---")
st.success(f"""
    ###  Analysis Complete
    
    **Summary:**
    - Total tumor volume: {metrics.get('total_tumor_volume_cm3', 0):.2f} cm³
    - Detected {len(np.unique(seg_data)) - 1} distinct tumor classes
    - Surface area: {metrics.get('surface_area_mm2', 0):.2f} mm²
""")

# Footer
st.markdown("---")
st.caption(" Results & Metrics | Brain Tumor Segmentation System")