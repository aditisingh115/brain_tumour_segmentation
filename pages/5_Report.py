"""
Report Helper Page
Generate comprehensive technical report
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import create_report_data, get_tumor_center, get_tumor_extent

st.set_page_config(page_title="Report Helper", page_icon="📄", layout="wide")

# Header
st.markdown("""
    <div style="background: linear-gradient(135deg, #f59e0b 0%, #10b981 100%); 
                padding: 2rem; border-radius: 10px; color: white; text-align: center;">
        <h1> Report Generator</h1>
        <p style="font-size: 1.1rem;">Create comprehensive technical documentation</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Check if data is available
if st.session_state.get('segmentation_result') is None:
    st.warning(" Complete the segmentation workflow first.")
    st.stop()

# Generate report button
if st.button(" Generate Report(For Major Project, Future)", type="primary", use_container_width=True):
    
    # Get all data
    mri_data = st.session_state.mri_data
    seg_data = st.session_state.segmentation_result
    metrics = st.session_state.metrics
    mri_metadata = st.session_state.mri_metadata
    
    # Create report data
    report_data = create_report_data(
        mri_metadata,
        seg_data,
        metrics
    )
    
# Quick stats
st.markdown("---")
st.markdown("###  Quick Summary")

sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

with sum_col1:
    st.metric("Total Volume", f"{st.session_state.metrics.get('total_tumor_volume_cm3', 0):.2f} cm³")

with sum_col2:
    st.metric("Tumor Classes", len(np.unique(st.session_state.segmentation_result)) - 1)

with sum_col3:
    st.metric("Processing Time", f"{st.session_state.get('segmentation_time', 0):.2f}s")

with sum_col4:
    st.metric("Report Status", "Ready" if st.session_state.segmentation_result is not None else "Pending")

st.markdown("---")
st.caption(" Report Helper | Brain Tumor Segmentation System")