"""
Upload & Preprocess Page
Handles MRI file upload and preprocessing
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocess import MRIPreprocessor, quick_preprocess, create_sample_mri
from visualize import plot_slice
from utils import validate_nifti
import matplotlib.pyplot as plt

st.set_page_config(page_title="Upload & Preprocess", page_icon="📤", layout="wide")

# Initialize session state
if 'mri_data' not in st.session_state:
    st.session_state.mri_data = None
if 'mri_metadata' not in st.session_state:
    st.session_state.mri_metadata = None
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None

# Header
st.markdown("""
    <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); 
                padding: 2rem; border-radius: 10px; color: white; text-align: center;">
        <h1> Upload & Preprocess MRI Scan</h1>
        <p style="font-size: 1.1rem;">Load your brain MRI and apply preprocessing pipeline</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main content
tab1, tab2, tab3 = st.tabs([" Upload", " Preprocess", " Preview"])

# ============ TAB 1: UPLOAD ============
with tab1:
    st.markdown("###  Upload MRI Scan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
            **Supported Format:** NIfTI (.nii, .nii.gz)
            
            **Requirements:**
            - 3D volumetric MRI scan
            - Brain MRI (T1, T1Gd, T2, or FLAIR)
            - File size < 500 MB
            - Standard radiological orientation
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a NIfTI file",
            type=['nii', 'gz'],
            help="Upload a brain MRI scan in NIfTI format"
        )
        
        # Option to use sample data
        st.markdown("---")
        st.markdown("**Or try with sample data:**")
        
        if st.button(" Generate Sample MRI", type="secondary"):
            with st.spinner("Generating synthetic MRI..."):
                sample_data = create_sample_mri(shape=(128, 128, 128), add_tumor=True)
                st.session_state.original_data = sample_data
                st.session_state.mri_data = sample_data
                st.session_state.mri_metadata = {
                    'original_shape': sample_data.shape,
                    'original_spacing': (1.0, 1.0, 1.0),
                    'source': 'synthetic'
                }
                st.success("Sample MRI generated successfully!")
                st.rerun()
    
    with col2:
        st.markdown("###  File Status")
        
        if uploaded_file is not None:
            st.success(" File uploaded")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
            
            # Validate and load file
            if st.button(" Validate & Load", type="primary"):
                with st.spinner("Validating NIfTI file..."):
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Validate
                        is_valid, message, metadata = validate_nifti(tmp_path)
                        
                        if is_valid:
                            st.success(f" {message}")
                            
                            # Load data
                            from preprocess import MRIPreprocessor
                            preprocessor = MRIPreprocessor()
                            data, affine, header = preprocessor.load_nifti(tmp_path)
                            
                            st.session_state.original_data = data
                            st.session_state.mri_data = data
                            st.session_state.mri_metadata = metadata
                            st.session_state.preprocessing_done = False
                            
                            st.rerun()
                        else:
                            st.error(f" {message}")
                    
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
        
        elif st.session_state.mri_data is not None:
            st.success(" MRI loaded")
            if st.session_state.mri_metadata:
                source = st.session_state.mri_metadata.get('source', 'uploaded')
                st.write(f"**Source:** {source}")
        else:
            st.info(" No file loaded")
    
    # Display metadata if available
    if st.session_state.mri_metadata is not None:
        st.markdown("---")
        st.markdown("###  MRI Metadata")
        
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        
        with meta_col1:
            shape = st.session_state.mri_metadata.get('shape', 'N/A')
            st.metric("Image Dimensions", f"{shape[0]}×{shape[1]}×{shape[2]}" if shape != 'N/A' else 'N/A')
        
        with meta_col2:
            spacing = st.session_state.mri_metadata.get('spacing', (0, 0, 0))
            st.metric("Voxel Spacing", f"{spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm")
        
        with meta_col3:
            dtype = st.session_state.mri_metadata.get('dtype', 'N/A')
            st.metric("Data Type", dtype)
        
        # Additional info in expander
        with st.expander(" Detailed Information"):
            st.json(st.session_state.mri_metadata)

# ============ TAB 2: PREPROCESS ============
with tab2:
    st.markdown("###  Preprocessing Pipeline")
    
    if st.session_state.original_data is None:
        st.warning(" Please upload an MRI scan first in the Upload tab.")
    else:
        # Preprocessing options
        st.markdown("#### Configuration")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            apply_n4 = st.checkbox("Apply N4 Bias Correction", value=True,
                                   help="Removes intensity inhomogeneities")
            apply_skull_strip = st.checkbox("Apply Skull Stripping", value=True,
                                           help="Isolates brain tissue")
            enhance_contrast = st.checkbox("Enhance Contrast", value=False,
                                          help="Applies adaptive histogram equalization")
        
        with config_col2:
            normalization = st.selectbox(
                "Normalization Method",
                ["zscore", "minmax", "percentile"],
                help="Choose intensity normalization technique"
            )
            
            target_shape = st.selectbox(
                "Target Shape",
                ["128×128×128", "96×96×96", "160×160×160"],
                help="Output dimensions for model input"
            )
            
            # Parse target shape
            target_dims = tuple(map(int, target_shape.split('×')))
        
        st.markdown("---")
        
        # Preprocessing button
        if st.button("▶ Run Preprocessing Pipeline", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize preprocessor
                status_text.text("Initializing preprocessor...")
                progress_bar.progress(10)
                
                preprocessor = MRIPreprocessor(
                    target_spacing=(1.0, 1.0, 1.0),
                    target_shape=target_dims
                )
                
                data = st.session_state.original_data.copy()
                metadata = {'steps_applied': [], 'original_shape': data.shape}
                
                # N4 correction
                if apply_n4:
                    status_text.text("Applying N4 bias correction...")
                    progress_bar.progress(25)
                    data = preprocessor.n4_bias_correction(data)
                    metadata['steps_applied'].append('n4_correction')
                
                # Skull stripping
                if apply_skull_strip:
                    status_text.text("Performing skull stripping...")
                    progress_bar.progress(45)
                    data, mask = preprocessor.skull_stripping(data)
                    metadata['steps_applied'].append('skull_stripping')
                
                # Contrast enhancement
                if enhance_contrast:
                    status_text.text("Enhancing contrast...")
                    progress_bar.progress(60)
                    data = preprocessor.enhance_contrast(data)
                    metadata['steps_applied'].append('contrast_enhancement')
                
                # Normalization
                status_text.text(f"Normalizing intensity ({normalization})...")
                progress_bar.progress(75)
                data = preprocessor.normalize_intensity(data, method=normalization)
                metadata['steps_applied'].append(f'normalization_{normalization}')
                
                # Resampling
                status_text.text("Resampling to target shape...")
                progress_bar.progress(90)
                data = preprocessor.resample(data, original_spacing=(1.0, 1.0, 1.0))
                metadata['steps_applied'].append('resampling')
                metadata['final_shape'] = data.shape
                
                # Complete
                progress_bar.progress(100)
                status_text.text(" Preprocessing complete!")
                
                # Save to session state
                st.session_state.preprocessed_data = data
                st.session_state.mri_data = data
                st.session_state.preprocessing_done = True
                st.session_state.mri_metadata.update(metadata)
                
                st.success(f" Preprocessing completed! Applied: {', '.join(metadata['steps_applied'])}")
                
            except Exception as e:
                st.error(f" Error during preprocessing: {str(e)}")
                progress_bar.empty()
                status_text.empty()
        
        # Show preprocessing status
        if st.session_state.preprocessing_done:
            st.markdown("---")
            st.info(f" **Status:** Preprocessing complete. Data ready for segmentation.")
            
            if st.session_state.mri_metadata:
                steps = st.session_state.mri_metadata.get('steps_applied', [])
                st.write(f"**Steps applied:** {', '.join(steps)}")

# ============ TAB 3: PREVIEW ============
with tab3:
    st.markdown("###  Data Preview")
    
    if st.session_state.mri_data is None:
        st.warning("⚠️ No MRI data loaded. Please upload and preprocess first.")
    else:
        data = st.session_state.mri_data
        
        # Slice selector
        st.markdown("#### Slice Navigation")
        
        slice_col1, slice_col2 = st.columns([3, 1])
        
        with slice_col1:
            slice_idx = st.slider(
                "Select slice",
                min_value=0,
                max_value=data.shape[2] - 1,
                value=data.shape[2] // 2,
                help="Navigate through axial slices"
            )
        
        with slice_col2:
            axis = st.selectbox(
                "View plane",
                [("Axial", 2), ("Sagittal", 0), ("Coronal", 1)],
                format_func=lambda x: x[0]
            )[1]
        
        # Display slice
        st.markdown("---")
        
        viz_col1, viz_col2 = st.columns([2, 1])
        
        with viz_col1:
            try:
                fig = plot_slice(data, slice_idx, axis=axis, title="MRI Slice")
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error displaying slice: {str(e)}")
        
        with viz_col2:
            st.markdown("#### Intensity Statistics")
            
            if axis == 0:
                slice_data = data[slice_idx, :, :]
            elif axis == 1:
                slice_data = data[:, slice_idx, :]
            else:
                slice_data = data[:, :, slice_idx]
            
            st.metric("Min Intensity", f"{np.min(slice_data):.2f}")
            st.metric("Max Intensity", f"{np.max(slice_data):.2f}")
            st.metric("Mean Intensity", f"{np.mean(slice_data):.2f}")
            st.metric("Std Dev", f"{np.std(slice_data):.2f}")
        
        # Comparison view if both original and preprocessed exist
        if st.session_state.original_data is not None and st.session_state.preprocessed_data is not None:
            st.markdown("---")
            st.markdown("####  Before/After Comparison")
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Original**")
                try:
                    # Resample original to same size for fair comparison
                    orig_slice_idx = int(slice_idx * st.session_state.original_data.shape[2] / data.shape[2])
                    orig_slice_idx = min(orig_slice_idx, st.session_state.original_data.shape[2] - 1)
                    
                    fig_orig = plot_slice(
                        st.session_state.original_data,
                        orig_slice_idx,
                        axis=axis,
                        title="Original",
                        figsize=(6, 6)
                    )
                    st.pyplot(fig_orig)
                    plt.close()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            with comp_col2:
                st.markdown("**Preprocessed**")
                try:
                    fig_prep = plot_slice(
                        st.session_state.preprocessed_data,
                        slice_idx,
                        axis=axis,
                        title="Preprocessed",
                        figsize=(6, 6)
                    )
                    st.pyplot(fig_prep)
                    plt.close()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Navigation helper
st.markdown("---")
st.info("""
    ###  Next Steps
    
    Once preprocessing is complete, navigate to the **⚙️ Segmentation** page to run the AI model.
""")

# Footer with status
st.markdown("---")
status_cols = st.columns(3)

with status_cols[0]:
    status = " Loaded" if st.session_state.mri_data is not None else " Pending"
    st.metric("Data Status", status)

with status_cols[1]:
    prep_status = " Complete" if st.session_state.preprocessing_done else " Pending"
    st.metric("Preprocessing", prep_status)

with status_cols[2]:
    if st.session_state.mri_data is not None:
        shape = st.session_state.mri_data.shape
        st.metric("Current Shape", f"{shape[0]}×{shape[1]}×{shape[2]}")
    else:
        st.metric("Current Shape", "N/A")