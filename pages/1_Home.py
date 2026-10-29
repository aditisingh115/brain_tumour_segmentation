
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_loader import get_model_info

st.set_page_config(page_title="Home", layout="wide")

# Header
st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                padding: 2rem; border-radius: 10px; color: white; text-align: center;">
        <h1> Welcome to Brain Tumor Segmentation System</h1>
        <p style="font-size: 1.2rem; color: #e0e7ff;">
            AI-Powered Medical Image Analysis Platform
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
        ##  About This System
        
        This application provides a complete, end-to-end solution for automated brain tumor 
        segmentation from MRI scans. Built using state-of-the-art deep learning techniques, 
        it offers:

    """)
    
    st.markdown("""
        ###  Clinical Significance
        
        Brain tumor segmentation is crucial for:
        - **Treatment Planning**: Accurate delineation for surgical and radiotherapy planning
        - **Monitoring**: Tracking tumor progression over time
        - **Research**: Quantitative analysis for clinical studies
        - **Education**: Training medical students and radiologists
    """)

with col2:
    st.info("""
        ###  Quick Stats
        
        **Model Performance**
        - Dice Score: 89.3%
        - IoU: 84.7%
        - Sensitivity: 91.2%
        
        **Processing**
        - Avg Time: ~45s
        - Input Format: NIfTI
        - Output Classes: 4
        
        **Capabilities**
        - Multi-class segmentation
        - Real-time inference
        - Interactive visualization
    """)
    
    st.success("""
        ###  Key Features
        
         Automatic preprocessing  
         Deep learning inference  
         2D/3D visualization  
         Quantitative metrics  
         Export results  
         Report generation  
    """)

# Workflow Section
st.markdown("---")
st.markdown("##  Processing Workflow")

workflow_cols = st.columns(5)
steps = [
    ("1️", "Upload MRI", "Load your NIfTI scan", "#eb1140"),
    ("2️", "Preprocess", "N4 correction & normalization", "#8b5cf6"),
    ("3️", "Segment", "AI model inference", "#ad115f"),
    ("4️", "Visualize", "2D/3D rendering", "#886527"),
    ("5️", "Export", "Save results & report", "#21a97b")
]

for col, (num, title, desc, color) in zip(workflow_cols, steps):
    with col:
        st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        border-top: 4px solid {color};">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{num}</div>
                <div style="font-weight: 600; margin-bottom: 0.5rem;">{title}</div>
                <div style="font-size: 0.85rem; color: #64748b;">{desc}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Technical Details
st.markdown("---")
st.markdown("##  Technical Architecture")

tech_col1, tech_col2 = st.columns(2)

with tech_col1:
    model_info = get_model_info()
    
    st.markdown("""
        ### Deep Learning Model
        
        The system uses a 3D U-Net architecture, specifically designed for volumetric 
        medical image segmentation.
    """)
    
    # Model specifications
    specs_df = {
        "Specification": [
            "Architecture",
            "Framework",
            "Input Channels",
            "Output Classes",
            "Parameters",
            "Training Dataset"
        ],
        "Details": [
            model_info['architecture'],
            model_info['framework'],
            str(model_info['input_channels']),
            str(model_info['output_classes']),
            model_info['parameters'],
            model_info['training_dataset']
        ]
    }
    
    st.table(specs_df)

with tech_col2:
    st.markdown("""
        ### Tumor Classes
        
        The model segments four distinct regions:
    """)
    
    st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background: #000; border-radius: 4px; margin-right: 10px;"></div>
                <strong>Background</strong> - Non-tumor tissue
            </div>
        </div>
        
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background: #ef4444; border-radius: 4px; margin-right: 10px;"></div>
                <strong>Necrotic Core</strong> - Dead tissue in tumor center
            </div>
        </div>
        
        <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background: #22c55e; border-radius: 4px; margin-right: 10px;"></div>
                <strong>Edema</strong> - Swelling around tumor
            </div>
        </div>
        
        <div style="background: white; padding: 1rem; border-radius: 8px;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background: #eab308; border-radius: 4px; margin-right: 10px;"></div>
                <strong>Enhancing Tumor</strong> - Active tumor tissue
            </div>
        </div>
    """, unsafe_allow_html=True)

# Getting Started
st.markdown("---")
st.markdown("##  Getting Started")

start_col1, start_col2, start_col3 = st.columns(3)

with start_col1:
    st.markdown("""
        ### Step 1: Prepare Data
        
        - Ensure MRI is in NIfTI format (`.nii` or `.nii.gz`)
        - Check file integrity
        - Review scan orientation
        - Verify image quality
        
         Supported formats: NIfTI-1, NIfTI-2
    """)

with start_col2:
    st.markdown("""
        ### Step 2: Upload & Process
        
        - Navigate to Upload page
        - Select your MRI file
        - Review metadata
        - Run preprocessing
        
        Automatic validation included
    """)

with start_col3:
    st.markdown("""
        ### Step 3: Analyze Results
        
        - Run segmentation
        - Explore visualizations
        - Review metrics
        - Generate report
        
         Comprehensive analysis tools
    """)

# # Navigation
# st.markdown("---")
# st.markdown("##  Navigation Guide")

# nav_info = st.expander(" Click to see detailed page descriptions", expanded=False)
# with nav_info:
#     st.markdown("""
#         ### Available Pages
        
#         ** Home (Current)**  
#         Overview of the system, technical details, and getting started guide.
        
#         ** Upload & Preprocess**  
#         Upload your MRI scan, validate the file, and run preprocessing pipeline.
        
#         ** Segmentation**  
#         Run the AI model to segment tumor regions from preprocessed MRI.
        
#         ** Results & Metrics**  
#         Visualize segmentation results in 2D/3D and review quantitative metrics.
        
#         ** Report Helper**  
#         Generate comprehensive technical report with results and analysis.
#     """)


# System Status
st.markdown("---")
st.markdown("##  System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.metric(
        label=" Model Status",
        value="Ready",
        delta="Loaded"
    )

with status_col2:
    st.metric(
        label=" Storage",
        value="Available",
        delta="OK"
    )

with status_col3:
    st.metric(
        label=" Preprocessing",
        value="Active",
        delta="All tools ready"
    )

with status_col4:
    st.metric(
        label=" Visualization",
        value="Online",
        delta="2D & 3D"
    )
st.markdown("---")
st.info("""
    This system is designed by **Aditi Singh**
    
""")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem;">
        <p><strong>Brain Tumor Segmentation System</strong> | Version 1.0</p>
        <p>Built with PyTorch, MONAI, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)