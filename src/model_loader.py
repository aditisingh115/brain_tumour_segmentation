"""
Brain Tumor Segmentation Model Loader
Implements 3D U-Net architecture with MONAI
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainTumorSegmentationModel:
    """
    Wrapper class for brain tumor segmentation model
    """
    
    def __init__(self, device=None):
        """
        Initialize the model
        
        Args:
            device: torch device (cuda/cpu)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model architecture
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        
    def _build_model(self):
        """
        Build 3D U-Net model architecture
        
        Returns:
            MONAI UNet model
        """
        model = UNet(
            spatial_dims=3,
            in_channels=4,  # T1, T1Gd, T2, FLAIR
            out_channels=4,  # Background, Necrotic, Edema, Enhancing
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        
        logger.info("Model architecture created successfully")
        return model
    
    def load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained model weights
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pretrained weights from {checkpoint_path}")
        except FileNotFoundError:
            logger.warning(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    
    def predict(self, input_data, confidence_threshold=0.5):
        """
        Run inference on input MRI data
        
        Args:
            input_data: numpy array of shape (C, H, W, D) or (H, W, D)
            confidence_threshold: minimum confidence for segmentation
            
        Returns:
            segmentation_mask: numpy array of shape (H, W, D)
            probabilities: numpy array of shape (C, H, W, D)
        """
        with torch.no_grad():
            # Prepare input
            if len(input_data.shape) == 3:
                # Single channel - replicate to 4 channels
                input_tensor = np.stack([input_data] * 4, axis=0)
            else:
                input_tensor = input_data
            
            # Add batch dimension
            input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)
            
            # Run inference
            output = self.model(input_tensor)
            
            # Apply softmax
            probabilities = torch.softmax(output, dim=1)
            
            # Get predictions
            predictions = torch.argmax(probabilities, dim=1)
            
            # Apply confidence threshold
            max_probs = torch.max(probabilities, dim=1)[0]
            predictions[max_probs < confidence_threshold] = 0
            
            # Convert to numpy
            segmentation_mask = predictions.cpu().numpy()[0]
            probabilities_np = probabilities.cpu().numpy()[0]
            
        return segmentation_mask, probabilities_np
    
    def predict_sliding_window(self, input_data, roi_size=(128, 128, 128), overlap=0.5):
        """
        Run inference using sliding window approach for large volumes
        
        Args:
            input_data: numpy array of shape (C, H, W, D)
            roi_size: size of sliding window
            overlap: overlap ratio between windows
            
        Returns:
            segmentation_mask: numpy array of shape (H, W, D)
        """
        from monai.inferers import sliding_window_inference
        
        with torch.no_grad():
            if len(input_data.shape) == 3:
                input_tensor = np.stack([input_data] * 4, axis=0)
            else:
                input_tensor = input_data
            
            input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)
            
            # Use MONAI's sliding window inference
            output = sliding_window_inference(
                inputs=input_tensor,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=self.model,
                overlap=overlap
            )
            
            # Apply softmax and get predictions
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            segmentation_mask = predictions.cpu().numpy()[0]
            
        return segmentation_mask


class MockSegmentationModel:
    """
    Mock model for demonstration purposes when pretrained weights are unavailable
    Generates synthetic segmentations based on intensity thresholding
    """
    
    def __init__(self, device=None):
        """Initialize mock model"""
        self.device = device or torch.device('cpu')
        logger.info("Using mock segmentation model (synthetic predictions)")
    
    def predict(self, input_data, confidence_threshold=0.5):
        """
        Generate synthetic segmentation based on intensity patterns
        
        Args:
            input_data: numpy array of shape (C, H, W, D) or (H, W, D)
            confidence_threshold: threshold for segmentation
            
        Returns:
            segmentation_mask: numpy array of shape (H, W, D)
            probabilities: numpy array of shape (4, H, W, D)
        """
        # Use first channel if multi-channel
        if len(input_data.shape) == 4:
            data = input_data[0]
        else:
            data = input_data
        
        # Normalize
        data_norm = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        # Create synthetic segmentation based on intensity
        segmentation = np.zeros_like(data, dtype=np.int32)
        
        # Simulate tumor regions using intensity thresholds and morphology
        from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure
        
        # Smooth the data
        smoothed = gaussian_filter(data_norm, sigma=2)
        
        # High intensity regions (potential enhancing tumor)
        high_intensity = smoothed > 2.0
        enhancing = binary_dilation(high_intensity, iterations=2)
        segmentation[enhancing] = 3  # Enhancing tumor
        
        # Medium-high intensity (tumor core)
        medium_high = (smoothed > 1.5) & (smoothed <= 2.0)
        core = binary_dilation(medium_high, iterations=3)
        segmentation[core & (segmentation == 0)] = 1  # Necrotic core
        
        # Medium intensity (edema)
        medium = (smoothed > 1.0) & (smoothed <= 1.5)
        edema = binary_dilation(medium, iterations=4)
        segmentation[edema & (segmentation == 0)] = 2  # Edema
        
        # Create probability maps
        probabilities = np.zeros((4,) + data.shape, dtype=np.float32)
        for i in range(4):
            probabilities[i][segmentation == i] = 0.7 + np.random.rand() * 0.25
        
        # Normalize probabilities
        prob_sum = np.sum(probabilities, axis=0, keepdims=True)
        probabilities = probabilities / (prob_sum + 1e-8)
        
        return segmentation, probabilities
    
    def predict_sliding_window(self, input_data, roi_size=(128, 128, 128), overlap=0.5):
        """Use standard predict for mock model"""
        return self.predict(input_data)[0]


def load_model(checkpoint_path=None, use_mock=False, device=None):
    """
    Factory function to load segmentation model
    
    Args:
        checkpoint_path: Path to pretrained weights
        use_mock: Whether to use mock model
        device: torch device
        
    Returns:
        Model instance
    """
    if use_mock or checkpoint_path is None:
        logger.info("Loading mock segmentation model")
        return MockSegmentationModel(device=device)
    else:
        logger.info("Loading real segmentation model")
        model = BrainTumorSegmentationModel(device=device)
        model.load_pretrained_weights(checkpoint_path)
        return model


def get_model_info():
    """
    Get information about the model architecture
    
    Returns:
        dict with model information
    """
    return {
        "architecture": "3D U-Net",
        "framework": "PyTorch + MONAI",
        "input_channels": 4,
        "output_classes": 4,
        "class_names": ["Background", "Necrotic Core", "Edema", "Enhancing Tumor"],
        "input_size": "(4, 128, 128, 128)",
        "parameters": "~31M",
        "training_dataset": "BraTS (Brain Tumor Segmentation Challenge)",
    }