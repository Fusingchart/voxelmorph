# -*- coding: utf-8 -*-
"""
VoxelMorph Framework for Pediatric LV Ejection Fraction Detection
Complete implementation with data integration, diffeomorphic registration,
and pre-trained weight loading capabilities.

This module integrates:
1. Data loading and preprocessing from cardiac video datasets
2. VoxelMorph-based deformable image registration with Scaling and Squaring
3. LV segmentation using U-Net architecture
4. Ejection Fraction calculation from cardiac cycle analysis
5. Pre-trained model weight loading for both registration and segmentation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Optional, Dict
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# ============================================================================
# SPATIAL TRANSFORMER LAYER - Manual Bilinear Sampling
# ============================================================================

class SpatialTransformerLayer(layers.Layer):
    """
    Spatial transformer layer for image warping using manual bilinear sampling.
    
    This layer implements bilinear interpolation without external dependencies,
    making it compatible with all TensorFlow environments.
    """

    def call(self, inputs):
        src, flow = inputs
        return self.flow_warp(src, flow)

    def flow_warp(self, src: tf.Tensor, flow: tf.Tensor) -> tf.Tensor:
        """
        Warp image using flow field with bilinear sampling implemented manually.
        
        Args:
            src: Source image tensor of shape (B, H, W, C)
            flow: Flow field tensor of shape (B, H, W, 2) with (dy, dx) displacements
            
        Returns:
            Warped image tensor of shape (B, H, W, C)
        """
        batch_size = tf.shape(src)[0]
        height = tf.cast(tf.shape(src)[1], tf.float32)
        width = tf.cast(tf.shape(src)[2], tf.float32)

        # Create base grid of (y, x) coordinates
        y = tf.range(0.0, height, dtype=tf.float32)
        x = tf.range(0.0, width, dtype=tf.float32)
        grid_y, grid_x = tf.meshgrid(y, x, indexing='ij')

        # Add flow to get sampled coordinates
        flow_y, flow_x = tf.unstack(flow, axis=-1)
        sampling_coords_y = grid_y + flow_y
        sampling_coords_x = grid_x + flow_x

        # Clamp coordinates to valid range
        sampling_coords_y = tf.clip_by_value(sampling_coords_y, 0.0, height - 1.0)
        sampling_coords_x = tf.clip_by_value(sampling_coords_x, 0.0, width - 1.0)

        # Get integer coordinates for 4 corners
        y0 = tf.cast(tf.floor(sampling_coords_y), tf.int32)
        x0 = tf.cast(tf.floor(sampling_coords_x), tf.int32)
        y1 = y0 + 1
        x1 = x0 + 1

        # Clip indices to bounds
        y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, tf.int32))
        x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, tf.int32))
        y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, tf.int32))
        x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, tf.int32))

        # Get fractional parts for weights
        dy = sampling_coords_y - tf.cast(y0, tf.float32)
        dx = sampling_coords_x - tf.cast(x0, tf.float32)

        # Prepare batch indices
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        batch_indices = tf.reshape(batch_indices, [batch_size, 1, 1])
        batch_indices = tf.tile(batch_indices, [1, tf.shape(src)[1], tf.shape(src)[2]])

        # Define 4 corner indices
        idx00 = tf.stack([batch_indices, y0, x0], axis=-1)
        idx01 = tf.stack([batch_indices, y0, x1], axis=-1)
        idx10 = tf.stack([batch_indices, y1, x0], axis=-1)
        idx11 = tf.stack([batch_indices, y1, x1], axis=-1)

        # Gather values at 4 corners
        val00 = tf.gather_nd(src, idx00)
        val01 = tf.gather_nd(src, idx01)
        val10 = tf.gather_nd(src, idx10)
        val11 = tf.gather_nd(src, idx11)

        # Calculate bilinear weights
        w00 = (1 - dx) * (1 - dy)
        w01 = dx * (1 - dy)
        w10 = (1 - dx) * dy
        w11 = dx * dy

        # Expand weights for channel dimension
        w00 = tf.expand_dims(w00, axis=-1)
        w01 = tf.expand_dims(w01, axis=-1)
        w10 = tf.expand_dims(w10, axis=-1)
        w11 = tf.expand_dims(w11, axis=-1)

        # Apply bilinear interpolation
        warped_image = w00 * val00 + w01 * val01 + w10 * val10 + w11 * val11

        return warped_image


# ============================================================================
# DIFFEOMORPHIC INTEGRATION LAYER - Scaling and Squaring
# ============================================================================

class DiffeomorphicIntegrationLayer(layers.Layer):
    """
    Integrates a velocity field to obtain a diffeomorphic displacement field
    using the Scaling and Squaring method.
    """

    def __init__(self, int_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.int_steps = int_steps
        self.spatial_transformer = SpatialTransformerLayer()

    def call(self, vel_field_scaled: tf.Tensor) -> tf.Tensor:
        """
        Integrate scaled velocity field using Scaling and Squaring.
        
        Args:
            vel_field_scaled: Velocity field scaled by 2^int_steps
            
        Returns:
            Integrated displacement field (phi)
        """
        phi = vel_field_scaled

        for _ in range(self.int_steps):
            # phi_{k+1} = phi_k + warp(v_0, phi_k)
            phi = phi + self.spatial_transformer.flow_warp(vel_field_scaled, phi)

        return phi


# ============================================================================
# VOXELMORPH MODEL - Registration Network
# ============================================================================

class VoxelMorphModel:
    """
    VoxelMorph model for deformable image registration with diffeomorphic
    transformation, optimized for pediatric cardiac imaging.
    """

    def __init__(self, vol_size: Tuple[int, int, int] = (160, 160, 1),
                 enc_nf: List[int] = [16, 32, 32, 32],
                 dec_nf: List[int] = [32, 32, 16],
                 int_steps: int = 7,
                 int_resolution: int = 1):
        """
        Initialize VoxelMorph model.

        Args:
            vol_size: Volume size (height, width, channels)
            enc_nf: Encoder filter numbers
            dec_nf: Decoder filter numbers (corrected for 160x160 output)
            int_steps: Integration steps for diffeomorphic transformation
            int_resolution: Integration resolution
        """
        self.vol_size = vol_size
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        self.int_steps = int_steps
        self.int_resolution = int_resolution
        self.model = None
        self.build_model()

    def unet_core(self, input_shape: Tuple[int, ...],
                  enc_nf: List[int], dec_nf: List[int]) -> keras.Model:
        """Build U-Net encoder-decoder architecture."""
        source = layers.Input(shape=input_shape, name='source')
        target = layers.Input(shape=input_shape, name='target')

        x = layers.Concatenate(axis=-1)([source, target])

        # Encoder
        enc_conv = []
        for i, nf in enumerate(enc_nf):
            if i == 0:
                x = layers.Conv2D(nf, 3, padding='same', name=f'enc_conv_{i}')(x)
            else:
                x = layers.Conv2D(nf, 3, strides=2, padding='same', name=f'enc_conv_{i}')(x)
            x = layers.LeakyReLU(0.2)(x)
            enc_conv.append(x)

        # Decoder with skip connections
        for i, nf in enumerate(dec_nf):
            if i > 0 and i < len(enc_conv):
                x = layers.Concatenate(axis=-1)([x, enc_conv[len(enc_conv) - 1 - i]])

            x = layers.Conv2DTranspose(nf, 3, strides=2, padding='same',
                                       name=f'dec_conv_{i}')(x)
            x = layers.LeakyReLU(0.2)(x)

        # Output velocity field
        x = layers.Conv2D(2, 3, padding='same', name='vel_field')(x)

        return keras.Model([source, target], x, name='unet_core')

    def build_model(self):
        """Build complete VoxelMorph model with diffeomorphic transformation."""
        source = layers.Input(shape=self.vol_size, name='source')
        target = layers.Input(shape=self.vol_size, name='target')

        # U-Net for velocity field
        unet = self.unet_core(self.vol_size, self.enc_nf, self.dec_nf)
        vel_field = unet([source, target])

        # Scale velocity field
        vel_field_scaled = layers.Lambda(
            lambda x: x / (2 ** self.int_steps), 
            name='scaled_vel_field'
        )(vel_field)

        # Diffeomorphic integration
        diffeo_integration_layer = DiffeomorphicIntegrationLayer(
            self.int_steps, 
            name='diffeomorphic_integration'
        )
        def_field = diffeo_integration_layer(vel_field_scaled)

        # Warp source image
        final_warper = SpatialTransformerLayer()
        warped = final_warper([source, def_field])

        self.model = keras.Model([source, target], [warped, def_field], 
                                name='voxelmorph')

    def load_weights(self, weights_path: str):
        """Load pre-trained weights into the VoxelMorph model."""
        if self.model is None:
            logger.warning("VoxelMorph model not built yet. Cannot load weights.")
            return
        if os.path.exists(weights_path):
            logger.info(f"Loading VoxelMorph weights from {weights_path}")
            self.model.load_weights(weights_path)
            logger.info("VoxelMorph weights loaded successfully.")
        else:
            logger.warning(f"VoxelMorph weights file not found at {weights_path}")


# ============================================================================
# LV SEGMENTATION MODEL - U-Net for Cardiac Segmentation
# ============================================================================

class LVSegmenter:
    """U-Net based segmentation model for left ventricle detection."""

    def __init__(self, input_shape: Tuple[int, int, int] = (160, 160, 1)):
        """
        Initialize LV segmenter.
        
        Args:
            input_shape: Input image shape (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = self.build_segmentation_model()

    def build_segmentation_model(self) -> keras.Model:
        """Build U-Net model for LV segmentation."""
        inputs = layers.Input(shape=self.input_shape, name='seg_input')

        # Encoder
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D(2)(c1)

        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D(2)(c2)

        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D(2)(c3)

        c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(c4)

        # Decoder
        u5 = layers.UpSampling2D(2)(c4)
        u5 = layers.Concatenate()([u5, c3])
        c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(u5)
        c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(c5)

        u6 = layers.UpSampling2D(2)(c5)
        u6 = layers.Concatenate()([u6, c2])
        c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(c6)

        u7 = layers.UpSampling2D(2)(c6)
        u7 = layers.Concatenate()([u7, c1])
        c7 = layers.Conv2D(32, 3, activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(32, 3, activation='relu', padding='same')(c7)

        outputs = layers.Conv2D(1, 1, activation='sigmoid', name='seg_output')(c7)

        model = keras.Model(inputs, outputs, name='lv_segmenter')
        return model

    def load_weights(self, weights_path: str):
        """Load pre-trained segmentation weights."""
        if os.path.exists(weights_path):
            logger.info(f"Loading segmentation weights from {weights_path}")
            self.model.load_weights(weights_path)
            logger.info("Segmentation weights loaded successfully.")
        else:
            logger.warning(f"Segmentation weights file not found at {weights_path}")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict LV segmentation.
        
        Args:
            image: Input image of shape (H, W) or (1, H, W, 1)
            
        Returns:
            Segmentation mask
        """
        if len(image.shape) == 2:
            image = np.expand_dims(np.expand_dims(image, 0), -1)
        elif len(image.shape) == 3 and image.shape[-1] != 1:
            image = np.expand_dims(image, -1)

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        prediction = self.model.predict(image, verbose=0)
        return prediction.squeeze()


# ============================================================================
# EJECTION FRACTION CALCULATOR
# ============================================================================

class EjectionFractionCalculator:
    """Static utility class for ejection fraction computation."""

    @staticmethod
    def calculate_lv_volume(mask: np.ndarray, pixel_size: float = 1.0) -> float:
        """
        Calculate LV volume from segmentation mask.
        
        Args:
            mask: Binary segmentation mask
            pixel_size: Physical size of each pixel in mm
            
        Returns:
            Volume in mm³
        """
        area_mm2 = np.sum(mask > 0.5) * (pixel_size ** 2)
        # Approximate volume assuming revolving contour (Simpson's rule)
        volume = area_mm2 * pixel_size
        return volume

    @staticmethod
    def calculate_ef(edv: float, esv: float) -> float:
        """
        Calculate ejection fraction.
        
        Args:
            edv: End-diastolic volume
            esv: End-systolic volume
            
        Returns:
            Ejection fraction as percentage
        """
        if edv == 0:
            return 0.0
        ef = ((edv - esv) / edv) * 100
        return np.clip(ef, 0, 100)

    @staticmethod
    def find_cardiac_phases(volumes: np.ndarray) -> Tuple[int, int]:
        """
        Identify ED and ES phases from volume curve.
        
        Args:
            volumes: Array of volumes across frames
            
        Returns:
            (ed_idx, es_idx) indices of end-diastole and end-systole
        """
        ed_idx = int(np.argmax(volumes))
        es_idx = int(np.argmin(volumes))
        return ed_idx, es_idx


# ============================================================================
# PEDIATRIC VOXELMORPH FRAMEWORK
# ============================================================================

class PediatricVoxelMorphEF:
    """
    Complete framework for pediatric LV ejection fraction detection.
    
    Integrates VoxelMorph registration, LV segmentation, and EF calculation.
    """

    def __init__(self, 
                 vol_size: Tuple[int, int, int] = (160, 160, 1),
                 vm_weights_path: Optional[str] = None,
                 seg_weights_path: Optional[str] = None,
                 int_steps: int = 7):
        """
        Initialize PediatricVoxelMorphEF framework.
        
        Args:
            vol_size: Volume size for registration
            vm_weights_path: Path to pre-trained VoxelMorph weights
            seg_weights_path: Path to pre-trained segmentation weights
            int_steps: Integration steps for diffeomorphic transformation
        """
        self.vol_size = vol_size
        self.int_steps = int_steps

        # Initialize models
        self.voxelmorph = VoxelMorphModel(
            vol_size=vol_size,
            int_steps=int_steps
        )
        self.segmenter = LVSegmenter(input_shape=vol_size)

        # Load weights if provided
        if vm_weights_path:
            self.voxelmorph.load_weights(vm_weights_path)
        if seg_weights_path:
            self.segmenter.load_weights(seg_weights_path)

        logger.info("PediatricVoxelMorphEF framework initialized successfully.")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess cardiac frame for model input.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame normalized to [0, 1]
        """
        # Resize to model input size
        frame = cv2.resize(frame, (self.vol_size[1], self.vol_size[0]))

        # Normalize
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max > frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        else:
            frame = np.zeros_like(frame, dtype=np.float32)

        # Add channel dimension if needed
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, -1)

        return frame.astype(np.float32)

    def register_frames(self, source_frame: np.ndarray, 
                       target_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register source frame to target frame.
        
        Args:
            source_frame: Source frame
            target_frame: Target frame
            
        Returns:
            (warped_image, deformation_field)
        """
        if self.voxelmorph.model is None:
            logger.error("VoxelMorph model is not built. Cannot register frames.")
            raise RuntimeError("VoxelMorph model is not initialized.")

        source = self.preprocess_frame(source_frame)
        target = self.preprocess_frame(target_frame)

        # Add batch dimension
        source = np.expand_dims(source, 0)
        target = np.expand_dims(target, 0)

        # Predict warping
        warped, def_field = self.voxelmorph.model.predict(
            [source, target], 
            verbose=0
        )

        return warped[0], def_field[0]

    def segment_lv(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment LV in cardiac frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Segmentation mask
        """
        processed_frame = self.preprocess_frame(frame)
        mask = self.segmenter.predict(processed_frame)
        return mask

    def calculate_ef_from_video(self, 
                                video_frames: List[np.ndarray],
                                pixel_size: float = 1.0) -> Dict[str, float]:
        """
        Calculate ejection fraction from video frames.
        
        Args:
            video_frames: List of video frames
            pixel_size: Physical pixel size in mm
            
        Returns:
            Dictionary with EF metrics
        """
        volumes = []

        logger.info(f"Processing {len(video_frames)} frames for EF calculation...")

        for i, frame in enumerate(video_frames):
            mask = self.segment_lv(frame)
            volume = EjectionFractionCalculator.calculate_lv_volume(
                mask, 
                pixel_size=pixel_size
            )
            volumes.append(volume)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed frame {i + 1}/{len(video_frames)}")

        volumes = np.array(volumes)

        # Find cardiac phases
        ed_idx, es_idx = EjectionFractionCalculator.find_cardiac_phases(volumes)
        edv = volumes[ed_idx]
        esv = volumes[es_idx]

        # Calculate EF
        ef = EjectionFractionCalculator.calculate_ef(edv, esv)

        results = {
            'ef': ef,
            'edv': edv,
            'esv': esv,
            'ed_frame_idx': int(ed_idx),
            'es_frame_idx': int(es_idx),
            'volumes': volumes.tolist()
        }

        logger.info(f"EF Calculation Complete: EF={ef:.1f}%, EDV={edv:.0f}, ESV={esv:.0f}")

        return results

    def visualize_results(self, 
                         video_frames: List[np.ndarray],
                         results: Dict[str, float],
                         save_path: Optional[str] = None):
        """
        Visualize registration, segmentation, and EF results.
        
        Args:
            video_frames: List of video frames
            results: Results dictionary from calculate_ef_from_video
            save_path: Optional path to save visualization
        """
        ed_idx = results['ed_frame_idx']
        es_idx = results['es_frame_idx']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"VoxelMorph EF Analysis - EF: {results['ef']:.1f}%", 
                     fontsize=16, fontweight='bold')

        # ED frame
        ed_frame = self.preprocess_frame(video_frames[int(ed_idx)])
        axes[0, 0].imshow(ed_frame.squeeze(), cmap='gray')
        axes[0, 0].set_title(f'ED Frame {ed_idx}')
        axes[0, 0].axis('off')

        # ED segmentation
        ed_mask = self.segment_lv(video_frames[int(ed_idx)])
        axes[0, 1].imshow(ed_frame.squeeze(), cmap='gray')
        axes[0, 1].imshow(ed_mask, cmap='Reds', alpha=0.5)
        axes[0, 1].set_title(f'ED Segmentation')
        axes[0, 1].axis('off')

        # ES frame
        es_frame = self.preprocess_frame(video_frames[int(es_idx)])
        axes[0, 2].imshow(es_frame.squeeze(), cmap='gray')
        axes[0, 2].set_title(f'ES Frame {es_idx}')
        axes[0, 2].axis('off')

        # ES segmentation
        es_mask = self.segment_lv(video_frames[int(es_idx)])
        axes[1, 0].imshow(es_frame.squeeze(), cmap='gray')
        axes[1, 0].imshow(es_mask, cmap='Reds', alpha=0.5)
        axes[1, 0].set_title(f'ES Segmentation')
        axes[1, 0].axis('off')

        # Volume curve
        volumes = np.array(results['volumes'])
        axes[1, 1].plot(volumes, 'b-', linewidth=2)
        axes[1, 1].plot(ed_idx, results['edv'], 'go', markersize=10, label='ED')
        axes[1, 1].plot(es_idx, results['esv'], 'ro', markersize=10, label='ES')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Volume')
        axes[1, 1].set_title('LV Volume Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # EF metrics
        axes[1, 2].axis('off')
        metrics_text = (
            f"Ejection Fraction: {results['ef']:.1f}%\n"
            f"EDV: {results['edv']:.0f} mm³\n"
            f"ESV: {results['esv']:.0f} mm³\n"
            f"ED Frame: {ed_idx}\n"
            f"ES Frame: {es_idx}"
        )
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                       verticalalignment='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        plt.show()


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_video_frames(video_path: str, 
                     max_frames: Optional[int] = None,
                     resize_to: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    """
    Load frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load
        resize_to: Optional (height, width) to resize frames
        
    Returns:
        List of video frames as numpy arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize if specified
        if resize_to:
            frame = cv2.resize(frame, resize_to)

        frames.append(frame)
        frame_count += 1

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    logger.info(f"Loaded {len(frames)} frames from {video_path}")

    return frames


def normalize_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Normalize frame values to [0, 1]."""
    normalized = []
    for frame in frames:
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max > frame_min:
            frame = (frame - frame_min) / (frame_max - frame_min)
        normalized.append(frame.astype(np.float32))
    return normalized


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def test_framework():
    """Test the complete framework with synthetic data."""
    logger.info("=" * 70)
    logger.info("Testing PediatricVoxelMorphEF Framework")
    logger.info("=" * 70)

    # Initialize framework
    framework = PediatricVoxelMorphEF(
        vol_size=(160, 160, 1),
        int_steps=7
    )

    # Generate synthetic cardiac video (30 frames)
    logger.info("Generating synthetic cardiac video...")
    frames = []
    for i in range(30):
        # Synthetic frame with moving circle (simulating LV)
        frame = np.zeros((160, 160), dtype=np.uint8)
        center_y = 80
        center_x = 80
        radius = int(30 + 10 * np.sin(2 * np.pi * i / 30))
        cv2.circle(frame, (center_x, center_y), radius, 255, -1)
        # Add noise
        frame = frame.astype(np.float32) + np.random.normal(0, 10, frame.shape)
        frame = np.clip(frame, 0, 255)
        frames.append(frame)

    logger.info(f"Generated {len(frames)} synthetic frames")

    # Calculate EF
    results = framework.calculate_ef_from_video(frames, pixel_size=1.0)

    # Visualize results
    logger.info("Generating visualization...")
    framework.visualize_results(frames, results)

    logger.info("Framework test completed successfully!")
    return framework, frames, results


if __name__ == "__main__":
    # Run framework test
    framework, frames, results = test_framework()
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Ejection Fraction: {results['ef']:.1f}%")
    print(f"End-Diastolic Volume: {results['edv']:.0f}")
    print(f"End-Systolic Volume: {results['esv']:.0f}")
    print(f"ED Frame Index: {results['ed_frame_idx']}")
    print(f"ES Frame Index: {results['es_frame_idx']}")
    print("=" * 70)
