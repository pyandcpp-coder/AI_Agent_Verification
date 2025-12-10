"""
Gender Detection Pipeline using buffalo_l Model (InsightFace)

This module provides gender detection for Aadhaar front and selfie photos
using the buffalo_l model from InsightFace, which includes:
- Face detection
- Gender classification  
- Age estimation
- Face alignment and other attributes

The GenderPipeline is designed to work in parallel with other pipeline components
and returns gender predictions with confidence scores.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenderDetectionResult:
    """Data class for gender detection results"""
    gender: str  # 'male', 'female', or 'unknown'
    confidence: float  # 0-1 confidence score
    age: Optional[float] = None  # Age estimation from face
    face_detected: bool = False  # Whether a face was found
    model_used: str = "buffalo_l"  # Model name for traceability
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'gender': self.gender,
            'confidence': float(self.confidence),
            'age': float(self.age) if self.age else None,
            'face_detected': self.face_detected,
            'model_used': self.model_used
        }


class GenderPipeline:
    """
    Gender Detection Pipeline using InsightFace buffalo_l model.
    
    This pipeline detects gender from images by:
    1. Loading the image (from file or URL)
    2. Using InsightFace to detect faces
    3. Extracting gender attribute from detected faces
    4. Computing confidence scores based on model output
    
    The buffalo_l model includes pre-trained attributes for:
    - Gender classification (male/female)
    - Age estimation
    - Face quality metrics
    """
    
    def __init__(self, model_name: str = 'buffalo_l'):
        """
        Initialize the Gender Detection Pipeline.
        
        Args:
            model_name: InsightFace model to use. Options:
                - 'buffalo_l' (Large, more accurate)
                - 'buffalo_m' (Medium, balanced)
                - 'buffalo_s' (Small, faster)
        """
        logger.info(f"🔄 Initializing Gender Pipeline with model: {model_name}")
        
        try:
            # Initialize InsightFace FaceAnalysis
            self.app = FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            # Prepare the model with face detection size
            # det_size controls the size of detection input - larger = more accurate but slower
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            self.model_name = model_name
            logger.info(f"✓ Gender Pipeline initialized successfully with {model_name}")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize Gender Pipeline: {e}")
            raise RuntimeError(f"Failed to initialize Gender Pipeline: {e}")
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path or URL.
        
        Args:
            image_path: Path to image file (local or URL)
            
        Returns:
            Image as numpy array or None if load fails
        """
        try:
            if isinstance(image_path, str) and (
                image_path.startswith('http://') or 
                image_path.startswith('https://')
            ):
                # Load from URL
                import requests
                from io import BytesIO
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"Could not decode image from URL: {image_path}")
                    return None
                return img
            else:
                # Load from local file
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.warning(f"Could not read image file: {image_path}")
                    return None
                return img
                
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def detect_gender(self, image_path: str) -> GenderDetectionResult:
        """
        Detect gender from a single image using buffalo_l model.
        
        Args:
            image_path: Path to image (local file or URL)
            
        Returns:
            GenderDetectionResult containing:
            - gender: 'male', 'female', or 'unknown'
            - confidence: Confidence score (0-1)
            - age: Estimated age from face
            - face_detected: Boolean indicating if face was found
            
        Pipeline Flow:
        1. Load image from path/URL
        2. Run InsightFace detection on image
        3. If face(s) found:
           a. Extract primary face (highest quality)
           b. Get gender attribute (InsightFace outputs 0=female, 1=male)
           c. Estimate confidence from face quality metrics
           d. Extract age estimation
        4. Return structured result
        """
        
        logger.info(f"🔍 Detecting gender from: {image_path}")
        
        # Load image
        img = self._load_image(image_path)
        if img is None:
            logger.warning(f"⚠ Failed to load image: {image_path}")
            return GenderDetectionResult(
                gender='unknown',
                confidence=0.0,
                face_detected=False
            )
        
        try:
            # Run face detection and analysis
            faces = self.app.get(img)
            
            if not faces:
                logger.warning(f"⚠ No faces detected in image: {image_path}")
                return GenderDetectionResult(
                    gender='unknown',
                    confidence=0.0,
                    face_detected=False
                )
            
            # Use the face with highest detection score (primary face)
            primary_face = max(faces, key=lambda f: f.det_score)
            
            # Extract gender (InsightFace: 0=female, 1=male)
            gender_label = primary_face.gender
            gender_score = primary_face.gender  # Actual attribute value
            
            # Normalize gender to string
            if gender_label == 1:
                gender_str = 'male'
            elif gender_label == 0:
                gender_str = 'female'
            else:
                gender_str = 'unknown'
            
            # Calculate confidence from:
            # 1. Face detection score (how confident the detector is)
            # 2. Face quality metrics if available
            det_score = float(primary_face.det_score)
            
            # Confidence is the detection score itself
            # det_score is typically 0-1, where higher is better
            confidence = min(float(det_score), 1.0)
            
            # Extract age estimation if available
            age = primary_face.age if hasattr(primary_face, 'age') else None
            
            result = GenderDetectionResult(
                gender=gender_str,
                confidence=confidence,
                age=age,
                face_detected=True,
                model_used=self.model_name
            )
            
            logger.info(
                f"✓ Gender detected: {result.gender.upper()} "
                f"(conf: {result.confidence:.2%}, age: {result.age})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Error during gender detection: {e}")
            return GenderDetectionResult(
                gender='unknown',
                confidence=0.0,
                face_detected=False
            )
    
    def detect_gender_batch(
        self, 
        image_paths: List[str]
    ) -> Dict[str, GenderDetectionResult]:
        """
        Detect gender from multiple images.
        
        Args:
            image_paths: List of image paths or URLs
            
        Returns:
            Dictionary mapping image path to GenderDetectionResult
        """
        results = {}
        logger.info(f"🔍 Processing {len(image_paths)} images for gender detection")
        
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"  [{idx}/{len(image_paths)}] Processing: {image_path}")
            results[image_path] = self.detect_gender(image_path)
        
        # Summary
        successful = sum(1 for r in results.values() if r.face_detected)
        logger.info(
            f"✓ Batch processing complete: {successful}/{len(image_paths)} "
            f"images successfully processed"
        )
        
        return results
    
    def detect_gender_with_fallback(
        self,
        primary_image_path: str,
        fallback_image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect gender with fallback logic.
        
        If primary image fails or confidence is too low, 
        tries fallback image.
        
        Args:
            primary_image_path: Primary image for detection (e.g., Aadhaar front)
            fallback_image_path: Fallback image if primary fails (e.g., selfie)
            
        Returns:
            Dictionary with:
            - primary_result: GenderDetectionResult from primary image
            - fallback_result: GenderDetectionResult from fallback (if used)
            - final_gender: Best gender determination
            - final_confidence: Best confidence score
            - used_fallback: Boolean indicating if fallback was used
            
        Logic:
        1. Try primary image
        2. If no face detected or confidence < 0.3, try fallback
        3. Return best result with metadata
        """
        
        logger.info("🔄 Starting gender detection with fallback logic")
        logger.info(f"  Primary: {primary_image_path}")
        if fallback_image_path:
            logger.info(f"  Fallback: {fallback_image_path}")
        
        # Try primary image
        primary_result = self.detect_gender(primary_image_path)
        
        result_dict = {
            'primary_result': primary_result.to_dict(),
            'fallback_result': None,
            'final_gender': primary_result.gender,
            'final_confidence': primary_result.confidence,
            'used_fallback': False
        }
        
        # If primary failed or confidence too low, try fallback
        if fallback_image_path and (
            not primary_result.face_detected or 
            primary_result.confidence < 0.3
        ):
            logger.info("📌 Primary detection insufficient, trying fallback...")
            fallback_result = self.detect_gender(fallback_image_path)
            
            result_dict['fallback_result'] = fallback_result.to_dict()
            
            # Use fallback if it's better
            if (fallback_result.face_detected and 
                fallback_result.confidence > primary_result.confidence):
                logger.info("✓ Using fallback result (higher confidence)")
                result_dict['final_gender'] = fallback_result.gender
                result_dict['final_confidence'] = fallback_result.confidence
                result_dict['used_fallback'] = True
            else:
                logger.info("ℹ Fallback was not better, using primary result")
        
        return result_dict
    
    def compare_genders(
        self,
        aadhar_image_path: str,
        selfie_image_path: str
    ) -> Dict[str, Any]:
        """
        Compare gender detected from Aadhaar and Selfie images.
        
        This is useful for the verification pipeline to check if
        gender from document matches gender from face detection.
        
        Args:
            aadhar_image_path: Path to Aadhaar front image
            selfie_image_path: Path to Selfie image
            
        Returns:
            Dictionary with:
            - aadhar_gender: Gender from Aadhaar
            - aadhar_confidence: Confidence for Aadhaar gender
            - selfie_gender: Gender from Selfie
            - selfie_confidence: Confidence for Selfie gender
            - match: Boolean indicating if genders match
            - match_confidence: Score (0-1) for how well they match
            
        Matching Logic:
        - Both genders must be 'male' or 'female' (not unknown)
        - Genders must be identical
        - match_confidence = average of both confidence scores
        """
        
        logger.info("🔍 Comparing gender across Aadhaar and Selfie")
        
        # Detect gender in both images
        aadhar_result = self.detect_gender(aadhar_image_path)
        selfie_result = self.detect_gender(selfie_image_path)
        
        # Determine if they match
        match = False
        match_confidence = 0.0
        
        if (aadhar_result.face_detected and 
            selfie_result.face_detected and
            aadhar_result.gender != 'unknown' and
            selfie_result.gender != 'unknown'):
            
            match = aadhar_result.gender == selfie_result.gender
            # Match confidence is average of both
            match_confidence = (aadhar_result.confidence + selfie_result.confidence) / 2
            
            if match:
                logger.info(
                    f"✓ Genders MATCH: {aadhar_result.gender.upper()} == "
                    f"{selfie_result.gender.upper()} "
                    f"(confidence: {match_confidence:.2%})"
                )
            else:
                logger.warning(
                    f"✗ Gender MISMATCH: {aadhar_result.gender.upper()} != "
                    f"{selfie_result.gender.upper()}"
                )
        else:
            logger.warning("⚠ Could not compare genders - one or both faces not detected")
        
        return {
            'aadhar_gender': aadhar_result.gender,
            'aadhar_confidence': float(aadhar_result.confidence),
            'aadhar_age': float(aadhar_result.age) if aadhar_result.age else None,
            'selfie_gender': selfie_result.gender,
            'selfie_confidence': float(selfie_result.confidence),
            'selfie_age': float(selfie_result.age) if selfie_result.age else None,
            'match': match,
            'match_confidence': float(match_confidence),
            'both_faces_detected': (
                aadhar_result.face_detected and selfie_result.face_detected
            )
        }


# Standalone function for quick gender detection
def detect_gender_quick(image_path: str, model_name: str = 'buffalo_l') -> str:
    """
    Quick gender detection from a single image.
    
    Returns just the gender string ('male', 'female', or 'unknown').
    Creates a temporary GenderPipeline instance.
    
    Args:
        image_path: Path to image
        model_name: InsightFace model to use
        
    Returns:
        Gender string: 'male', 'female', or 'unknown'
    """
    pipeline = GenderPipeline(model_name)
    result = pipeline.detect_gender(image_path)
    return result.gender
