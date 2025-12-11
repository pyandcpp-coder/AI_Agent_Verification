import logging
import math 
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import torch
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityAgent:
    def __init__(self, model_path="models/best.pt", other_lang_code='hin+tel+ben', save_debug_images=True):

        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA is available. Models will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("CUDA not available. Models will fall back to CPU.")

        # --- Model Loading ---
        self.entity_model_path = model_path
        self.save_debug_images = save_debug_images
        
        # Create debug output directories
        if self.save_debug_images:
            self.debug_base_dir = Path("debug_output")
            self.debug_base_dir.mkdir(exist_ok=True)
            logger.info(f"Debug images will be saved to: {self.debug_base_dir}")
        
        logger.info("Checking for entity detection YOLO model on filesystem...")
        if not Path(self.entity_model_path).exists():
            logger.critical(f"Entity model not found at {self.entity_model_path}. Aborting startup.")
            raise FileNotFoundError(f"Entity model not found at {self.entity_model_path}")

        logger.info("Loading entity detection model from filesystem...")
        self.model2 = YOLO(self.entity_model_path)
        
        self.other_lang_code = other_lang_code
        self._check_tesseract()

        logger.info("YOLOv8 entity detection model loaded successfully.")
        logger.info(f"EntityAgent initialized to use '{self.other_lang_code}' for other language fields.")
        
        self.entity_classes = {
            0: 'aadharnumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
            4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
            8: 'name', 9: 'name_otherlang', 10: 'pincode'
        }

    def _check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            logger.critical("Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
            raise RuntimeError("Tesseract not found")
        except Exception as e:
             logger.critical(f"An error occurred while checking Tesseract: {e}")
             raise RuntimeError(f"Error checking Tesseract: {e}")

    def _detect_card_rotation(self, img: np.ndarray, session_dir: Path = None) -> int:
        """
        Detect if the entire Aadhaar card is rotated.
        
        Strategy:
        1. Try all 4 rotations (0°, 90°, 180°, 270°)
        2. For each rotation, run entity detection
        3. Score based on:
           - Number of entities detected
           - Average confidence of detections
           - Presence of critical entities (aadhaar, dob, gender)
        4. Return the best rotation angle
        
        Returns:
            Rotation angle: 0, 90, 180, or 270
        """
        logger.info("\n🔄 Detecting card-level rotation...")
        
        rotation_scores = {}
        rotation_details = {}
        
        for angle in [0, 90, 180, 270]:
            # Rotate image
            if angle == 0:
                rotated = img.copy()
            elif angle == 90:
                rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
            # Run detection
            results = self.model2(rotated, device=self.device, verbose=False)
            
            detected_entities = []
            total_conf = 0
            critical_entities = {'aadharnumber': 0, 'dob': 0, 'gender': 0}
            
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf >= 0.15:  # Use same threshold
                    class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
                    detected_entities.append((class_name, conf))
                    total_conf += conf
                    
                    # Track critical entities
                    if class_name in critical_entities:
                        critical_entities[class_name] += 1
            
            # Calculate score
            num_entities = len(detected_entities)
            avg_conf = total_conf / num_entities if num_entities > 0 else 0
            critical_count = sum(1 for v in critical_entities.values() if v > 0)
            
            # Scoring formula:
            # - 40% weight on number of entities detected
            # - 30% weight on average confidence
            # - 30% weight on critical entities present
            score = (
                (num_entities / 10) * 0.4 +  # Normalize by expected max ~10 entities
                avg_conf * 0.3 +
                (critical_count / 3) * 0.3  # 3 critical entities
            )
            
            rotation_scores[angle] = score
            rotation_details[angle] = {
                'num_entities': num_entities,
                'avg_confidence': avg_conf,
                'critical_entities': critical_entities,
                'entities': detected_entities
            }
            
            logger.info(f"  Rotation {angle}°: score={score:.3f}, entities={num_entities}, "
                       f"avg_conf={avg_conf:.3f}, critical={critical_count}/3")
        
        # Select best rotation
        best_rotation = max(rotation_scores.keys(), key=lambda k: rotation_scores[k])
        best_score = rotation_scores[best_rotation]
        
        logger.info(f"✓ Best card rotation: {best_rotation}° (score: {best_score:.3f})")
        
        # Save visualization if debug enabled
        if self.save_debug_images and session_dir:
            self._save_rotation_comparison(img, rotation_details, best_rotation, session_dir)
        
        return best_rotation

    def _save_rotation_comparison(self, img: np.ndarray, rotation_details: dict, 
                                   best_rotation: int, session_dir: Path):
        """Create a comparison visualization of all 4 rotations."""
        try:
            comparisons = []
            
            for angle in [0, 90, 180, 270]:
                # Rotate image
                if angle == 0:
                    rotated = img.copy()
                elif angle == 90:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle == 180:
                    rotated = cv2.rotate(img, cv2.ROTATE_180)
                elif angle == 270:
                    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                # Resize to consistent size for comparison
                h, w = rotated.shape[:2]
                if h > 600:
                    scale = 600 / h
                    new_w, new_h = int(w * scale), 600
                    rotated = cv2.resize(rotated, (new_w, new_h))
                
                # Add info text
                details = rotation_details[angle]
                is_best = angle == best_rotation
                color = (0, 255, 0) if is_best else (255, 255, 255)
                thickness = 3 if is_best else 2
                
                cv2.putText(rotated, f"{angle} degrees", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
                cv2.putText(rotated, f"Entities: {details['num_entities']}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(rotated, f"Conf: {details['avg_confidence']:.2f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if is_best:
                    cv2.rectangle(rotated, (0, 0), (rotated.shape[1]-1, rotated.shape[0]-1),
                                 (0, 255, 0), 5)
                    cv2.putText(rotated, "BEST", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                comparisons.append(rotated)
            
            # Create 2x2 grid
            top_row = np.hstack([comparisons[0], comparisons[1]])
            bottom_row = np.hstack([comparisons[2], comparisons[3]])
            grid = np.vstack([top_row, bottom_row])
            
            rotation_path = session_dir / "00_card_rotation_detection.jpg"
            cv2.imwrite(str(rotation_path), grid)
            logger.info(f"  Saved rotation comparison: {rotation_path}")
            
        except Exception as e:
            logger.warning(f"  Could not save rotation comparison: {e}")

    def _check_critical_fields(self, final_data: Dict[str, Any], card_side: str) -> Dict[str, bool]:
        """
        Check if critical fields are present and valid.
        Returns dict with field name and whether it's missing/invalid.
        """
        if card_side.lower() == 'front':
            critical_fields = ['aadharnumber', 'dob', 'gender']
        else:  # back
            critical_fields = ['aadharnumber']
        
        missing = {}
        for field in critical_fields:
            value = final_data.get(field, "")
            
            if field == 'aadharnumber':
                # Check if valid 12-digit number
                digits = re.sub(r'\D', '', str(value))
                missing[field] = len(digits) != 12
            elif field == 'dob':
                # Check if not "Invalid Format" or "Not Detected"
                missing[field] = value in ['Invalid Format', 'Not Detected', '', None]
            elif field == 'gender':
                # Check if not "Not Detected" or "Other"
                missing[field] = value in ['Not Detected', 'Other', '', None]
            else:
                missing[field] = not value
        
        return missing

    def _merge_results(self, cropped_data: Dict[str, Any], full_data: Dict[str, Any], 
                       missing_fields: Dict[str, bool]) -> Dict[str, Any]:
        """
        Merge results from cropped and full image detections.
        Prefer full image results for missing/invalid fields.
        """
        merged = cropped_data.copy()
        
        logger.info("🔀 Merging results from cropped and full image detections...")
        
        for field, is_missing in missing_fields.items():
            if is_missing:
                full_value = full_data.get(field)
                cropped_value = cropped_data.get(field)
                
                # For aadharnumber, check if full version is better
                if field == 'aadharnumber':
                    full_digits = re.sub(r'\D', '', str(full_value)) if full_value else ""
                    cropped_digits = re.sub(r'\D', '', str(cropped_value)) if cropped_value else ""
                    
                    if len(full_digits) == 12 and len(cropped_digits) != 12:
                        logger.info(f"  ✓ Using full image {field}: {full_digits}")
                        merged[field] = full_digits
                        merged['aadhar_status'] = full_data.get('aadhar_status', 'aadhar_approved')
                    elif len(full_digits) > len(cropped_digits):
                        logger.info(f"  ✓ Using full image {field} (more digits): {full_digits}")
                        merged[field] = full_digits
                
                # For other fields, use full if cropped is invalid
                elif full_value and full_value not in ['Invalid Format', 'Not Detected', 'Other', '']:
                    logger.info(f"  ✓ Using full image {field}: {full_value}")
                    merged[field] = full_value
                    
                    # Update related status fields
                    if field == 'dob':
                        merged['age'] = full_data.get('age')
                        merged['age_status'] = full_data.get('age_status')
        
        return merged

    def extract_from_file(self, file_path: str, crop_coords: List[int] = None, 
                         confidence_threshold: float = 0.15, card_side: str = 'front'):
        """
        Main Entry Point with CARD ROTATION DETECTION + FALLBACK mechanism.
        """
        try:
            # Create session-specific debug directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = None
            if self.save_debug_images:
                session_dir = self.debug_base_dir / f"session_{timestamp}_{card_side}"
                session_dir.mkdir(exist_ok=True)
                logger.info(f"Session debug directory: {session_dir}")
            
            # 1. Read ORIGINAL Image
            original_img = cv2.imread(file_path)
            if original_img is None:
                logger.error(f"Failed to read image: {file_path}")
                return {"error": "failed_to_read_file"}

            # 2. DETECT CARD-LEVEL ROTATION
            card_rotation = self._detect_card_rotation(original_img, session_dir)
            
            # 3. Apply card rotation to original image
            if card_rotation != 0:
                logger.info(f"📐 Applying card rotation: {card_rotation}°")
                if card_rotation == 90:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif card_rotation == 180:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_180)
                elif card_rotation == 270:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
                
                # Save rotated image
                if self.save_debug_images and session_dir:
                    rotated_path = session_dir / f"00a_rotated_{card_rotation}deg.jpg"
                    cv2.imwrite(str(rotated_path), original_img)
                    logger.info(f"  Saved rotated image: {rotated_path}")

            # === ATTEMPT 1: Try with Crop Coordinates ===
            logger.info("=" * 60)
            logger.info("🔍 ATTEMPT 1: Extraction with CROP coordinates")
            logger.info("=" * 60)
            
            img = original_img.copy()
            
            # Apply crop if coords provided
            if crop_coords:
                x1, y1, x2, y2 = crop_coords
                h, w = img.shape[:2]
                
                # IMPORTANT: Adjust crop coordinates if card was rotated
                if card_rotation == 90:
                    # When rotated 90° CCW: x,y swap and adjust
                    x1, y1, x2, y2 = y1, w - x2, y2, w - x1
                elif card_rotation == 180:
                    # When rotated 180°: flip both dimensions
                    x1, y1, x2, y2 = w - x2, h - y2, w - x1, h - y1
                elif card_rotation == 270:
                    # When rotated 270° (90° CW): x,y swap and adjust
                    x1, y1, x2, y2 = h - y2, x1, h - y1, x2
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                logger.info(f"Cropping image to: [{x1}, {y1}, {x2}, {y2}] (after rotation adjustment)")
                
                # Save visualization of crop region
                if self.save_debug_images and session_dir:
                    img_with_crop_box = img.copy()
                    cv2.rectangle(img_with_crop_box, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(img_with_crop_box, f"Crop Region: {card_side}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    crop_box_path = session_dir / f"00b_crop_region_{card_side}.jpg"
                    cv2.imwrite(str(crop_box_path), img_with_crop_box)
                
                img = img[y1:y2, x1:x2]
                
                if self.save_debug_images and session_dir:
                    input_path = session_dir / f"00c_cropped_input_{card_side}.jpg"
                    cv2.imwrite(str(input_path), img)
            else:
                logger.info("No crop coordinates provided, using full image")

            # Run detection and OCR on cropped image
            all_detections = self.detect_entities_in_image(img, confidence_threshold, card_side, session_dir)
            
            if not all_detections:
                logger.warning("⚠️  No entities detected in cropped region")
                cropped_result = {
                    "success": False,
                    "message": "no_entities_detected",
                    "data": {"aadharnumber": "", "dob": "Not Detected", "gender": "Not Detected"}
                }
            else:
                self.crop_entities(all_detections, session_dir)
                ocr_results = self.perform_multi_language_ocr(all_detections)
                organized = self.organize_results_by_card_type(all_detections, ocr_results, confidence_threshold)
                cropped_data = self.extract_main_fields(organized)
                
                cropped_result = {
                    "success": True,
                    "data": cropped_data
                }
            
            # === CHECK: Are critical fields missing? ===
            logger.info("\n" + "=" * 60)
            logger.info("🔍 Checking for missing critical fields...")
            logger.info("=" * 60)
            
            missing_fields = self._check_critical_fields(cropped_result.get("data", {}), card_side)
            has_missing = any(missing_fields.values())
            
            if has_missing:
                logger.warning("⚠️  CRITICAL FIELDS MISSING from cropped extraction:")
                for field, is_missing in missing_fields.items():
                    if is_missing:
                        logger.warning(f"  ❌ {field}: Missing or Invalid")
                
                # === ATTEMPT 2: FALLBACK to Full Image ===
                logger.info("\n" + "=" * 60)
                logger.info("🔄 ATTEMPT 2: FALLBACK - Running on FULL ORIGINAL image")
                logger.info("=" * 60)
                
                # Run detection on FULL original image (already rotated)
                full_detections = self.detect_entities_in_image(
                    original_img, confidence_threshold, card_side, session_dir
                )
                
                if full_detections:
                    # Create subfolder for full image results
                    if session_dir:
                        full_crops_dir = session_dir / "crops_fullimage"
                        full_crops_dir.mkdir(exist_ok=True)
                    else:
                        full_crops_dir = None
                    
                    self.crop_entities(full_detections, full_crops_dir)
                    full_ocr_results = self.perform_multi_language_ocr(full_detections)
                    full_organized = self.organize_results_by_card_type(
                        full_detections, full_ocr_results, confidence_threshold
                    )
                    full_data = self.extract_main_fields(full_organized)
                    
                    # Merge results
                    logger.info("\n" + "=" * 60)
                    logger.info("🔀 MERGING results from both attempts...")
                    logger.info("=" * 60)
                    
                    final_data = self._merge_results(
                        cropped_result.get("data", {}), 
                        full_data, 
                        missing_fields
                    )
                    
                    logger.info("\n✅ Final merged data:")
                    for key in ['aadharnumber', 'dob', 'gender', 'age', 'age_status', 'aadhar_status']:
                        logger.info(f"  {key}: {final_data.get(key)}")
                    
                    return {
                        "success": True,
                        "data": final_data,
                        "debug_dir": str(session_dir) if session_dir else None,
                        "card_rotation": card_rotation,
                        "used_fallback": True
                    }
                else:
                    logger.error("❌ Fallback detection on full image also failed")
                    return cropped_result
            else:
                logger.info("✅ All critical fields present in cropped extraction")
                return {
                    "success": True,
                    "data": cropped_result.get("data", {}),
                    "debug_dir": str(session_dir) if session_dir else None,
                    "card_rotation": card_rotation,
                    "used_fallback": False
                }

        except Exception as e:
            logger.error(f"Error in extraction: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def detect_entities_in_image(self, image_input, confidence_threshold: float, card_side: str = 'front', session_dir: Path = None):
        """
        Modified to accept card_side parameter and filter entities accordingly.
        Also saves annotated detection images.
        """
        logger.info(f"\nStep 1: Detecting entities in image (Side: {card_side}, Threshold: {confidence_threshold})")
        
        # Handle input type
        if isinstance(image_input, str):
            img = cv2.imread(str(image_input))
            input_name = Path(image_input).stem
        elif isinstance(image_input, np.ndarray):
            img = image_input
            input_name = "memory_crop"
        else:
            return {}

        if img is None:
            return {}
        
        # Define entities based on card side
        if card_side.lower() == 'front':
            target_entities = {'aadharnumber', 'dob', 'gender', 'name', 'name_otherlang'}
        elif card_side.lower() == 'back':
            target_entities = {'aadharnumber', 'address', 'address_other_lang', 'pincode', 'mobile_no', 'city'}
        else:
            # Default to basic entities if side not specified
            target_entities = {'aadharnumber', 'dob', 'gender'}
        
        logger.info(f"  Target entities for {card_side}: {target_entities}")
        
        # Run entity detection
        results = self.model2(img, device=self.device, verbose=False)
        card_detections = []
        
        # Create annotated image
        annotated_img = img.copy()
        
        for box in results[0].boxes:
            if float(box.conf[0]) < confidence_threshold: 
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
            
            # Only process target entities
            if class_name not in target_entities:
                continue
            
            card_detections.append({
                'bbox': (x1, y1, x2, y2), 
                'class_name': class_name, 
                'confidence': float(box.conf[0])
            })
            
            # Draw bounding box on annotated image
            color = (0, 255, 0)  # Green
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{class_name}: {float(box.conf[0]):.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 5), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        logger.info(f"  Detected {len(card_detections)} entities")
        
        # Save annotated image
        if self.save_debug_images and session_dir:
            annotated_path = session_dir / f"01_detections_{card_side}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_img)
            logger.info(f"Saved annotated detections: {annotated_path}")
        
        if not card_detections:
            return {}

        # Wrap in your original structure
        all_detections = {
            input_name: {
                "card_image": img,
                "card_type": card_side,
                "detections": card_detections
            }
        }
        
        return all_detections

    def crop_entities(self, all_detections: Dict[str, Dict[str, Any]], session_dir: Path = None):
        """Step 3: Crop individual entities with bounds checking and save crops"""
        logger.info(f"\nStep 3: Cropping individual entities")
        
        # Create crops directory
        crops_dir = None
        if self.save_debug_images and session_dir:
            crops_dir = session_dir if isinstance(session_dir, Path) and session_dir.name.startswith("crops") else session_dir / "crops"
            crops_dir.mkdir(exist_ok=True)
        
        for card_name, card_data in all_detections.items():
            img = card_data['card_image']
            h, w = img.shape[:2]
            card_type = card_data.get('card_type', 'unknown')
            
            for i, detection in enumerate(card_data['detections']):
                x1, y1, x2, y2 = detection['bbox']
                
                # Sanitize bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"  Invalid bbox for {detection['class_name']}, skipping")
                    detection['cropped_image'] = None
                    continue
                
                crop = img[y1:y2, x1:x2]
                detection['cropped_image'] = crop
                entity_key = f"{card_name}_{detection['class_name']}_{i}"
                detection['entity_key'] = entity_key
                
                # Save individual crop
                if self.save_debug_images and crops_dir:
                    class_name = detection['class_name']
                    conf = detection['confidence']
                    crop_filename = f"{card_type}_{class_name}_{i}_conf{conf:.2f}.jpg"
                    crop_path = crops_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                    logger.info(f"  Saved crop: {crop_filename}")
        
        return all_detections
    
    def _preprocess_for_aadhaar_ocr(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Specialized preprocessing for Aadhaar numbers which are often in specific fonts/formats.
        """
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Resize if too small (Aadhaar numbers need good resolution)
            h, w = gray.shape
            if h < 50:
                scale = 50 / h
                new_w, new_h = int(w * scale), int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Try multiple preprocessing approaches and pick best
            preprocessed_versions = []
            
            # Version 1: Simple threshold
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_versions.append(('otsu', thresh1))
            
            # Version 2: Adaptive threshold
            thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            preprocessed_versions.append(('adaptive', thresh2))
            
            # Version 3: Enhanced contrast + threshold
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_versions.append(('clahe', thresh3))
            
            # Version 4: Denoising + threshold
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, thresh4 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_versions.append(('denoised', thresh4))
            
            return preprocessed_versions
            
        except Exception as e:
            logger.warning(f"  Error in Aadhaar preprocessing: {e}")
            return [('original', img)]
    
    def _extract_aadhaar_with_multiple_methods(self, img: np.ndarray) -> str:
        """
        Try multiple OCR methods specifically for Aadhaar numbers.
        Tries ALL 4 rotations for EACH preprocessing method.
        """
        try:
            preprocessed_versions = self._preprocess_for_aadhaar_ocr(img)
            
            # Try different PSM modes and configurations
            configs = [
                '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line, digits only
                '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word, digits only
                '--psm 6 -c tessedit_char_whitelist=0123456789',  # Block of text, digits only
                '--psm 7',  # Single line, all chars
                '--psm 13',  # Raw line
            ]
            
            best_result = ""
            best_digit_count = 0
            
            # Try all 4 rotations for each preprocessing method
            for method_name, processed_img in preprocessed_versions:
                for rotation in [0, 90, 180, 270]:
                    # Rotate the preprocessed image
                    if rotation == 0:
                        rotated_img = processed_img
                    elif rotation == 90:
                        rotated_img = cv2.rotate(processed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotation == 180:
                        rotated_img = cv2.rotate(processed_img, cv2.ROTATE_180)
                    elif rotation == 270:
                        rotated_img = cv2.rotate(processed_img, cv2.ROTATE_90_CLOCKWISE)
                    
                    for config in configs:
                        try:
                            from PIL import Image
                            pil_img = Image.fromarray(rotated_img)
                            text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
                            
                            # Extract digits only
                            digits = re.sub(r'\D', '', text)
                            
                            logger.debug(f"    Method: {method_name}, Rotation: {rotation}°, Config: {config[:20]}, Result: {digits}")
                            
                            # Keep track of best result (most digits found)
                            if len(digits) > best_digit_count:
                                best_digit_count = len(digits)
                                best_result = digits
                                logger.info(f"    Better result found: {digits} (method: {method_name}, rotation: {rotation}°)")
                            
                            # If we got 12 digits, we're done!
                            if len(digits) == 12:
                                logger.info(f"    Perfect Aadhaar found: {digits} (method: {method_name}, rotation: {rotation}°)")
                                return digits
                                
                        except Exception as e:
                            logger.debug(f"    OCR attempt failed: {e}")
                            continue
            
            logger.info(f"    Best Aadhaar result: {best_result} ({best_digit_count} digits)")
            return best_result
            
        except Exception as e:
            logger.error(f"  Error in Aadhaar extraction: {e}")
            return ""
    
    def _correct_entity_orientation_and_preprocess(self, entity_image: np.ndarray, entity_key: str, class_name: str = None, osd_confidence_threshold: float = 0.5) -> Optional[Any]:
        """
        Enhanced preprocessing with special handling for Aadhaar numbers.
        """
        try:
            img = entity_image
            if img is None or img.size == 0:
                logger.warning(f"  Entity image data for {entity_key} is empty, skipping.")
                return None
            
            # Special handling for Aadhaar numbers - skip complex orientation detection
            if class_name == 'aadharnumber':
                logger.info(f"  Using specialized Aadhaar preprocessing for {entity_key}")
                # Return original image for specialized Aadhaar processing
                return img
            
            h, w = img.shape[:2]
            if h < 100:
                scale_factor = 100 / h
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img_for_analysis = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                img_for_analysis = img

            best_rotation = self._detect_orientation_by_letters(img_for_analysis, entity_key)
            
            if best_rotation is None:
                try:
                    osd = pytesseract.image_to_osd(img_for_analysis, output_type=pytesseract.Output.DICT)
                    if osd['orientation_conf'] > osd_confidence_threshold:
                        best_rotation = osd['rotate']
                        logger.info(f" Using Tesseract OSD for {entity_key}: {best_rotation}° (conf: {osd['orientation_conf']:.2f})")
                    else:
                        best_rotation = 0
                except pytesseract.TesseractError as e:
                    logger.warning(f" OSD failed for {entity_key}. Assuming 0° rotation.")
                    best_rotation = 0
            
            corrected_img = img
            if best_rotation != 0:
                logger.info(f"   Correcting entity {entity_key} orientation by {best_rotation}°")
                if best_rotation == 90: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif best_rotation == 180: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_180)
                elif best_rotation == 270: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            h_corr, w_corr = corrected_img.shape[:2]
            if h_corr > w_corr and 'address' not in entity_key:
                logger.info(f"   Rotating vertical entity {entity_key} to horizontal format")
                corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_90_CLOCKWISE)

            from PIL import Image
            return Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
            
        except Exception as e:
            logger.error(f"   Unhandled error during entity orientation/preprocessing for {entity_key}: {e}")
            return None

    def _detect_orientation_by_letters(self, img: np.ndarray, entity_key: str) -> Optional[int]:
        """
        Detect the correct orientation by analyzing letter shapes and OCR confidence
        at different rotation angles.
        """
        try:
            rotations = [0, 90, 180, 270]
            rotation_scores = {}
            
            for rotation in rotations:
                if rotation == 0:
                    rotated_img = img
                elif rotation == 90:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == 180:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                elif rotation == 270:
                    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
                score = self._calculate_orientation_score(rotated_img, rotation)
                rotation_scores[rotation] = score
                logger.debug(f"      Rotation {rotation}°: score = {score:.3f}")
            
            best_rotation = max(rotation_scores.keys(), key=lambda k: rotation_scores[k])
            best_score = rotation_scores[best_rotation]
            
            if best_score > 0.1:
                logger.info(f" Letter-based analysis for {entity_key}: {best_rotation}° (score: {best_score:.3f})")
                return best_rotation
            else:
                logger.warning(f"   Letter-based analysis inconclusive for {entity_key} (best score: {best_score:.3f})")
                return None
                
        except Exception as e:
            logger.warning(f"  Error in letter-based orientation detection for {entity_key}: {e}")
            return None

    def _calculate_orientation_score(self, img: np.ndarray, rotation: int) -> float:
        """
        Calculate a comprehensive score for how likely this orientation is correct.
        Combines OCR confidence, letter shapes, and text line analysis.
        """
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
                
            ocr_score = self._get_ocr_confidence_score(gray)
            shape_score = self._analyze_letter_shapes(gray)
            line_score = self._analyze_text_lines(gray)
            
            total_score = (ocr_score * 0.5 + shape_score * 0.3 + line_score * 0.2)
            return total_score
            
        except Exception as e:
            logger.debug(f"      Error calculating orientation score: {e}")
            return 0.0

    def _get_ocr_confidence_score(self, gray_img: np.ndarray) -> float:
        """
        Get OCR confidence and text quality score by trying multiple PSM modes.
        Returns a normalized score between 0 and 1.
        """
        try:
            psm_modes = [6, 7, 8, 13]
            best_confidence = 0.0
            best_text_length = 0
            
            for psm in psm_modes:
                try:
                    data = pytesseract.image_to_data(gray_img, config=f'--psm {psm}', output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        text_length = sum(len(text.strip()) for text in data['text'] if text.strip())
                        
                        if avg_confidence > best_confidence or (avg_confidence == best_confidence and text_length > best_text_length):
                            best_confidence = avg_confidence
                            best_text_length = text_length
                            
                except pytesseract.TesseractError:
                    continue
            
            confidence_factor = best_confidence / 100.0
            length_factor = min(best_text_length / 10.0, 1.0)
            return confidence_factor * 0.7 + length_factor * 0.3
            
        except Exception:
            return 0.0

    def _analyze_letter_shapes(self, gray_img: np.ndarray) -> float:
        """
        Analyze the shapes of detected contours to determine if they look like upright letters.
        Returns a normalized score between 0 and 1.
        """
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            
            upright_score = 0.0
            valid_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 5 or h < 5 or w > gray_img.shape[1] * 0.8 or h > gray_img.shape[0] * 0.8:
                    continue
                
                aspect_ratio = h / w
                if 0.3 <= aspect_ratio <= 4.0:
                    valid_contours += 1
                    if 1.0 <= aspect_ratio <= 2.5:
                        upright_score += 1.0
                    elif 0.5 <= aspect_ratio <= 3.5:
                        upright_score += 0.7
                    else:
                        upright_score += 0.3
            
            if valid_contours == 0:
                return 0.0
            return min(upright_score / valid_contours, 1.0)
            
        except Exception:
            return 0.0

    def _analyze_text_lines(self, gray_img: np.ndarray) -> float:
        """
        Analyze text line orientation using morphological operations.
        Horizontal text should have more horizontal lines than vertical lines.
        Returns a normalized score between 0 and 1.
        """
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            horizontal_pixels = cv2.countNonZero(horizontal_lines)
            vertical_pixels = cv2.countNonZero(vertical_lines)
            
            total_pixels = horizontal_pixels + vertical_pixels
            if total_pixels == 0:
                return 0.5
            
            horizontal_ratio = horizontal_pixels / total_pixels
            return horizontal_ratio
            
        except Exception:
            return 0.5

    def perform_multi_language_ocr(self, all_detections: Dict[str, Dict[str, Any]]):
        """
        Step 4: Correcting orientation and perform OCR with specialized handling for Aadhaar.
        """
        logger.info(f"\nStep 4: Correcting Entity Orientation & Performing Multi-Language OCR")
        ocr_results = {}
        for card_name, card_data in all_detections.items():
            for detection in card_data['detections']:
                cropped_image = detection.get('cropped_image')
                entity_key = detection.get('entity_key')
                class_name = detection.get('class_name')

                if cropped_image is None or entity_key is None:
                    continue

                logger.info(f"  Processing entity: {entity_key} (Class: {class_name})")
                
                # Special handling for Aadhaar numbers
                if class_name == 'aadharnumber':
                    extracted_text = self._extract_aadhaar_with_multiple_methods(cropped_image)
                    ocr_results[entity_key] = extracted_text
                    if extracted_text:
                        logger.info(f"    Aadhaar OCR Result: {extracted_text}")
                    else:
                        logger.warning(f"    Aadhaar OCR failed to extract number")
                    continue
                
                # Regular processing for other entities
                lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                
                processed_pil_img = self._correct_entity_orientation_and_preprocess(cropped_image, entity_key, class_name)

                if processed_pil_img:
                    try:
                        text = pytesseract.image_to_string(processed_pil_img, lang=lang_to_use, config='--psm 6')
                        extracted_text = ' '.join(text.split()).strip()
                        ocr_results[entity_key] = extracted_text
                        logger.info(f"    OCR Result: {extracted_text[:50]}..." if len(extracted_text) > 50 else f"    OCR Result: {extracted_text}")
                    except Exception as e:
                        logger.error(f" OCR failed for {entity_key}: {e}")
                        ocr_results[entity_key] = None
        return ocr_results

    def organize_results_by_card_type(self, all_detections: Dict[str, Dict[str, Any]], ocr_results: Dict[str, str], confidence_threshold: float):
        """Your EXACT original result organization"""
        logger.info("\nStep 5: Organizing final results")
        organized_results = {
            'front': {}, 'back': {},
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'confidence_threshold_used': confidence_threshold
            }
        }
        for card_name, card_data in all_detections.items():
            card_type = card_data['card_type']
            
            if card_name not in organized_results[card_type]:
                organized_results[card_type][card_name] = {'entities': {}}
            
            for detection in card_data['detections']:
                 entity_name = detection['class_name']
                 entity_key = detection.get('entity_key')
                 extracted_text = ocr_results.get(entity_key)

                 if entity_name not in organized_results[card_type][card_name]['entities']:
                      organized_results[card_type][card_name]['entities'][entity_name] = []
                 
                 organized_results[card_type][card_name]['entities'][entity_name].append({
                     'confidence': detection['confidence'], 
                     'bbox': detection['bbox'],
                     'extracted_text': extracted_text
                 })
        return organized_results

    def extract_main_fields(self, organized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Your EXACT original Validation Logic (Moved inside class)"""
        fields = ['aadharnumber', 'dob', 'gender']
        data = {key: "" for key in fields}
        
        # Special handling for Aadhar number to check both front and back
        aadhar_front = ""
        aadhar_back = ""
        
        for side in ['front', 'back']:
            for card in organized_results.get(side, {}).values():
                for field in fields:
                    if field in card['entities'] and card['entities'][field]:
                        all_texts = [item.get('extracted_text', '') for item in card['entities'][field]]
                        first_valid_text = next((text for text in all_texts if text), '')
                        
                        if first_valid_text:
                            if field == 'aadharnumber':
                                if side == 'front':
                                    aadhar_front = first_valid_text
                                elif side == 'back':
                                    aadhar_back = first_valid_text
                            else:
                                data[field] = first_valid_text
        
        # --- Special Logic for Aadhar Number from Front & Back ---
        best_aadhar = ""
        aadhar_front_digits = re.sub(r'\D', '', aadhar_front) if aadhar_front else ""
        aadhar_back_digits = re.sub(r'\D', '', aadhar_back) if aadhar_back else ""
        
        logger.info(f"Aadhar Front: '{aadhar_front}' -> Digits: '{aadhar_front_digits}'")
        logger.info(f"Aadhar Back: '{aadhar_back}' -> Digits: '{aadhar_back_digits}'")
        
        if aadhar_front_digits == aadhar_back_digits and aadhar_front_digits:
            # Both match - use this
            logger.info(f"Aadhar match between front and back: {aadhar_front_digits}")
            best_aadhar = aadhar_front_digits
        else:
            # Different values - prefer the one with all 12 digits, or the one with more digits
            if len(aadhar_front_digits) == 12:
                logger.info(f"Using Aadhar from front (complete 12 digits): {aadhar_front_digits}")
                best_aadhar = aadhar_front_digits
            elif len(aadhar_back_digits) == 12:
                logger.info(f"Using Aadhar from back (complete 12 digits): {aadhar_back_digits}")
                best_aadhar = aadhar_back_digits
            elif len(aadhar_front_digits) > len(aadhar_back_digits):
                logger.info(f"Using Aadhar from front (more digits): {aadhar_front_digits}")
                best_aadhar = aadhar_front_digits
            elif len(aadhar_back_digits) > len(aadhar_front_digits):
                logger.info(f"Using Aadhar from back (more digits): {aadhar_back_digits}")
                best_aadhar = aadhar_back_digits
            elif aadhar_front_digits:
                logger.info(f"Using Aadhar from front (fallback): {aadhar_front_digits}")
                best_aadhar = aadhar_front_digits
            elif aadhar_back_digits:
                logger.info(f"Using Aadhar from back (fallback): {aadhar_back_digits}")
                best_aadhar = aadhar_back_digits
        
        data['aadharnumber'] = best_aadhar
        
        # --- Aadhaar Number Validation ---
        aadhar_status = "aadhar_approved"
        if data.get('aadharnumber'):
            aad = data['aadharnumber']
            # Extract only digits, remove all spaces and special characters
            aad_digits_only = re.sub(r'\D', '', aad)  # Remove all non-digit characters
            
            # Check for masked Aadhaar (with X's) in original text
            if (aadhar_front and re.search(r'X{4}', aadhar_front, re.IGNORECASE)) or \
               (aadhar_back and re.search(r'X{4}', aadhar_back, re.IGNORECASE)):
                aadhar_status = "aadhar_disapproved"
            # Validate that we have exactly 12 digits
            elif len(aad_digits_only) == 12:
                data['aadharnumber'] = aad_digits_only
            else:
                # If not exactly 12 digits, disapprove
                aadhar_status = "aadhar_disapproved"
                data['aadharnumber'] = aad_digits_only  # Store what we got anyway
        else:
            aadhar_status = "aadhar_disapproved"
        data['aadhar_status'] = aadhar_status
        
        # --- DOB Processing and Age Verification (Year-based only) ---
        age_status = "age_disapproved"
        birth_year = None
        
        if data.get('dob'):
            # Extract all digit groups from DOB text
            digit_groups = re.findall(r'\d+', data['dob'])
            
            # Look for a 4-digit year
            year = next((g for g in digit_groups if len(g) == 4 and 1900 <= int(g) <= datetime.now().year), None)
            
            if year:
                try:
                    birth_year = int(year)
                    data['dob'] = str(birth_year)  # Store only the year
                    
                    # Calculate age based on year only
                    current_year = datetime.now().year
                    age = current_year - birth_year
                    data['age'] = age
                    
                    # Approve if age >= 18
                    if age >= 18:
                        age_status = "age_approved"
                        logger.info(f"Age approved: {age} years old (born in {birth_year})")
                    else:
                        logger.info(f"Age rejected: {age} years old (born in {birth_year})")
                except ValueError:
                    data['dob'] = 'Invalid Format'
                    data['age'] = None
                    logger.warning(f"Could not parse year from DOB: {data.get('dob')}")
            else:
                data['dob'] = 'Invalid Format'
                data['age'] = None
                logger.warning(f"No valid 4-digit year found in DOB: {data.get('dob')}")
        else:
            data['dob'] = 'Not Detected'
            data['age'] = None
            logger.info("DOB not detected")
        
        data['age_status'] = age_status
        
        # --- Gender normalization (Enhanced) ---
        if data['gender']:
            gender = data['gender'].strip().lower()
            # Remove special characters and extra spaces
            gender = re.sub(r'[^a-z]', '', gender)
            
            # Check for male variations
            if 'male' in gender and 'female' not in gender:
                data['gender'] = 'Male'
            # Check for female variations
            elif 'female' in gender or 'femal' in gender:
                data['gender'] = 'Female'
            # Check for other common variations
            elif gender in ['m', 'man', 'boy']:
                data['gender'] = 'Male'
            elif gender in ['f', 'woman', 'girl', 'femlae', 'femaie']:
                data['gender'] = 'Female'
            else:
                data['gender'] = 'Other'
        else:
            data['gender'] = 'Not Detected'
            
        return data

# Usage Check
if __name__ == "__main__":
    try:
        agent = EntityAgent(model_path="models/best.pt")
        print("✓ Entity Agent initialized successfully")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")

# import logging
# import math 
# import os
# import re
# import sys
# import traceback
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import cv2
# import numpy as np
# import pytesseract
# import torch
# from dotenv import load_dotenv
# from ultralytics import YOLO

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class EntityAgent:
#     def __init__(self, model_path="models/best.pt", other_lang_code='hin+tel+ben', save_debug_images=True):

#         if torch.cuda.is_available():
#             self.device = "cuda"
#             logger.info(f"CUDA is available. Models will run on GPU ({torch.cuda.get_device_name(0)}).")
#         else:
#             self.device = "cpu"
#             logger.info("CUDA not available. Models will fall back to CPU.")

#         # --- Model Loading ---
#         self.entity_model_path = model_path
#         self.save_debug_images = save_debug_images
        
#         # Create debug output directories
#         if self.save_debug_images:
#             self.debug_base_dir = Path("debug_output")
#             self.debug_base_dir.mkdir(exist_ok=True)
#             logger.info(f"Debug images will be saved to: {self.debug_base_dir}")
        
#         logger.info("Checking for entity detection YOLO model on filesystem...")
#         if not Path(self.entity_model_path).exists():
#             logger.critical(f"Entity model not found at {self.entity_model_path}. Aborting startup.")
#             raise FileNotFoundError(f"Entity model not found at {self.entity_model_path}")

#         logger.info("Loading entity detection model from filesystem...")
#         self.model2 = YOLO(self.entity_model_path)
        
#         self.other_lang_code = other_lang_code
#         self._check_tesseract()

#         logger.info("YOLOv8 entity detection model loaded successfully.")
#         logger.info(f"EntityAgent initialized to use '{self.other_lang_code}' for other language fields.")
        
#         self.entity_classes = {
#             0: 'aadharnumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
#             4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
#             8: 'name', 9: 'name_otherlang', 10: 'pincode'
#         }

#     def _check_tesseract(self):
#         try:
#             pytesseract.get_tesseract_version()
#         except pytesseract.TesseractNotFoundError:
#             logger.critical("Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
#             raise RuntimeError("Tesseract not found")
#         except Exception as e:
#              logger.critical(f"An error occurred while checking Tesseract: {e}")
#              raise RuntimeError(f"Error checking Tesseract: {e}")
    
#     def _visualize_all_detections_on_original(self, img: np.ndarray, confidence_threshold: float, card_side: str, session_dir: Path):
#         """
#         Visualize ALL detections from the model on the original image (before cropping).
#         This helps debug if the issue is with detection or cropping.
#         """
#         try:
#             logger.info(f"  Visualizing ALL detections on original {card_side} image...")
            
#             # Run detection on original image
#             results = self.model2(img, device=self.device, verbose=False)
            
#             # Create visualization image
#             viz_img = img.copy()
            
#             # Define color map for different entity types
#             color_map = {
#                 'aadharnumber': (0, 255, 0),      # Green
#                 'dob': (255, 0, 0),                # Blue
#                 'gender': (0, 255, 255),           # Yellow
#                 'name': (255, 0, 255),             # Magenta
#                 'name_otherlang': (255, 128, 0),  # Orange
#                 'address': (128, 255, 0),          # Lime
#                 'address_other_lang': (0, 128, 255), # Sky blue
#                 'pincode': (255, 0, 128),          # Pink
#                 'mobile_no': (128, 0, 255),        # Purple
#                 'city': (0, 255, 128),             # Spring green
#                 'gender_other_lang': (128, 255, 255) # Light cyan
#             }
            
#             detection_count = 0
#             detection_summary = []
            
#             for box in results[0].boxes:
#                 conf = float(box.conf[0])
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 class_id = int(box.cls[0])
#                 class_name = self.entity_classes.get(class_id, "unknown")
                
#                 # Get color for this class
#                 color = color_map.get(class_name, (200, 200, 200))
                
#                 # Determine if this meets threshold
#                 meets_threshold = conf >= confidence_threshold
#                 line_thickness = 3 if meets_threshold else 1
                
#                 # Draw bounding box
#                 cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, line_thickness)
                
#                 # Add label with confidence
#                 label = f"{class_name}: {conf:.2f}"
#                 if not meets_threshold:
#                     label += " (LOW)"
                
#                 # Calculate label background size
#                 font_scale = 0.6
#                 font_thickness = 2
#                 (label_width, label_height), baseline = cv2.getTextSize(
#                     label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
#                 )
                
#                 # Draw label background
#                 cv2.rectangle(viz_img, 
#                             (x1, y1 - label_height - baseline - 5),
#                             (x1 + label_width, y1),
#                             color, -1)
                
#                 # Draw label text
#                 cv2.putText(viz_img, label, (x1, y1 - 5),
#                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                
#                 detection_count += 1
#                 detection_summary.append(f"  - {class_name}: {conf:.3f} {'✓' if meets_threshold else '✗'}")
            
#             # Add summary text on image
#             summary_text = f"Total Detections: {detection_count} | Threshold: {confidence_threshold}"
#             cv2.putText(viz_img, summary_text, (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
#             cv2.putText(viz_img, summary_text, (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
#             # Save visualization
#             viz_path = session_dir / f"00a_ORIGINAL_all_detections_{card_side}.jpg"
#             cv2.imwrite(str(viz_path), viz_img)
#             logger.info(f"  Saved ALL detections visualization: {viz_path}")
#             logger.info(f"  Detection summary on original image:")
#             for summary_line in detection_summary:
#                 logger.info(summary_line)
            
#             # Create a legend image
#             self._create_detection_legend(session_dir, color_map, card_side)
            
#         except Exception as e:
#             logger.error(f"  Error visualizing original detections: {e}\n{traceback.format_exc()}")
    
#     def _create_detection_legend(self, session_dir: Path, color_map: dict, card_side: str):
#         """Create a color legend for the detection visualization."""
#         try:
#             # Create a white image for legend
#             legend_height = 40 + len(color_map) * 30
#             legend_width = 400
#             legend_img = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
            
#             # Add title
#             cv2.putText(legend_img, "Detection Legend", (10, 25),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
#             # Add each entity type with its color
#             y_offset = 50
#             for entity_name, color in sorted(color_map.items()):
#                 # Draw color box
#                 cv2.rectangle(legend_img, (10, y_offset - 15), (40, y_offset), color, -1)
#                 cv2.rectangle(legend_img, (10, y_offset - 15), (40, y_offset), (0, 0, 0), 1)
                
#                 # Draw entity name
#                 cv2.putText(legend_img, entity_name, (50, y_offset - 2),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
#                 y_offset += 30
            
#             # Save legend
#             legend_path = session_dir / f"00a_legend_{card_side}.jpg"
#             cv2.imwrite(str(legend_path), legend_img)
#             logger.info(f"  Saved detection legend: {legend_path}")
            
#         except Exception as e:
#             logger.warning(f"  Could not create legend: {e}")

#     def _check_critical_fields(self, final_data: Dict[str, Any], card_side: str) -> Dict[str, bool]:
#         """
#         Check if critical fields are present and valid.
#         Returns dict with field name and whether it's missing/invalid.
#         """
#         if card_side.lower() == 'front':
#             critical_fields = ['aadharnumber', 'dob', 'gender']
#         else:  # back
#             critical_fields = ['aadharnumber']
        
#         missing = {}
#         for field in critical_fields:
#             value = final_data.get(field, "")
            
#             if field == 'aadharnumber':
#                 # Check if valid 12-digit number
#                 digits = re.sub(r'\D', '', str(value))
#                 missing[field] = len(digits) != 12
#             elif field == 'dob':
#                 # Check if not "Invalid Format" or "Not Detected"
#                 missing[field] = value in ['Invalid Format', 'Not Detected', '', None]
#             elif field == 'gender':
#                 # Check if not "Not Detected" or "Other"
#                 missing[field] = value in ['Not Detected', 'Other', '', None]
#             else:
#                 missing[field] = not value
        
#         return missing

#     def _merge_results(self, cropped_data: Dict[str, Any], full_data: Dict[str, Any], 
#                        missing_fields: Dict[str, bool]) -> Dict[str, Any]:
#         """
#         Merge results from cropped and full image detections.
#         Prefer full image results for missing/invalid fields.
#         """
#         merged = cropped_data.copy()
        
#         logger.info("🔀 Merging results from cropped and full image detections...")
        
#         for field, is_missing in missing_fields.items():
#             if is_missing:
#                 full_value = full_data.get(field)
#                 cropped_value = cropped_data.get(field)
                
#                 # For aadharnumber, check if full version is better
#                 if field == 'aadharnumber':
#                     full_digits = re.sub(r'\D', '', str(full_value)) if full_value else ""
#                     cropped_digits = re.sub(r'\D', '', str(cropped_value)) if cropped_value else ""
                    
#                     if len(full_digits) == 12 and len(cropped_digits) != 12:
#                         logger.info(f"  ✓ Using full image {field}: {full_digits}")
#                         merged[field] = full_digits
#                         merged['aadhar_status'] = full_data.get('aadhar_status', 'aadhar_approved')
#                     elif len(full_digits) > len(cropped_digits):
#                         logger.info(f"  ✓ Using full image {field} (more digits): {full_digits}")
#                         merged[field] = full_digits
                
#                 # For other fields, use full if cropped is invalid
#                 elif full_value and full_value not in ['Invalid Format', 'Not Detected', 'Other', '']:
#                     logger.info(f"  ✓ Using full image {field}: {full_value}")
#                     merged[field] = full_value
                    
#                     # Update related status fields
#                     if field == 'dob':
#                         merged['age'] = full_data.get('age')
#                         merged['age_status'] = full_data.get('age_status')
        
#         return merged

#     def extract_from_file(self, file_path: str, crop_coords: List[int] = None, 
#                          confidence_threshold: float = 0.15, card_side: str = 'front'):
#         """
#         Main Entry Point with FALLBACK mechanism.
#         1. Try extraction with crop_coords if provided
#         2. Check if critical fields are missing
#         3. If missing, retry on FULL original image
#         4. Merge best results
#         """
#         try:
#             # Create session-specific debug directory
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             session_dir = None
#             if self.save_debug_images:
#                 session_dir = self.debug_base_dir / f"session_{timestamp}_{card_side}"
#                 session_dir.mkdir(exist_ok=True)
#                 logger.info(f"Session debug directory: {session_dir}")
            
#             # 1. Read ORIGINAL Image
#             original_img = cv2.imread(file_path)
#             if original_img is None:
#                 logger.error(f"Failed to read image: {file_path}")
#                 return {"error": "failed_to_read_file"}

#             # Save ORIGINAL image with ALL detections (before any cropping)
#             if self.save_debug_images and session_dir:
#                 logger.info(f"📊 Visualizing ALL detections on ORIGINAL image...")
#                 self._visualize_all_detections_on_original(original_img.copy(), confidence_threshold, card_side, session_dir)

#             # === ATTEMPT 1: Try with Crop Coordinates ===
#             logger.info("=" * 60)
#             logger.info("🔍 ATTEMPT 1: Extraction with CROP coordinates")
#             logger.info("=" * 60)
            
#             img = original_img.copy()
#             attempt_name = "cropped"
            
#             # Apply crop if coords provided
#             if crop_coords:
#                 x1, y1, x2, y2 = crop_coords
#                 h, w = img.shape[:2]
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
                
#                 logger.info(f"Cropping image to: {crop_coords}")
                
#                 # Save visualization of crop region
#                 if self.save_debug_images and session_dir:
#                     img_with_crop_box = original_img.copy()
#                     cv2.rectangle(img_with_crop_box, (x1, y1), (x2, y2), (255, 0, 0), 3)
#                     cv2.putText(img_with_crop_box, f"Crop Region: {card_side}", (x1, y1-10),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#                     crop_box_path = session_dir / f"00b_crop_region_{card_side}.jpg"
#                     cv2.imwrite(str(crop_box_path), img_with_crop_box)
                
#                 img = img[y1:y2, x1:x2]
                
#                 if self.save_debug_images and session_dir:
#                     input_path = session_dir / f"00c_cropped_input_{card_side}.jpg"
#                     cv2.imwrite(str(input_path), img)
#             else:
#                 logger.info("No crop coordinates provided, using full image")
#                 attempt_name = "full"

#             # Run detection and OCR on cropped image
#             all_detections = self.detect_entities_in_image(img, confidence_threshold, card_side, session_dir)
            
#             if not all_detections:
#                 logger.warning("⚠️  No entities detected in cropped region")
#                 cropped_result = {
#                     "success": False,
#                     "message": "no_entities_detected",
#                     "data": {"aadharnumber": "", "dob": "Not Detected", "gender": "Not Detected"}
#                 }
#             else:
#                 self.crop_entities(all_detections, session_dir)
#                 ocr_results = self.perform_multi_language_ocr(all_detections)
#                 organized = self.organize_results_by_card_type(all_detections, ocr_results, confidence_threshold)
#                 cropped_data = self.extract_main_fields(organized)
                
#                 cropped_result = {
#                     "success": True,
#                     "data": cropped_data
#                 }
            
#             # === CHECK: Are critical fields missing? ===
#             logger.info("\n" + "=" * 60)
#             logger.info("🔍 Checking for missing critical fields...")
#             logger.info("=" * 60)
            
#             missing_fields = self._check_critical_fields(cropped_result.get("data", {}), card_side)
#             has_missing = any(missing_fields.values())
            
#             if has_missing:
#                 logger.warning("⚠️  CRITICAL FIELDS MISSING from cropped extraction:")
#                 for field, is_missing in missing_fields.items():
#                     if is_missing:
#                         logger.warning(f"  ❌ {field}: Missing or Invalid")
                
#                 # === ATTEMPT 2: FALLBACK to Full Image ===
#                 logger.info("\n" + "=" * 60)
#                 logger.info("🔄 ATTEMPT 2: FALLBACK - Running on FULL ORIGINAL image")
#                 logger.info("=" * 60)
                
#                 # Run detection on FULL original image
#                 full_detections = self.detect_entities_in_image(
#                     original_img, confidence_threshold, card_side, session_dir
#                 )
                
#                 if full_detections:
#                     # Create subfolder for full image results
#                     if session_dir:
#                         full_crops_dir = session_dir / "crops_fullimage"
#                         full_crops_dir.mkdir(exist_ok=True)
#                     else:
#                         full_crops_dir = None
                    
#                     self.crop_entities(full_detections, full_crops_dir)
#                     full_ocr_results = self.perform_multi_language_ocr(full_detections)
#                     full_organized = self.organize_results_by_card_type(
#                         full_detections, full_ocr_results, confidence_threshold
#                     )
#                     full_data = self.extract_main_fields(full_organized)
                    
#                     # Merge results
#                     logger.info("\n" + "=" * 60)
#                     logger.info("🔀 MERGING results from both attempts...")
#                     logger.info("=" * 60)
                    
#                     final_data = self._merge_results(
#                         cropped_result.get("data", {}), 
#                         full_data, 
#                         missing_fields
#                     )
                    
#                     logger.info("\n✅ Final merged data:")
#                     for key in ['aadharnumber', 'dob', 'gender', 'age', 'age_status', 'aadhar_status']:
#                         logger.info(f"  {key}: {final_data.get(key)}")
                    
#                     return {
#                         "success": True,
#                         "data": final_data,
#                         "debug_dir": str(session_dir) if session_dir else None,
#                         "used_fallback": True
#                     }
#                 else:
#                     logger.error("❌ Fallback detection on full image also failed")
#                     return cropped_result
#             else:
#                 logger.info("✅ All critical fields present in cropped extraction")
#                 return {
#                     "success": True,
#                     "data": cropped_result.get("data", {}),
#                     "debug_dir": str(session_dir) if session_dir else None,
#                     "used_fallback": False
#                 }

#         except Exception as e:
#             logger.error(f"Error in extraction: {e}\n{traceback.format_exc()}")
#             return {"success": False, "error": str(e)}

#     def detect_entities_in_image(self, image_input, confidence_threshold: float, card_side: str = 'front', session_dir: Path = None):
#         """
#         Modified to accept card_side parameter and filter entities accordingly.
#         Also saves annotated detection images.
#         """
#         logger.info(f"\nStep 1: Detecting entities in image (Side: {card_side}, Threshold: {confidence_threshold})")
        
#         # Handle input type
#         if isinstance(image_input, str):
#             img = cv2.imread(str(image_input))
#             input_name = Path(image_input).stem
#         elif isinstance(image_input, np.ndarray):
#             img = image_input
#             input_name = "memory_crop"
#         else:
#             return {}

#         if img is None:
#             return {}
        
#         # Define entities based on card side
#         if card_side.lower() == 'front':
#             target_entities = {'aadharnumber', 'dob', 'gender', 'name', 'name_otherlang'}
#         elif card_side.lower() == 'back':
#             target_entities = {'aadharnumber', 'address', 'address_other_lang', 'pincode', 'mobile_no', 'city'}
#         else:
#             # Default to basic entities if side not specified
#             target_entities = {'aadharnumber', 'dob', 'gender'}
        
#         logger.info(f"  Target entities for {card_side}: {target_entities}")
        
#         # Run entity detection
#         results = self.model2(img, device=self.device, verbose=False)
#         card_detections = []
        
#         # Create annotated image
#         annotated_img = img.copy()
        
#         for box in results[0].boxes:
#             if float(box.conf[0]) < confidence_threshold: 
#                 continue
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
            
#             # Only process target entities
#             if class_name not in target_entities:
#                 continue
            
#             card_detections.append({
#                 'bbox': (x1, y1, x2, y2), 
#                 'class_name': class_name, 
#                 'confidence': float(box.conf[0])
#             })
            
#             # Draw bounding box on annotated image
#             color = (0, 255, 0)  # Green
#             cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
#             # Add label with confidence
#             label = f"{class_name}: {float(box.conf[0]):.2f}"
#             label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 5), 
#                          (x1 + label_size[0], y1), color, -1)
#             cv2.putText(annotated_img, label, (x1, y1 - 5), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
#         logger.info(f"  Detected {len(card_detections)} entities")
        
#         # Save annotated image
#         if self.save_debug_images and session_dir:
#             annotated_path = session_dir / f"01_detections_{card_side}_annotated.jpg"
#             cv2.imwrite(str(annotated_path), annotated_img)
#             logger.info(f"Saved annotated detections: {annotated_path}")
        
#         if not card_detections:
#             return {}

#         # Wrap in your original structure
#         all_detections = {
#             input_name: {
#                 "card_image": img,
#                 "card_type": card_side,
#                 "detections": card_detections
#             }
#         }
        
#         return all_detections

#     def crop_entities(self, all_detections: Dict[str, Dict[str, Any]], session_dir: Path = None):
#         """Step 3: Crop individual entities with bounds checking and save crops"""
#         logger.info(f"\nStep 3: Cropping individual entities")
        
#         # Create crops directory
#         crops_dir = None
#         if self.save_debug_images and session_dir:
#             crops_dir = session_dir if isinstance(session_dir, Path) and session_dir.name.startswith("crops") else session_dir / "crops"
#             crops_dir.mkdir(exist_ok=True)
        
#         for card_name, card_data in all_detections.items():
#             img = card_data['card_image']
#             h, w = img.shape[:2]
#             card_type = card_data.get('card_type', 'unknown')
            
#             for i, detection in enumerate(card_data['detections']):
#                 x1, y1, x2, y2 = detection['bbox']
                
#                 # Sanitize bounds
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
                
#                 if x2 <= x1 or y2 <= y1:
#                     logger.warning(f"  Invalid bbox for {detection['class_name']}, skipping")
#                     detection['cropped_image'] = None
#                     continue
                
#                 crop = img[y1:y2, x1:x2]
#                 detection['cropped_image'] = crop
#                 entity_key = f"{card_name}_{detection['class_name']}_{i}"
#                 detection['entity_key'] = entity_key
                
#                 # Save individual crop
#                 if self.save_debug_images and crops_dir:
#                     class_name = detection['class_name']
#                     conf = detection['confidence']
#                     crop_filename = f"{card_type}_{class_name}_{i}_conf{conf:.2f}.jpg"
#                     crop_path = crops_dir / crop_filename
#                     cv2.imwrite(str(crop_path), crop)
#                     logger.info(f"  Saved crop: {crop_filename}")
        
#         return all_detections
    
#     def _preprocess_for_aadhaar_ocr(self, img: np.ndarray) -> np.ndarray:
#         """
#         Specialized preprocessing for Aadhaar numbers which are often in specific fonts/formats.
#         """
#         try:
#             # Convert to grayscale if needed
#             if len(img.shape) == 3:
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray = img.copy()
            
#             # Resize if too small (Aadhaar numbers need good resolution)
#             h, w = gray.shape
#             if h < 50:
#                 scale = 50 / h
#                 new_w, new_h = int(w * scale), int(h * scale)
#                 gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
#             # Try multiple preprocessing approaches and pick best
#             preprocessed_versions = []
            
#             # Version 1: Simple threshold
#             _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             preprocessed_versions.append(('otsu', thresh1))
            
#             # Version 2: Adaptive threshold
#             thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                            cv2.THRESH_BINARY, 11, 2)
#             preprocessed_versions.append(('adaptive', thresh2))
            
#             # Version 3: Enhanced contrast + threshold
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#             enhanced = clahe.apply(gray)
#             _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             preprocessed_versions.append(('clahe', thresh3))
            
#             # Version 4: Denoising + threshold
#             denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
#             _, thresh4 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             preprocessed_versions.append(('denoised', thresh4))
            
#             return preprocessed_versions
            
#         except Exception as e:
#             logger.warning(f"  Error in Aadhaar preprocessing: {e}")
#             return [('original', img)]
    
#     def _extract_aadhaar_with_multiple_methods(self, img: np.ndarray) -> str:
#         """
#         Try multiple OCR methods specifically for Aadhaar numbers.
#         """
#         try:
#             preprocessed_versions = self._preprocess_for_aadhaar_ocr(img)
            
#             # Try different PSM modes and configurations
#             configs = [
#                 '--psm 7 -c tessedit_char_whitelist=0123456789',  # Single line, digits only
#                 '--psm 8 -c tessedit_char_whitelist=0123456789',  # Single word, digits only
#                 '--psm 6 -c tessedit_char_whitelist=0123456789',  # Block of text, digits only
#                 '--psm 7',  # Single line, all chars
#                 '--psm 13',  # Raw line
#             ]
            
#             best_result = ""
#             best_digit_count = 0
            
#             for method_name, processed_img in preprocessed_versions:
#                 for config in configs:
#                     try:
#                         from PIL import Image
#                         pil_img = Image.fromarray(processed_img)
#                         text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
                        
#                         # Extract digits only
#                         digits = re.sub(r'\D', '', text)
                        
#                         logger.debug(f"    Method: {method_name}, Config: {config[:20]}, Result: {digits}")
                        
#                         # Keep track of best result (most digits found)
#                         if len(digits) > best_digit_count:
#                             best_digit_count = len(digits)
#                             best_result = digits
#                             logger.info(f"    Better result found: {digits} (method: {method_name})")
                        
#                         # If we got 12 digits, we're done!
#                         if len(digits) == 12:
#                             logger.info(f"    Perfect Aadhaar found: {digits}")
#                             return digits
                            
#                     except Exception as e:
#                         logger.debug(f"    OCR attempt failed: {e}")
#                         continue
            
#             logger.info(f"    Best Aadhaar result: {best_result} ({best_digit_count} digits)")
#             return best_result
            
#         except Exception as e:
#             logger.error(f"  Error in Aadhaar extraction: {e}")
#             return ""
    
#     def _correct_entity_orientation_and_preprocess(self, entity_image: np.ndarray, entity_key: str, class_name: str = None, osd_confidence_threshold: float = 0.5) -> Optional[Any]:
#         """
#         Enhanced preprocessing with special handling for Aadhaar numbers.
#         """
#         try:
#             img = entity_image
#             if img is None or img.size == 0:
#                 logger.warning(f"  Entity image data for {entity_key} is empty, skipping.")
#                 return None
            
#             # Special handling for Aadhaar numbers - skip complex orientation detection
#             if class_name == 'aadharnumber':
#                 logger.info(f"  Using specialized Aadhaar preprocessing for {entity_key}")
#                 # Return original image for specialized Aadhaar processing
#                 return img
            
#             h, w = img.shape[:2]
#             if h < 100:
#                 scale_factor = 100 / h
#                 new_w, new_h = int(w * scale_factor), int(h * scale_factor)
#                 img_for_analysis = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#             else:
#                 img_for_analysis = img

#             best_rotation = self._detect_orientation_by_letters(img_for_analysis, entity_key)
            
#             if best_rotation is None:
#                 try:
#                     osd = pytesseract.image_to_osd(img_for_analysis, output_type=pytesseract.Output.DICT)
#                     if osd['orientation_conf'] > osd_confidence_threshold:
#                         best_rotation = osd['rotate']
#                         logger.info(f" Using Tesseract OSD for {entity_key}: {best_rotation}° (conf: {osd['orientation_conf']:.2f})")
#                     else:
#                         best_rotation = 0
#                 except pytesseract.TesseractError as e:
#                     logger.warning(f" OSD failed for {entity_key}. Assuming 0° rotation.")
#                     best_rotation = 0
            
#             corrected_img = img
#             if best_rotation != 0:
#                 logger.info(f"   Correcting entity {entity_key} orientation by {best_rotation}°")
#                 if best_rotation == 90: 
#                     corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 elif best_rotation == 180: 
#                     corrected_img = cv2.rotate(img, cv2.ROTATE_180)
#                 elif best_rotation == 270: 
#                     corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

#             h_corr, w_corr = corrected_img.shape[:2]
#             if h_corr > w_corr and 'address' not in entity_key:
#                 logger.info(f"   Rotating vertical entity {entity_key} to horizontal format")
#                 corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_90_CLOCKWISE)

#             from PIL import Image
#             return Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
            
#         except Exception as e:
#             logger.error(f"   Unhandled error during entity orientation/preprocessing for {entity_key}: {e}")
#             return None

#     def _detect_orientation_by_letters(self, img: np.ndarray, entity_key: str) -> Optional[int]:
#         """
#         Detect the correct orientation by analyzing letter shapes and OCR confidence
#         at different rotation angles.
#         """
#         try:
#             rotations = [0, 90, 180, 270]
#             rotation_scores = {}
            
#             for rotation in rotations:
#                 if rotation == 0:
#                     rotated_img = img
#                 elif rotation == 90:
#                     rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 elif rotation == 180:
#                     rotated_img = cv2.rotate(img, cv2.ROTATE_180)
#                 elif rotation == 270:
#                     rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                
#                 score = self._calculate_orientation_score(rotated_img, rotation)
#                 rotation_scores[rotation] = score
#                 logger.debug(f"      Rotation {rotation}°: score = {score:.3f}")
            
#             best_rotation = max(rotation_scores.keys(), key=lambda k: rotation_scores[k])
#             best_score = rotation_scores[best_rotation]
            
#             if best_score > 0.1:
#                 logger.info(f" Letter-based analysis for {entity_key}: {best_rotation}° (score: {best_score:.3f})")
#                 return best_rotation
#             else:
#                 logger.warning(f"   Letter-based analysis inconclusive for {entity_key} (best score: {best_score:.3f})")
#                 return None
                
#         except Exception as e:
#             logger.warning(f"  Error in letter-based orientation detection for {entity_key}: {e}")
#             return None

#     def _calculate_orientation_score(self, img: np.ndarray, rotation: int) -> float:
#         """
#         Calculate a comprehensive score for how likely this orientation is correct.
#         Combines OCR confidence, letter shapes, and text line analysis.
#         """
#         try:
#             if len(img.shape) == 3:
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             else:
#                 gray = img
                
#             ocr_score = self._get_ocr_confidence_score(gray)
#             shape_score = self._analyze_letter_shapes(gray)
#             line_score = self._analyze_text_lines(gray)
            
#             total_score = (ocr_score * 0.5 + shape_score * 0.3 + line_score * 0.2)
#             return total_score
            
#         except Exception as e:
#             logger.debug(f"      Error calculating orientation score: {e}")
#             return 0.0

#     def _get_ocr_confidence_score(self, gray_img: np.ndarray) -> float:
#         """
#         Get OCR confidence and text quality score by trying multiple PSM modes.
#         Returns a normalized score between 0 and 1.
#         """
#         try:
#             psm_modes = [6, 7, 8, 13]
#             best_confidence = 0.0
#             best_text_length = 0
            
#             for psm in psm_modes:
#                 try:
#                     data = pytesseract.image_to_data(gray_img, config=f'--psm {psm}', output_type=pytesseract.Output.DICT)
#                     confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
#                     if confidences:
#                         avg_confidence = sum(confidences) / len(confidences)
#                         text_length = sum(len(text.strip()) for text in data['text'] if text.strip())
                        
#                         if avg_confidence > best_confidence or (avg_confidence == best_confidence and text_length > best_text_length):
#                             best_confidence = avg_confidence
#                             best_text_length = text_length
                            
#                 except pytesseract.TesseractError:
#                     continue
            
#             confidence_factor = best_confidence / 100.0
#             length_factor = min(best_text_length / 10.0, 1.0)
#             return confidence_factor * 0.7 + length_factor * 0.3
            
#         except Exception:
#             return 0.0

#     def _analyze_letter_shapes(self, gray_img: np.ndarray) -> float:
#         """
#         Analyze the shapes of detected contours to determine if they look like upright letters.
#         Returns a normalized score between 0 and 1.
#         """
#         try:
#             _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#             contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             if not contours:
#                 return 0.0
            
#             upright_score = 0.0
#             valid_contours = 0
#             for contour in contours:
#                 area = cv2.contourArea(contour)
#                 if area < 20:
#                     continue
#                 x, y, w, h = cv2.boundingRect(contour)
#                 if w < 5 or h < 5 or w > gray_img.shape[1] * 0.8 or h > gray_img.shape[0] * 0.8:
#                     continue
                
#                 aspect_ratio = h / w
#                 if 0.3 <= aspect_ratio <= 4.0:
#                     valid_contours += 1
#                     if 1.0 <= aspect_ratio <= 2.5:
#                         upright_score += 1.0
#                     elif 0.5 <= aspect_ratio <= 3.5:
#                         upright_score += 0.7
#                     else:
#                         upright_score += 0.3
            
#             if valid_contours == 0:
#                 return 0.0
#             return min(upright_score / valid_contours, 1.0)
            
#         except Exception:
#             return 0.0

#     def _analyze_text_lines(self, gray_img: np.ndarray) -> float:
#         """
#         Analyze text line orientation using morphological operations.
#         Horizontal text should have more horizontal lines than vertical lines.
#         Returns a normalized score between 0 and 1.
#         """
#         try:
#             _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
#             horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
#             horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
#             vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
#             vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
#             horizontal_pixels = cv2.countNonZero(horizontal_lines)
#             vertical_pixels = cv2.countNonZero(vertical_lines)
            
#             total_pixels = horizontal_pixels + vertical_pixels
#             if total_pixels == 0:
#                 return 0.5
            
#             horizontal_ratio = horizontal_pixels / total_pixels
#             return horizontal_ratio
            
#         except Exception:
#             return 0.5

#     def perform_multi_language_ocr(self, all_detections: Dict[str, Dict[str, Any]]):
#         """
#         Step 4: Correcting orientation and perform OCR with specialized handling for Aadhaar.
#         """
#         logger.info(f"\nStep 4: Correcting Entity Orientation & Performing Multi-Language OCR")
#         ocr_results = {}
#         for card_name, card_data in all_detections.items():
#             for detection in card_data['detections']:
#                 cropped_image = detection.get('cropped_image')
#                 entity_key = detection.get('entity_key')
#                 class_name = detection.get('class_name')

#                 if cropped_image is None or entity_key is None:
#                     continue

#                 logger.info(f"  Processing entity: {entity_key} (Class: {class_name})")
                
#                 # Special handling for Aadhaar numbers
#                 if class_name == 'aadharnumber':
#                     extracted_text = self._extract_aadhaar_with_multiple_methods(cropped_image)
#                     ocr_results[entity_key] = extracted_text
#                     if extracted_text:
#                         logger.info(f"    Aadhaar OCR Result: {extracted_text}")
#                     else:
#                         logger.warning(f"    Aadhaar OCR failed to extract number")
#                     continue
                
#                 # Regular processing for other entities
#                 lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                
#                 processed_pil_img = self._correct_entity_orientation_and_preprocess(cropped_image, entity_key, class_name)

#                 if processed_pil_img:
#                     try:
#                         text = pytesseract.image_to_string(processed_pil_img, lang=lang_to_use, config='--psm 6')
#                         extracted_text = ' '.join(text.split()).strip()
#                         ocr_results[entity_key] = extracted_text
#                         logger.info(f"    OCR Result: {extracted_text[:50]}..." if len(extracted_text) > 50 else f"    OCR Result: {extracted_text}")
#                     except Exception as e:
#                         logger.error(f" OCR failed for {entity_key}: {e}")
#                         ocr_results[entity_key] = None
#         return ocr_results

#     def organize_results_by_card_type(self, all_detections: Dict[str, Dict[str, Any]], ocr_results: Dict[str, str], confidence_threshold: float):
#         """Your EXACT original result organization"""
#         logger.info("\nStep 5: Organizing final results")
#         organized_results = {
#             'front': {}, 'back': {},
#             'metadata': {
#                 'processing_timestamp': datetime.now().isoformat(),
#                 'confidence_threshold_used': confidence_threshold
#             }
#         }
#         for card_name, card_data in all_detections.items():
#             card_type = card_data['card_type']
            
#             if card_name not in organized_results[card_type]:
#                 organized_results[card_type][card_name] = {'entities': {}}
            
#             for detection in card_data['detections']:
#                  entity_name = detection['class_name']
#                  entity_key = detection.get('entity_key')
#                  extracted_text = ocr_results.get(entity_key)

#                  if entity_name not in organized_results[card_type][card_name]['entities']:
#                       organized_results[card_type][card_name]['entities'][entity_name] = []
                 
#                  organized_results[card_type][card_name]['entities'][entity_name].append({
#                      'confidence': detection['confidence'], 
#                      'bbox': detection['bbox'],
#                      'extracted_text': extracted_text
#                  })
#         return organized_results

#     def extract_main_fields(self, organized_results: Dict[str, Any]) -> Dict[str, Any]:
#         """Your EXACT original Validation Logic (Moved inside class)"""
#         fields = ['aadharnumber', 'dob', 'gender']
#         data = {key: "" for key in fields}
        
#         # Special handling for Aadhar number to check both front and back
#         aadhar_front = ""
#         aadhar_back = ""
        
#         for side in ['front', 'back']:
#             for card in organized_results.get(side, {}).values():
#                 for field in fields:
#                     if field in card['entities'] and card['entities'][field]:
#                         all_texts = [item.get('extracted_text', '') for item in card['entities'][field]]
#                         first_valid_text = next((text for text in all_texts if text), '')
                        
#                         if first_valid_text:
#                             if field == 'aadharnumber':
#                                 if side == 'front':
#                                     aadhar_front = first_valid_text
#                                 elif side == 'back':
#                                     aadhar_back = first_valid_text
#                             else:
#                                 data[field] = first_valid_text
        
#         # --- Special Logic for Aadhar Number from Front & Back ---
#         best_aadhar = ""
#         aadhar_front_digits = re.sub(r'\D', '', aadhar_front) if aadhar_front else ""
#         aadhar_back_digits = re.sub(r'\D', '', aadhar_back) if aadhar_back else ""
        
#         logger.info(f"Aadhar Front: '{aadhar_front}' -> Digits: '{aadhar_front_digits}'")
#         logger.info(f"Aadhar Back: '{aadhar_back}' -> Digits: '{aadhar_back_digits}'")
        
#         if aadhar_front_digits == aadhar_back_digits and aadhar_front_digits:
#             # Both match - use this
#             logger.info(f"Aadhar match between front and back: {aadhar_front_digits}")
#             best_aadhar = aadhar_front_digits
#         else:
#             # Different values - prefer the one with all 12 digits, or the one with more digits
#             if len(aadhar_front_digits) == 12:
#                 logger.info(f"Using Aadhar from front (complete 12 digits): {aadhar_front_digits}")
#                 best_aadhar = aadhar_front_digits
#             elif len(aadhar_back_digits) == 12:
#                 logger.info(f"Using Aadhar from back (complete 12 digits): {aadhar_back_digits}")
#                 best_aadhar = aadhar_back_digits
#             elif len(aadhar_front_digits) > len(aadhar_back_digits):
#                 logger.info(f"Using Aadhar from front (more digits): {aadhar_front_digits}")
#                 best_aadhar = aadhar_front_digits
#             elif len(aadhar_back_digits) > len(aadhar_front_digits):
#                 logger.info(f"Using Aadhar from back (more digits): {aadhar_back_digits}")
#                 best_aadhar = aadhar_back_digits
#             elif aadhar_front_digits:
#                 logger.info(f"Using Aadhar from front (fallback): {aadhar_front_digits}")
#                 best_aadhar = aadhar_front_digits
#             elif aadhar_back_digits:
#                 logger.info(f"Using Aadhar from back (fallback): {aadhar_back_digits}")
#                 best_aadhar = aadhar_back_digits
        
#         data['aadharnumber'] = best_aadhar
        
#         # --- Aadhaar Number Validation ---
#         aadhar_status = "aadhar_approved"
#         if data.get('aadharnumber'):
#             aad = data['aadharnumber']
#             # Extract only digits, remove all spaces and special characters
#             aad_digits_only = re.sub(r'\D', '', aad)  # Remove all non-digit characters
            
#             # Check for masked Aadhaar (with X's) in original text
#             if (aadhar_front and re.search(r'X{4}', aadhar_front, re.IGNORECASE)) or \
#                (aadhar_back and re.search(r'X{4}', aadhar_back, re.IGNORECASE)):
#                 aadhar_status = "aadhar_disapproved"
#             # Validate that we have exactly 12 digits
#             elif len(aad_digits_only) == 12:
#                 data['aadharnumber'] = aad_digits_only
#             else:
#                 # If not exactly 12 digits, disapprove
#                 aadhar_status = "aadhar_disapproved"
#                 data['aadharnumber'] = aad_digits_only  # Store what we got anyway
#         else:
#             aadhar_status = "aadhar_disapproved"
#         data['aadhar_status'] = aadhar_status
        
#         # --- DOB Processing and Age Verification (Year-based only) ---
#         age_status = "age_disapproved"
#         birth_year = None
        
#         if data.get('dob'):
#             # Extract all digit groups from DOB text
#             digit_groups = re.findall(r'\d+', data['dob'])
            
#             # Look for a 4-digit year
#             year = next((g for g in digit_groups if len(g) == 4 and 1900 <= int(g) <= datetime.now().year), None)
            
#             if year:
#                 try:
#                     birth_year = int(year)
#                     data['dob'] = str(birth_year)  # Store only the year
                    
#                     # Calculate age based on year only
#                     current_year = datetime.now().year
#                     age = current_year - birth_year
#                     data['age'] = age
                    
#                     # Approve if age >= 18
#                     if age >= 18:
#                         age_status = "age_approved"
#                         logger.info(f"Age approved: {age} years old (born in {birth_year})")
#                     else:
#                         logger.info(f"Age rejected: {age} years old (born in {birth_year})")
#                 except ValueError:
#                     data['dob'] = 'Invalid Format'
#                     data['age'] = None
#                     logger.warning(f"Could not parse year from DOB: {data.get('dob')}")
#             else:
#                 data['dob'] = 'Invalid Format'
#                 data['age'] = None
#                 logger.warning(f"No valid 4-digit year found in DOB: {data.get('dob')}")
#         else:
#             data['dob'] = 'Not Detected'
#             data['age'] = None
#             logger.info("DOB not detected")
        
#         data['age_status'] = age_status
        
#         # --- Gender normalization (Enhanced) ---
#         if data['gender']:
#             gender = data['gender'].strip().lower()
#             # Remove special characters and extra spaces
#             gender = re.sub(r'[^a-z]', '', gender)
            
#             # Check for male variations
#             if 'male' in gender and 'female' not in gender:
#                 data['gender'] = 'Male'
#             # Check for female variations
#             elif 'female' in gender or 'femal' in gender:
#                 data['gender'] = 'Female'
#             # Check for other common variations
#             elif gender in ['m', 'man', 'boy']:
#                 data['gender'] = 'Male'
#             elif gender in ['f', 'woman', 'girl', 'femlae', 'femaie']:
#                 data['gender'] = 'Female'
#             else:
#                 data['gender'] = 'Other'
#         else:
#             data['gender'] = 'Not Detected'
            
#         return data

# # Usage Check
# if __name__ == "__main__":
#     try:
#         agent = EntityAgent(model_path="models/best.pt")
#         print("✓ Entity Agent initialized successfully")
#     except Exception as e:
#         print(f"✗ Initialization failed: {e}")