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

# Import Qwen fallback
from app.qwen_fallback import QwenFallbackAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityAgent:
    def __init__(self, model_path="models/best.pt", other_lang_code='hin+tel+ben', 
                 save_debug_images=True, enable_qwen_fallback=True):

        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA is available. Models will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("CUDA not available. Models will fall back to CPU.")

        # --- Model Loading ---
        self.entity_model_path = model_path
        self.save_debug_images = save_debug_images
        
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
        
        # Initialize Qwen fallback agent
        self.enable_qwen_fallback = enable_qwen_fallback
        self.qwen_agent = None
        
        if self.enable_qwen_fallback:
            try:
                logger.info("Initializing Qwen3-VL fallback agent...")
                self.qwen_agent = QwenFallbackAgent()
                logger.info("✅ Qwen3-VL fallback agent ready")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize Qwen fallback: {e}")
                logger.warning("Continuing without Qwen fallback support")
                self.enable_qwen_fallback = False

    def _check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            logger.critical("Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
            raise RuntimeError("Tesseract not found")
        except Exception as e:
             logger.critical(f"An error occurred while checking Tesseract: {e}")
             raise RuntimeError(f"Error checking Tesseract: {e}")

    def _is_masked_aadhaar(self, text: str) -> bool:
        """
        Enhanced masked Aadhaar detection with multiple checks.
        
        Returns True if the Aadhaar is masked (contains X's).
        """
        if not text:
            return False
        
        text_upper = str(text).upper()
        
        # Check 1: Direct presence of 'X' character
        if 'X' in text_upper:
            logger.warning(f"🚫 MASKED AADHAAR DETECTED (contains X): '{text}'")
            return True
        
        # Check 2: Pattern matching for common masked formats
        # XXXX XXXX 1234, XXXXXXXX1234, etc.
        masked_patterns = [
            r'X{4,}',  # 4 or more X's in a row
            r'[X\s-]{8,}[0-9]{4}',  # 8+ chars of X/space/dash followed by 4 digits
            r'[0-9X\s-]*X{2,}[0-9X\s-]*',  # Any sequence containing 2+ X's
        ]
        
        for pattern in masked_patterns:
            if re.search(pattern, text_upper):
                logger.warning(f"🚫 MASKED AADHAAR DETECTED (pattern match): '{text}'")
                return True
        
        # Check 3: Look for common masked Aadhaar indicators
        # Sometimes OCR might read 'X' as similar characters
        suspicious_chars = ['×', 'x', 'X', '*', '×']
        text_clean = text.replace(' ', '').replace('-', '')
        
        suspicious_count = sum(text_clean.count(char) for char in suspicious_chars)
        if suspicious_count >= 4:  # If 4+ suspicious characters
            logger.warning(f"🚫 MASKED AADHAAR DETECTED (suspicious chars): '{text}'")
            return True
        
        return False

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
        
        return best_rotation

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
                # Check if valid 12-digit number AND not masked
                digits = re.sub(r'\D', '', str(value))
                is_masked = self._is_masked_aadhaar(str(value))
                missing[field] = (len(digits) != 12) or is_masked
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
                    
                    # Check for masking
                    full_masked = self._is_masked_aadhaar(str(full_value))
                    cropped_masked = self._is_masked_aadhaar(str(cropped_value))
                    
                    if not full_masked and len(full_digits) == 12 and (len(cropped_digits) != 12 or cropped_masked):
                        logger.info(f"  ✓ Using full image {field}: {full_digits}")
                        merged[field] = full_digits
                        merged['aadhar_status'] = 'aadhar_approved'
                        if 'aadhar_rejection_reason' in merged:
                            del merged['aadhar_rejection_reason']
                    elif len(full_digits) > len(cropped_digits) and not full_masked:
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
        Main Entry Point with CARD ROTATION DETECTION + FALLBACK mechanism + QWEN FALLBACK.
        
        Flow:
        1. Detect card rotation
        2. Try extraction with crop coordinates
        3. If critical fields missing, try full image (2nd fallback)
        4. If still missing/invalid, try Qwen VLM (3rd fallback)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = None
            
            # 1. Read ORIGINAL Image
            original_img = cv2.imread(file_path)
            if original_img is None:
                logger.error(f"Failed to read image: {file_path}")
                return {"error": "failed_to_read_file"}

            # 2. DETECT CARD-LEVEL ROTATION
            card_rotation = self._detect_card_rotation(original_img, session_dir)
            
            # 3. Apply card rotation to original image
            if card_rotation != 0:
                logger.info(f"🔄 Applying card rotation: {card_rotation}°")
                if card_rotation == 90:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif card_rotation == 180:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_180)
                elif card_rotation == 270:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)

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
                    x1, y1, x2, y2 = y1, w - x2, y2, w - x1
                elif card_rotation == 180:
                    x1, y1, x2, y2 = w - x2, h - y2, w - x1, h - y1
                elif card_rotation == 270:
                    x1, y1, x2, y2 = h - y2, x1, h - y1, x2
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                logger.info(f"Cropping image to: [{x1}, {y1}, {x2}, {y2}] (after rotation adjustment)")
                img = img[y1:y2, x1:x2]
            else:
                logger.info("No crop coordinates provided, using full image")

            # Run detection and OCR on cropped image
            all_detections = self.detect_entities_in_image(img, confidence_threshold, card_side, session_dir)
            
            if not all_detections:
                logger.warning("⚠️ No entities detected in cropped region")
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
            
            # Track which image to use for Qwen fallback
            image_for_qwen = img  # Start with cropped image
            
            if has_missing:
                logger.warning("⚠️ CRITICAL FIELDS MISSING from cropped extraction:")
                for field, is_missing in missing_fields.items():
                    if is_missing:
                        logger.warning(f"  ❌ {field}: Missing or Invalid")
                
                # === ATTEMPT 2: FALLBACK to Full Image ===
                logger.info("\n" + "=" * 60)
                logger.info("🔄 ATTEMPT 2: FALLBACK - Running on FULL ORIGINAL image")
                logger.info("=" * 60)
                
                full_detections = self.detect_entities_in_image(
                    original_img, confidence_threshold, card_side, session_dir
                )
                
                if full_detections:
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
                    
                    # Use full image for Qwen if needed
                    image_for_qwen = original_img
                else:
                    logger.error("❌ Fallback detection on full image also failed")
                    final_data = cropped_result.get("data", {})
                
                # Check again after merge
                missing_fields_after_merge = self._check_critical_fields(final_data, card_side)
                has_missing = any(missing_fields_after_merge.values())
            else:
                logger.info("✅ All critical fields present in cropped extraction")
                final_data = cropped_result.get("data", {})
                missing_fields_after_merge = missing_fields
            
            # === ATTEMPT 3: QWEN VLM FALLBACK ===
            used_qwen_fallback = False
            
            if self.enable_qwen_fallback and self.qwen_agent:
                should_use_qwen, qwen_missing_fields = self.qwen_agent.should_use_fallback(final_data)
                
                if should_use_qwen:
                    logger.info("\n" + "=" * 60)
                    logger.info("🤖 ATTEMPT 3: QWEN VLM FALLBACK")
                    logger.info("=" * 60)
                    
                    try:
                        qwen_results = self.qwen_agent.extract_fields(
                            image_for_qwen, 
                            qwen_missing_fields, 
                            card_side
                        )
                        
                        if qwen_results:
                            # Merge Qwen results
                            final_data = self.qwen_agent.validate_and_merge(final_data, qwen_results)
                            used_qwen_fallback = True
                            logger.info("✅ Qwen fallback completed successfully")
                        else:
                            logger.warning("⚠️ Qwen fallback returned no results")
                            
                    except Exception as e:
                        logger.error(f"❌ Qwen fallback failed: {e}")
                        logger.error(traceback.format_exc())
            
            # Final validation summary
            logger.info("\n" + "=" * 60)
            logger.info("📋 FINAL EXTRACTION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Aadhaar Number: {final_data.get('aadharnumber', 'N/A')}")
            logger.info(f"Aadhaar Status: {final_data.get('aadhar_status', 'N/A')}")
            if final_data.get('aadhar_rejection_reason'):
                logger.info(f"Rejection Reason: {final_data.get('aadhar_rejection_reason')}")
            logger.info(f"DOB: {final_data.get('dob', 'N/A')}")
            logger.info(f"Age: {final_data.get('age', 'N/A')}")
            logger.info(f"Age Status: {final_data.get('age_status', 'N/A')}")
            logger.info(f"Gender: {final_data.get('gender', 'N/A')}")
            logger.info(f"Card Rotation: {card_rotation}°")
            logger.info(f"Used Full Image Fallback: {has_missing}")
            logger.info(f"Used Qwen Fallback: {used_qwen_fallback}")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "data": final_data,
                "debug_dir": str(session_dir) if session_dir else None,
                "card_rotation": card_rotation,
                "used_fallback": has_missing,
                "used_qwen_fallback": used_qwen_fallback
            }

        except Exception as e:
            logger.error(f"Error in extraction: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def detect_entities_in_image(self, image_input, confidence_threshold: float, card_side: str = 'front', session_dir: Path = None):
        """
        Modified to accept card_side parameter and filter entities accordingly.
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
            target_entities = {'aadharnumber', 'dob', 'gender'}
        
        logger.info(f"  Target entities for {card_side}: {target_entities}")
        
        # Run entity detection
        results = self.model2(img, device=self.device, verbose=False)
        card_detections = []
        
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
        
        logger.info(f"  Detected {len(card_detections)} entities")
        
        if not card_detections:
            return {}

        # Wrap in original structure
        all_detections = {
            input_name: {
                "card_image": img,
                "card_type": card_side,
                "detections": card_detections
            }
        }
        
        return all_detections

    def crop_entities(self, all_detections: Dict[str, Dict[str, Any]], session_dir: Path = None):
        """Step 3: Crop individual entities with bounds checking"""
        logger.info(f"\nStep 3: Cropping individual entities")
        
        for card_name, card_data in all_detections.items():
            img = card_data['card_image']
            h, w = img.shape[:2]
            
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
            
            # Resize if too small
            h, w = gray.shape
            if h < 50:
                scale = 50 / h
                new_w, new_h = int(w * scale), int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
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
        Returns the RAW OCR text (preserving X's for masked detection).
        """
        try:
            preprocessed_versions = self._preprocess_for_aadhaar_ocr(img)
            
            configs = [
                '--psm 7',
                '--psm 8',
                '--psm 6',
                '--psm 7 -c tessedit_char_whitelist=0123456789X',
                '--psm 13',
            ]
            
            best_result = ""
            best_score = 0
            
            for method_name, processed_img in preprocessed_versions:
                for rotation in [0, 90, 180, 270]:
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
                            text = pytesseract.image_to_string(pil_img, lang='eng', config=config).strip()
                            
                            valid_chars = re.findall(r'[0-9X]', text, re.IGNORECASE)
                            score = len(valid_chars)
                            
                            if score > best_score:
                                best_score = score
                                best_result = text
                            
                            if score >= 12:
                                return text
                                
                        except Exception as e:
                            continue
            
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
            
            if class_name == 'aadharnumber':
                logger.info(f"  Using specialized Aadhaar preprocessing for {entity_key}")
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
                    else:
                        best_rotation = 0
                except:
                    best_rotation = 0
            
            corrected_img = img
            if best_rotation != 0:
                if best_rotation == 90: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif best_rotation == 180: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_180)
                elif best_rotation == 270: 
                    corrected_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            h_corr, w_corr = corrected_img.shape[:2]
            if h_corr > w_corr and 'address' not in entity_key:
                corrected_img = cv2.rotate(corrected_img, cv2.ROTATE_90_CLOCKWISE)

            from PIL import Image
            return Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
            
        except Exception as e:
            logger.error(f"   Error during entity orientation/preprocessing for {entity_key}: {e}")
            return None

    def _detect_orientation_by_letters(self, img: np.ndarray, entity_key: str) -> Optional[int]:
        """Detect orientation by analyzing letter shapes and OCR confidence"""
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
            
            best_rotation = max(rotation_scores.keys(), key=lambda k: rotation_scores[k])
            best_score = rotation_scores[best_rotation]
            
            if best_score > 0.1:
                return best_rotation
            return None
                
        except Exception as e:
            return None

    def _calculate_orientation_score(self, img: np.ndarray, rotation: int) -> float:
        """Calculate orientation score"""
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
            
        except:
            return 0.0

    def _get_ocr_confidence_score(self, gray_img: np.ndarray) -> float:
        """Get OCR confidence score"""
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
                except:
                    continue
            
            confidence_factor = best_confidence / 100.0
            length_factor = min(best_text_length / 10.0, 1.0)
            return confidence_factor * 0.7 + length_factor * 0.3
            
        except:
            return 0.0

    def _analyze_letter_shapes(self, gray_img: np.ndarray) -> float:
        """Analyze letter shapes"""
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
            
        except:
            return 0.0

    def _analyze_text_lines(self, gray_img: np.ndarray) -> float:
        """Analyze text line orientation"""
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
            
            return horizontal_pixels / total_pixels
            
        except:
            return 0.5

    def perform_multi_language_ocr(self, all_detections: Dict[str, Dict[str, Any]]):
        """Step 4: Perform OCR with specialized handling for Aadhaar"""
        logger.info(f"\nStep 4: Performing Multi-Language OCR")
        ocr_results = {}
        
        for card_name, card_data in all_detections.items():
            for detection in card_data['detections']:
                cropped_image = detection.get('cropped_image')
                entity_key = detection.get('entity_key')
                class_name = detection.get('class_name')

                if cropped_image is None or entity_key is None:
                    continue

                logger.info(f"  Processing: {entity_key}")
                
                if class_name == 'aadharnumber':
                    extracted_text = self._extract_aadhaar_with_multiple_methods(cropped_image)
                    ocr_results[entity_key] = extracted_text
                    if extracted_text:
                        logger.info(f"    Result: {extracted_text}")
                    continue
                
                lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                processed_pil_img = self._correct_entity_orientation_and_preprocess(cropped_image, entity_key, class_name)

                if processed_pil_img:
                    try:
                        text = pytesseract.image_to_string(processed_pil_img, lang=lang_to_use, config='--psm 6')
                        extracted_text = ' '.join(text.split()).strip()
                        ocr_results[entity_key] = extracted_text
                        logger.info(f"    Result: {extracted_text[:50]}...")
                    except Exception as e:
                        ocr_results[entity_key] = None
        return ocr_results

    def organize_results_by_card_type(self, all_detections: Dict[str, Dict[str, Any]], ocr_results: Dict[str, str], confidence_threshold: float):
        """Step 5: Organize results"""
        logger.info("\nStep 5: Organizing results")
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
        """Extract and validate main fields with STRICT masked Aadhaar rejection"""
        fields = ['aadharnumber', 'dob', 'gender']
        data = {key: "" for key in fields}
        
        aadhar_front = ""
        aadhar_back = ""
        
        for side in ['front', 'back']:
            for card in organized_results.get(side, {}).values():
                for field in fields:
                    if field in card['entities'] and card['entities'][field]:
                        all_texts = [item.get('extracted_text', '') for item in card['entities'][field]]
                        
                        if field == 'aadharnumber':
                            first_valid_text = next((text for text in all_texts if text), '')
                            if first_valid_text:
                                if side == 'front':
                                    aadhar_front = first_valid_text
                                elif side == 'back':
                                    aadhar_back = first_valid_text
                        elif field == 'dob':
                            for text in all_texts:
                                if text:
                                    digit_groups = re.findall(r'\d+', text)
                                    has_valid_year = any(
                                        len(g) == 4 and 1900 <= int(g) <= datetime.now().year 
                                        for g in digit_groups
                                    )
                                    if has_valid_year:
                                        data[field] = text
                                        break
                            if not data.get('dob'):
                                first_valid_text = next((text for text in all_texts if text), '')
                                if first_valid_text:
                                    data[field] = first_valid_text
                        else:
                            first_valid_text = next((text for text in all_texts if text), '')
                            if first_valid_text:
                                data[field] = first_valid_text
        
        # === CRITICAL: Check for Masked Aadhaar FIRST ===
        logger.info("=" * 60)   
        logger.info("🔍 CHECKING FOR MASKED AADHAAR")
        logger.info("=" * 60)
        logger.info(f"Aadhar Front (RAW): '{aadhar_front}'")
        logger.info(f"Aadhar Back (RAW): '{aadhar_back}'")
        
        is_masked_front = self._is_masked_aadhaar(aadhar_front)
        is_masked_back = self._is_masked_aadhaar(aadhar_back)
        
        if is_masked_front or is_masked_back:
            masked_source = 'front' if is_masked_front else 'back'
            if is_masked_front and is_masked_back:
                masked_source = 'both'
            
            logger.error(f"🚫 REJECTING: Masked Aadhaar detected in {masked_source}")
            
            # Store the masked value
            data['aadharnumber'] = aadhar_front if aadhar_front else aadhar_back
            data['aadhar_status'] = "aadhar_disapproved"
            data['aadhar_rejection_reason'] = "masked_aadhar"
            
            # Continue processing DOB and gender but Aadhaar is REJECTED
        else:
            logger.info("✅ No masking detected")
            
            # Process Aadhaar normally
            best_aadhar = ""
            aadhar_front_digits = re.sub(r'\D', '', aadhar_front) if aadhar_front else ""
            aadhar_back_digits = re.sub(r'\D', '', aadhar_back) if aadhar_back else ""
            
            if aadhar_front_digits == aadhar_back_digits and aadhar_front_digits:
                best_aadhar = aadhar_front_digits
            else:
                if len(aadhar_front_digits) == 12:
                    best_aadhar = aadhar_front_digits
                elif len(aadhar_back_digits) == 12:
                    best_aadhar = aadhar_back_digits
                elif len(aadhar_front_digits) > len(aadhar_back_digits):
                    best_aadhar = aadhar_front_digits
                elif len(aadhar_back_digits) > len(aadhar_front_digits):
                    best_aadhar = aadhar_back_digits
                elif aadhar_front_digits:
                    best_aadhar = aadhar_front_digits
                elif aadhar_back_digits:
                    best_aadhar = aadhar_back_digits
            
            data['aadharnumber'] = best_aadhar
            
            # Validate Aadhaar
            aadhar_status = "aadhar_approved"
            aadhar_rejection_reason = None
            
            if data.get('aadharnumber'):
                aad_digits_only = re.sub(r'\D', '', data['aadharnumber'])
                
                if len(aad_digits_only) == 12:
                    data['aadharnumber'] = aad_digits_only
                else:
                    aadhar_status = "aadhar_disapproved"
                    aadhar_rejection_reason = "invalid_length"
                    data['aadharnumber'] = aad_digits_only
            else:
                aadhar_status = "aadhar_disapproved"
                aadhar_rejection_reason = "not_detected"
            
            data['aadhar_status'] = aadhar_status
            if aadhar_rejection_reason:
                data['aadhar_rejection_reason'] = aadhar_rejection_reason
        
        # DOB Processing
        age_status = "age_disapproved"
        
        if data.get('dob'):
            digit_groups = re.findall(r'\d+', data['dob'])
            year = next((g for g in digit_groups if len(g) == 4 and 1900 <= int(g) <= datetime.now().year), None)
            
            if year:
                birth_year = int(year)
                data['dob'] = str(birth_year)
                
                current_year = datetime.now().year
                age = current_year - birth_year
                data['age'] = age
                
                if age >= 18:
                    age_status = "age_approved"
            else:
                data['dob'] = 'Invalid Format'
                data['age'] = None
        else:
            data['dob'] = 'Not Detected'
            data['age'] = None
        
        data['age_status'] = age_status
        
        # Gender normalization
        if data['gender']:
            gender = data['gender'].strip().lower()
            gender = re.sub(r'[^a-z]', '', gender)
            
            if 'male' in gender and 'female' not in gender:
                data['gender'] = 'Male'
            elif 'female' in gender or 'femal' in gender:
                data['gender'] = 'Female'
            elif gender in ['m', 'man', 'boy']:
                data['gender'] = 'Male'
            elif gender in ['f', 'woman', 'girl']:
                data['gender'] = 'Female'
            else:
                data['gender'] = 'Other'
        else:
            data['gender'] = 'Not Detected'
        
        return data


if __name__ == "__main__":
    try:
        agent = EntityAgent(model_path="models/best.pt", enable_qwen_fallback=True)
        print("✅ Entity Agent initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")