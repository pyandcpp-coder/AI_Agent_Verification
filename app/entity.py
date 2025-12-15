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

    def _is_garbage_text(self, text: str) -> bool:
        """
        Detect if text is garbage/nonsense (random characters, OCR errors, etc.).
        
        Returns True if text appears to be garbage.
        """
        if not text:
            return True
        
        text_str = str(text).strip()
        
        # Check 1: Too long (Aadhaar should be ~12 chars, even with spaces maybe 20)
        if len(text_str) > 50:
            logger.warning(f"  Garbage check: Too long ({len(text_str)} chars)")
            return True
        
        # Check 2: Contains newlines (Aadhaar should be single line)
        if '\n' in text_str or '\r' in text_str:
            logger.warning(f"  Garbage check: Contains newlines")
            return True
        
        # Check 3: Too many non-alphanumeric characters
        # Aadhaar can have spaces, dashes, but not much else
        non_alphanum = sum(1 for c in text_str if not c.isalnum() and c not in [' ', '-'])
        if non_alphanum > 5:
            logger.warning(f"  Garbage check: Too many special chars ({non_alphanum})")
            return True
        
        # Check 4: Contains obvious garbage patterns
        garbage_patterns = [
            r'[a-zA-Z]{10,}',  # Long sequences of letters (10+ in a row)
            r'\.{3,}',  # Multiple dots
            r'\s{5,}',  # Long whitespace sequences
            r'[^\w\s-]{3,}',  # 3+ special chars in a row
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, text_str):
                logger.warning(f"  Garbage check: Matched pattern {pattern}")
                return True
        
        # Check 5: Very low digit ratio (Aadhaar should be mostly digits)
        digits = sum(1 for c in text_str if c.isdigit())
        total_chars = len(text_str.replace(' ', '').replace('-', ''))
        
        if total_chars > 0:
            digit_ratio = digits / total_chars
            if digit_ratio < 0.5:  # Less than 50% digits
                logger.warning(f"  Garbage check: Low digit ratio ({digit_ratio:.2%})")
                return True
        
        return False

    def _is_masked_aadhaar(self, text: str) -> bool:
        """
        ULTRA STRICT masked Aadhaar detection with multiple checks.
        
        Returns True if the Aadhaar is masked (contains X's or similar characters).
        """
        if not text:
            return False
        
        text_upper = str(text).upper()
        text_clean = text.replace(' ', '').replace('-', '').upper()
        
        # Check 1: Direct presence of 'X' character (case insensitive)
        if 'X' in text_upper:
            logger.error(f"🚫 MASKED AADHAAR DETECTED (contains X): '{text}'")
            return True
        
        # Check 2: Pattern matching for common masked formats
        masked_patterns = [
            r'X{2,}',  # 2 or more X's in a row
            r'[X\s-]{4,}[0-9]{4}',  # 4+ chars of X/space/dash followed by 4 digits
            r'[0-9X\s-]*X+[0-9X\s-]*',  # Any sequence containing X
            r'\*{2,}',  # 2 or more asterisks
            r'[*\s-]{4,}[0-9]{4}',  # Asterisk patterns
        ]
        
        for pattern in masked_patterns:
            if re.search(pattern, text_upper):
                logger.error(f"🚫 MASKED AADHAAR DETECTED (pattern: {pattern}): '{text}'")
                return True
        
        # Check 3: Look for suspicious characters that might be X or similar
        suspicious_chars = ['×', 'x', 'X', '*', '×', '✕', '✖', 'χ']
        suspicious_count = sum(text_clean.count(char) for char in suspicious_chars)
        
        if suspicious_count >= 2:  # Even 2 suspicious characters mean masked
            logger.error(f"🚫 MASKED AADHAAR DETECTED ({suspicious_count} suspicious chars): '{text}'")
            return True
        
        # Check 4: Check if cleaned text has non-digit characters in positions 0-7
        # Normal Aadhaar: all digits. Masked: XXXX XXXX 1234
        if len(text_clean) >= 8:
            first_eight = text_clean[:8]
            non_digit_count = sum(1 for c in first_eight if not c.isdigit())
            if non_digit_count >= 2:  # If first 8 chars have 2+ non-digits, likely masked
                logger.error(f"🚫 MASKED AADHAAR DETECTED (non-digits in first 8 chars): '{text}'")
                return True
        
        return False

    def _validate_aadhaar_number(self, text: str) -> Tuple[bool, str, Optional[str]]:
        """
        STRICT Aadhaar validation.
        
        Returns:
            (is_valid, cleaned_number, rejection_reason)
        """
        if not text:
            return False, "", "not_detected"
        
        text_str = str(text).strip()
        
        # CRITICAL: Check for garbage FIRST
        if self._is_garbage_text(text_str):
            logger.error(f"🗑️ VALIDATION FAILED: Garbage text detected: '{text_str[:100]}...'")
            # Try to salvage any digits
            digits_only = re.sub(r'\D', '', text_str)
            if len(digits_only) >= 8:
                return False, digits_only, "invalid_length"
            return False, "", "not_detected"
        
        # First check for masking - THIS IS PRIORITY #1
        if self._is_masked_aadhaar(text_str):
            logger.error(f"🚫 VALIDATION FAILED: Masked Aadhaar: '{text_str}'")
            return False, text_str, "masked_aadhar"
        
        # Extract only digits
        digits_only = re.sub(r'\D', '', text_str)
        
        # Check length - MUST BE EXACTLY 12 DIGITS
        if len(digits_only) != 12:
            logger.error(f"🚫 VALIDATION FAILED: Invalid length ({len(digits_only)} digits): '{text_str}' -> '{digits_only}'")
            return False, digits_only, "invalid_length"
        
        # Check for all zeros or all same digit
        if digits_only == '0' * 12 or len(set(digits_only)) == 1:
            logger.error(f"🚫 VALIDATION FAILED: Invalid pattern (all same): '{digits_only}'")
            return False, digits_only, "invalid_pattern"
        
        logger.info(f"✅ VALIDATION PASSED: '{digits_only}'")
        return True, digits_only, None

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
                if conf >= 0.15:
                    class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
                    detected_entities.append((class_name, conf))
                    total_conf += conf
                    
                    if class_name in critical_entities:
                        critical_entities[class_name] += 1
            
            # Calculate score
            num_entities = len(detected_entities)
            avg_conf = total_conf / num_entities if num_entities > 0 else 0
            critical_count = sum(1 for v in critical_entities.values() if v > 0)
            
            score = (
                (num_entities / 10) * 0.4 +
                avg_conf * 0.3 +
                (critical_count / 3) * 0.3
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
                # Use strict validation
                is_valid, _, _ = self._validate_aadhaar_number(value)
                missing[field] = not is_valid
            elif field == 'dob':
                missing[field] = value in ['Invalid Format', 'Not Detected', '', None]
            elif field == 'gender':
                missing[field] = value in ['Not Detected', 'Other', '', None]
            else:
                missing[field] = not value
        
        return missing

    def _merge_results_without_mask_blocking(self, cropped_data: Dict[str, Any], 
                                             full_data: Dict[str, Any], 
                                             missing_fields: Dict[str, bool]) -> Dict[str, Any]:
        """
        Merge results WITHOUT blocking masked Aadhaar early.
        Let Qwen verify later.
        """
        merged = cropped_data.copy()
        
        logger.info("🔀 Merging results (without early mask rejection)...")
        
        for field, is_missing in missing_fields.items():
            if is_missing:
                full_value = full_data.get(field)
                cropped_value = cropped_data.get(field)
                
                if field == 'aadharnumber':
                    # Validate both values but DON'T reject masked yet
                    full_valid, full_clean, full_reason = self._validate_aadhaar_number(full_value)
                    cropped_valid, cropped_clean, cropped_reason = self._validate_aadhaar_number(cropped_value)
                    
                    # If full is valid (not masked, 12 digits), use it
                    if full_valid:
                        logger.info(f"  ✓ Using VALID full image {field}: {full_clean}")
                        merged[field] = full_clean
                        merged['aadhar_status'] = 'aadhar_approved'
                        if 'aadhar_rejection_reason' in merged:
                            del merged['aadhar_rejection_reason']
                    # Otherwise, use the one with more digits (even if masked)
                    elif len(full_clean) > len(cropped_clean):
                        logger.warning(f"  ⚠️ Using full image {field} (more digits): {full_clean}")
                        merged[field] = full_clean
                        # Keep rejection info but let Qwen verify later
                        # 🚫 HARD STOP for masked Aadhaar
                        if full_reason == 'masked_aadhar' or cropped_reason == 'masked_aadhar':
                            merged['aadharnumber'] = full_value or cropped_value
                            merged['aadhar_status'] = 'aadhar_disapproved'
                            merged['aadhar_rejection_reason'] = 'masked_aadhar'
                            return merged  # ⛔ STOP HERE

                
                elif full_value and full_value not in ['Invalid Format', 'Not Detected', 'Other', '']:
                    logger.info(f"  ✓ Using full image {field}: {full_value}")
                    merged[field] = full_value
                    
                    if field == 'dob':
                        merged['age'] = full_data.get('age')
                        merged['age_status'] = full_data.get('age_status')
        
        return merged

    def extract_from_file(self, file_path: str, crop_coords: List[int] = None, 
                         confidence_threshold: float = 0.15, card_side: str = 'front'):
        """
        Main Entry Point with CARD ROTATION + 2 FALLBACKS + QWEN (with delayed masked rejection).
        
        MODIFIED LOGIC:
        1. Detect card rotation
        2. Try extraction with crop coordinates
        3. If critical fields missing/invalid, try full image (2nd fallback)
        4. If Aadhaar still invalid OR masked, try Qwen (3rd fallback) - GIVE QWEN A CHANCE
        5. ONLY reject masked Aadhaar if Qwen ALSO detects it as masked
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
            
            # 3. Apply card rotation
            if card_rotation != 0:
                logger.info(f"🔄 Applying card rotation: {card_rotation}°")
                if card_rotation == 90:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif card_rotation == 180:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_180)
                elif card_rotation == 270:
                    original_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)

            # === ATTEMPT 1: Extraction with CROP ===
            logger.info("=" * 60)
            logger.info("🔍 ATTEMPT 1: Extraction with CROP coordinates")
            logger.info("=" * 60)
            
            img = original_img.copy()
            
            if crop_coords:
                x1, y1, x2, y2 = crop_coords
                h, w = img.shape[:2]
                
                # Adjust crop for rotation
                if card_rotation == 90:
                    x1, y1, x2, y2 = y1, w - x2, y2, w - x1
                elif card_rotation == 180:
                    x1, y1, x2, y2 = w - x2, h - y2, w - x1, h - y1
                elif card_rotation == 270:
                    x1, y1, x2, y2 = h - y2, x1, h - y1, x2
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                logger.info(f"Cropping to: [{x1}, {y1}, {x2}, {y2}]")
                img = img[y1:y2, x1:x2]
            else:
                logger.info("No crop coordinates, using full image")

            # Run detection on cropped
            all_detections = self.detect_entities_in_image(img, confidence_threshold, card_side, session_dir)
            
            if not all_detections:
                logger.warning("⚠️ No entities in crop")
                cropped_result = {
                    "success": False,
                    "data": {"aadharnumber": "", "dob": "Not Detected", "gender": "Not Detected"}
                }
            else:
                self.crop_entities(all_detections, session_dir)
                ocr_results = self.perform_multi_language_ocr(all_detections)
                organized = self.organize_results_by_card_type(all_detections, ocr_results, confidence_threshold)
                cropped_data = self.extract_main_fields(organized)
                cropped_result = {"success": True, "data": cropped_data}
            
            # === CHECK CRITICAL FIELDS ===
            logger.info("\n" + "=" * 60)
            logger.info("🔍 Checking critical fields from CROP...")
            logger.info("=" * 60)
            
            missing_fields = self._check_critical_fields(cropped_result.get("data", {}), card_side)
            has_missing = any(missing_fields.values())
            
            image_for_qwen = img
            final_data = cropped_result.get("data", {})
            
            # === ATTEMPT 2: FALLBACK to Full Image ===
            if has_missing:
                logger.warning("⚠️ CRITICAL FIELDS MISSING/INVALID from crop:")
                for field, is_missing in missing_fields.items():
                    if is_missing:
                        logger.warning(f"  ❌ {field}")
                
                logger.info("\n" + "=" * 60)
                logger.info("🔄 ATTEMPT 2: FALLBACK - Full Image")
                logger.info("=" * 60)
                
                full_detections = self.detect_entities_in_image(
                    original_img, confidence_threshold, card_side, session_dir
                )
                
                if full_detections:
                    self.crop_entities(full_detections, None)
                    full_ocr_results = self.perform_multi_language_ocr(full_detections)
                    full_organized = self.organize_results_by_card_type(
                        full_detections, full_ocr_results, confidence_threshold
                    )
                    full_data = self.extract_main_fields(full_organized)
                    
                    logger.info("\n" + "=" * 60)
                    logger.info("🔀 MERGING crop + full image results...")
                    logger.info("=" * 60)
                    
                    # MODIFIED: Don't block merge for masked Aadhaar - let Qwen verify later
                    final_data = self._merge_results_without_mask_blocking(
                        cropped_result.get("data", {}), full_data, missing_fields
                    )
                    image_for_qwen = original_img
                else:
                    logger.error("❌ Fallback detection failed")
            else:
                logger.info("✅ All critical fields present in crop")
            
            # === RECHECK AFTER MERGE ===
            missing_after_merge = self._check_critical_fields(final_data, card_side)
            aadhaar_invalid = missing_after_merge.get('aadharnumber', False)
            
            # =====================================================================
            # === ATTEMPT 3: QWEN FALLBACK (MODIFIED LOGIC - NO EARLY REJECTION)
            # =====================================================================
            used_qwen_fallback = False
            qwen_detected_masked = False  # Track if Qwen also detects masked
            
            if self.enable_qwen_fallback and self.qwen_agent:
                aadhaar_value = final_data.get('aadharnumber', '')
                aadhaar_rejection = final_data.get('aadhar_rejection_reason')
                
                # NEW LOGIC: Allow Qwen to try even if masked or invalid detected
                # Only block if there's already a valid 12-digit Aadhaar
                has_valid_aadhaar = (
                    aadhaar_value and 
                    len(re.sub(r'\D', '', aadhaar_value)) == 12 and
                    aadhaar_rejection != 'masked_aadhar' and
                    aadhaar_rejection != 'invalid_length'
                )
                
                # Trigger Qwen if:
                # 1. Aadhaar is invalid (not 12 digits) - EVEN IF MASKED
                # 2. Aadhaar is masked (give Qwen a chance to find the real one)
                # 3. Aadhaar not detected at all
                # 4. Other critical fields are missing
                should_use_qwen = (
                    not has_valid_aadhaar and
                    aadhaar_rejection != 'masked_aadhar' and
                    (aadhaar_invalid or any(missing_after_merge.values()))
                )

                
                if should_use_qwen:
                    logger.info("\n" + "=" * 60)
                    logger.info("🤖 ATTEMPT 3: QWEN VLM FALLBACK")
                    logger.info("=" * 60)
                    
                    # Determine which fields need Qwen
                    qwen_fields = []
                    
                    # MODIFIED: Always include aadharnumber if it's invalid OR masked
                    if aadhaar_invalid or aadhaar_rejection in ['masked_aadhar', 'invalid_length', 'not_detected']:
                        qwen_fields.append('aadharnumber')
                        if aadhaar_rejection == 'masked_aadhar':
                            logger.warning("⚠️ Masked Aadhaar detected - giving Qwen a chance to find real number")
                    
                    # Add other missing fields
                    for field, is_missing in missing_after_merge.items():
                        if is_missing and field != 'aadharnumber':
                            qwen_fields.append(field)
                    
                    logger.info(f"Requesting Qwen for fields: {qwen_fields}")
                    
                    try:
                        qwen_results = self.qwen_agent.extract_fields(
                            image_for_qwen, 
                            qwen_fields, 
                            card_side
                        )
                        
                        if qwen_results:
                            # Validate Qwen Aadhaar result with STRICT checks
                            if 'aadharnumber' in qwen_results:
                                qwen_aadhar = qwen_results['aadharnumber']
                                logger.info(f"🔍 Qwen returned Aadhaar: '{qwen_aadhar[:100]}...'")
                                
                                # Step 1: Check for garbage
                                is_garbage = self._is_garbage_text(qwen_aadhar)
                                
                                if is_garbage:
                                    logger.error(f"🗑️ QWEN returned GARBAGE: IGNORING")
                                    qwen_results.pop('aadharnumber')
                                else:
                                    # Step 2: Check if Qwen ALSO detected masked Aadhaar
                                    qwen_is_masked = self._is_masked_aadhaar(qwen_aadhar)
                                    
                                    if qwen_is_masked:
                                        logger.error(f"🚫 QWEN DETECTED MASKED AADHAAR: '{qwen_aadhar}'")
                                        logger.error(f"🚫 IMMEDIATE REJECTION - Qwen confirmed masking")
                                        qwen_detected_masked = True
                                        # IMMEDIATE REJECTION - Set final data right now
                                        final_data['aadharnumber'] = qwen_aadhar
                                        final_data['aadhar_status'] = 'aadhar_disapproved'
                                        final_data['aadhar_rejection_reason'] = 'masked_aadhar'
                                        # Mark for immediate rejection
                                        qwen_results['aadharnumber'] = qwen_aadhar
                                        qwen_results['is_masked'] = True
                                        qwen_results['immediate_reject'] = True
                                    else:
                                        # Step 3: Validate normally
                                        is_valid, clean_aadhar, rejection = self._validate_aadhaar_number(qwen_aadhar)
                                        
                                        if is_valid:
                                            logger.info(f"✅ QWEN found VALID Aadhaar: {clean_aadhar}")
                                            qwen_results['aadharnumber'] = clean_aadhar
                                            qwen_results['is_valid'] = True
                                        else:
                                            logger.warning(f"⚠️ QWEN Aadhaar invalid: {clean_aadhar} (reason: {rejection})")
                                            # Keep whatever we got, might be better than nothing
                                            qwen_results['aadharnumber'] = clean_aadhar if clean_aadhar else qwen_aadhar
                            
                            # Merge Qwen results
                            for field, value in qwen_results.items():
                                if field == 'aadharnumber':
                                    if qwen_results.get('immediate_reject'):
                                        # IMMEDIATE REJECTION - Don't process further
                                        logger.error(f"  🚫 QWEN DETECTED MASKED - IMMEDIATE REJECTION")
                                        logger.error(f"  🚫 Stopping further processing")
                                        break  # Stop processing other fields
                                    elif qwen_results.get('is_valid'):
                                        # Qwen found a valid Aadhaar - override everything
                                        final_data['aadharnumber'] = value
                                        final_data['aadhar_status'] = 'aadhar_approved'
                                        if 'aadhar_rejection_reason' in final_data:
                                            del final_data['aadhar_rejection_reason']
                                        logger.info(f"  ✅ OVERRIDING with Qwen's VALID Aadhaar: {value}")
                                    elif qwen_results.get('is_masked'):
                                        # This shouldn't happen (immediate_reject should catch it)
                                        # But keep as safety net
                                        final_data['aadharnumber'] = value
                                        final_data['aadhar_status'] = 'aadhar_disapproved'
                                        final_data['aadhar_rejection_reason'] = 'masked_aadhar'
                                        logger.error(f"  🚫 Qwen CONFIRMED masked - REJECTING")
                                    else:
                                        # Qwen got something but not valid - use if better than current
                                        current_digits = len(re.sub(r'\D', '', final_data.get('aadharnumber', '')))
                                        qwen_digits = len(re.sub(r'\D', '', value))
                                        if qwen_digits > current_digits:
                                            final_data['aadharnumber'] = value
                                            logger.info(f"  ⚠️ Using Qwen's result (more digits): {value}")
                                elif value and value not in ['Not Detected', 'Invalid Format', '']:
                                    # Only process other fields if we didn't immediately reject
                                    if not qwen_results.get('immediate_reject'):
                                        final_data[field] = value
                                        logger.info(f"  ✓ Updated {field} from Qwen: {value}")
                            
                            used_qwen_fallback = True
                            logger.info("✅ Qwen fallback completed")
                        else:
                            logger.warning("⚠️ Qwen returned no results")
                            
                    except Exception as e:
                        logger.error(f"❌ Qwen fallback failed: {e}")
                        logger.error(traceback.format_exc())
            
            # =====================================================================
            # === FINAL VALIDATION (AFTER QWEN)
            # =====================================================================
            
            # If Qwen already rejected due to masked Aadhaar, skip further validation
            if qwen_detected_masked and final_data.get('aadhar_rejection_reason') == 'masked_aadhar':
                logger.error(f"🚫 QWEN IMMEDIATE REJECTION - Skipping further validation")
                logger.error(f"🚫 Final Status: REJECTED due to masked Aadhaar detected by Qwen")
            else:
                # Only do final validation if not already rejected by Qwen
                final_aadhar = final_data.get('aadharnumber', '')
                
                # Check if final result is garbage
                if final_aadhar and self._is_garbage_text(final_aadhar):
                    logger.error(f"🗑️ FINAL AADHAAR is GARBAGE - cleaning up")
                    digits_only = re.sub(r'\D', '', str(final_aadhar))
                    if len(digits_only) >= 8:
                        final_aadhar = digits_only
                        final_data['aadharnumber'] = digits_only
                    else:
                        final_aadhar = ""
                        final_data['aadharnumber'] = ""
                
                # NOW do final validation
                is_valid, clean_aadhar, rejection_reason = self._validate_aadhaar_number(final_aadhar)
                # 🚫 FINAL HARD BLOCK FOR MASKED AADHAAR
                if rejection_reason == 'masked_aadhar':
                    final_data['aadharnumber'] = clean_aadhar or final_aadhar
                    final_data['aadhar_status'] = 'aadhar_disapproved'
                    final_data['aadhar_rejection_reason'] = 'masked_aadhar'
                    logger.error("🚫 HARD REJECT: Masked Aadhaar detected")
                    return {
                        "success": True,
                        "data": final_data,
                        "card_rotation": card_rotation,
                        "used_qwen_fallback": False,
                        "qwen_detected_masked": False
                    }

                
                elif not is_valid:
                    logger.error(f"🚫 FINAL VALIDATION FAILED: {rejection_reason}")
                    if rejection_reason == 'masked_aadhar':
                        final_data['aadharnumber'] = clean_aadhar if clean_aadhar else final_aadhar
                    elif rejection_reason == 'invalid_length':
                        final_data['aadharnumber'] = clean_aadhar if clean_aadhar else ""
                    else:
                        final_data['aadharnumber'] = clean_aadhar if clean_aadhar else ""
                    
                    final_data['aadhar_status'] = 'aadhar_disapproved'
                    final_data['aadhar_rejection_reason'] = rejection_reason
                else:
                    logger.info(f"✅ FINAL VALIDATION PASSED: {clean_aadhar}")
                    final_data['aadharnumber'] = clean_aadhar
                    final_data['aadhar_status'] = 'aadhar_approved'
                    if 'aadhar_rejection_reason' in final_data:
                        del final_data['aadhar_rejection_reason']
            
            # === FINAL CLEANUP ===
            final_aadhar_value = final_data.get('aadharnumber', '')
            if final_aadhar_value:
                if len(str(final_aadhar_value)) > 30 or '\n' in str(final_aadhar_value):
                    logger.error(f"🗑️ Cleaning up garbage in final Aadhaar field")
                    digits_only = re.sub(r'\D', '', str(final_aadhar_value))
                    final_data['aadharnumber'] = digits_only if len(digits_only) <= 15 else ""
            
            # === SUMMARY ===
            logger.info("\n" + "=" * 60)
            logger.info("📋 FINAL EXTRACTION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Aadhaar Number: {final_data.get('aadharnumber', 'N/A')}")
            logger.info(f"Aadhaar Status: {final_data.get('aadhar_status', 'N/A')}")
            if final_data.get('aadhar_rejection_reason'):
                logger.info(f"Rejection Reason: {final_data.get('aadhar_rejection_reason')}")
            logger.info(f"DOB: {final_data.get('dob', 'N/A')}")
            logger.info(f"Age: {final_data.get('age', 'N/A')}")
            logger.info(f"Gender: {final_data.get('gender', 'N/A')}")
            logger.info(f"Card Rotation: {card_rotation}°")
            logger.info(f"Used Full Image Fallback: {has_missing}")
            logger.info(f"Used Qwen Fallback: {used_qwen_fallback}")
            logger.info(f"Qwen Detected Masked: {qwen_detected_masked}")
            logger.info("=" * 60)
            
            # === FINAL SANITIZATION CHECK ===
            if 'aadharnumber' in final_data:
                aadhar_val = str(final_data['aadharnumber'])
                if len(aadhar_val) > 20 or '\n' in aadhar_val or '\r' in aadhar_val:
                    logger.error(f"🚨 SANITIZATION: Removing garbage Aadhaar from final response")
                    final_data['aadharnumber'] = ""
                    if final_data.get('aadhar_status') == 'aadhar_approved':
                        final_data['aadhar_status'] = 'aadhar_disapproved'
                        final_data['aadhar_rejection_reason'] = 'not_detected'
            
            return {
                "success": True,
                "data": final_data,
                "debug_dir": str(session_dir) if session_dir else None,
                "card_rotation": card_rotation,
                "used_fallback": has_missing,
                "used_qwen_fallback": used_qwen_fallback,
                "qwen_detected_masked": qwen_detected_masked
            }

        except Exception as e:
            logger.error(f"Error in extraction: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def detect_entities_in_image(self, image_input, confidence_threshold: float, card_side: str = 'front', session_dir: Path = None):
        """Detect entities with side-specific filtering"""
        logger.info(f"\nStep 1: Detecting entities (Side: {card_side}, Threshold: {confidence_threshold})")
        
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
        
        if card_side.lower() == 'front':
            target_entities = {'aadharnumber', 'dob', 'gender', 'name', 'name_otherlang'}
        elif card_side.lower() == 'back':
            target_entities = {'aadharnumber', 'address', 'address_other_lang', 'pincode', 'mobile_no', 'city'}
        else:
            target_entities = {'aadharnumber', 'dob', 'gender'}
        
        logger.info(f"  Target entities: {target_entities}")
        
        results = self.model2(img, device=self.device, verbose=False)
        card_detections = []
        
        for box in results[0].boxes:
            if float(box.conf[0]) < confidence_threshold: 
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
            
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

        all_detections = {
            input_name: {
                "card_image": img,
                "card_type": card_side,
                "detections": card_detections
            }
        }
        
        return all_detections

    def crop_entities(self, all_detections: Dict[str, Dict[str, Any]], session_dir: Path = None):
        """Crop individual entities with bounds checking"""
        logger.info(f"\nStep 3: Cropping entities")
        
        for card_name, card_data in all_detections.items():
            img = card_data['card_image']
            h, w = img.shape[:2]
            
            for i, detection in enumerate(card_data['detections']):
                x1, y1, x2, y2 = detection['bbox']
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"  Invalid bbox for {detection['class_name']}")
                    detection['cropped_image'] = None
                    continue
                
                crop = img[y1:y2, x1:x2]
                detection['cropped_image'] = crop
                detection['entity_key'] = f"{card_name}_{detection['class_name']}_{i}"
        
        return all_detections
    
    def _preprocess_for_aadhaar_ocr(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Specialized preprocessing for Aadhaar numbers"""
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            h, w = gray.shape
            if h < 50:
                scale = 50 / h
                new_w, new_h = int(w * scale), int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            preprocessed_versions = []
            
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_versions.append(('otsu', thresh1))
            
            thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            preprocessed_versions.append(('adaptive', thresh2))
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_versions.append(('clahe', thresh3))
            
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, thresh4 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_versions.append(('denoised', thresh4))
            
            return preprocessed_versions
            
        except Exception as e:
            logger.warning(f"  Aadhaar preprocessing error: {e}")
            return [('original', img)]
    
    def _extract_aadhaar_with_multiple_methods(self, img: np.ndarray) -> str:
        """Try multiple OCR methods for Aadhaar - preserves X for mask detection"""
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
                                
                        except:
                            continue
            
            return best_result
            
        except Exception as e:
            logger.error(f"  Aadhaar extraction error: {e}")
            return ""
    
    def _correct_entity_orientation_and_preprocess(self, entity_image: np.ndarray, entity_key: str, class_name: str = None, osd_confidence_threshold: float = 0.5) -> Optional[Any]:
        """Enhanced preprocessing with Aadhaar special handling"""
        try:
            img = entity_image
            if img is None or img.size == 0:
                logger.warning(f"  Empty image for {entity_key}")
                return None
            
            if class_name == 'aadharnumber':
                logger.info(f"  Aadhaar preprocessing for {entity_key}")
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
            logger.error(f"   Orientation error for {entity_key}: {e}")
            return None

    def _detect_orientation_by_letters(self, img: np.ndarray, entity_key: str) -> Optional[int]:
        """Detect orientation by analyzing letter shapes"""
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
                
        except:
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
            
            return ocr_score * 0.5 + shape_score * 0.3 + line_score * 0.2
            
        except:
            return 0.0

    def _get_ocr_confidence_score(self, gray_img: np.ndarray) -> float:
        """Get OCR confidence"""
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
        """Analyze text lines"""
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
        """OCR with Aadhaar special handling"""
        logger.info(f"\nStep 4: Multi-Language OCR")
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
        """Organize results by card type"""
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
        """
        Extract and validate main fields with ULTRA STRICT masked Aadhaar rejection.
        NO COMPROMISES - masked Aadhaar is ALWAYS rejected.
        """
        fields = ['aadharnumber', 'dob', 'gender']
        data = {key: "" for key in fields}
        
        aadhar_front = ""
        aadhar_back = ""
        
        # Extract raw OCR text from both sides
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
        
        # === CRITICAL SECTION: AADHAAR VALIDATION ===
        logger.info("=" * 60)   
        logger.info("🔍 STRICT AADHAAR VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Front RAW: '{aadhar_front}'")
        logger.info(f"Back RAW: '{aadhar_back}'")
        
        # Check BOTH front and back for masking
        front_masked = self._is_masked_aadhaar(aadhar_front) if aadhar_front else False
        back_masked = self._is_masked_aadhaar(aadhar_back) if aadhar_back else False
        
        if front_masked or back_masked:
            masked_source = []
            if front_masked:
                masked_source.append('front')
            if back_masked:
                masked_source.append('back')
            
            logger.error(f"🚫 REJECTION: Masked Aadhaar in {', '.join(masked_source)}")
            
            # Use whichever value we have (even if masked) to show what was detected
            data['aadharnumber'] = aadhar_front if aadhar_front else aadhar_back
            data['aadhar_status'] = "aadhar_disapproved"
            data['aadhar_rejection_reason'] = "masked_aadhar"
            
            logger.error(f"🚫 FINAL: Rejected with value '{data['aadharnumber']}'")
        else:
            logger.info("✅ No masking detected, proceeding with validation")
            
            # Choose best Aadhaar between front and back
            best_aadhar = ""
            front_digits = re.sub(r'\D', '', aadhar_front) if aadhar_front else ""
            back_digits = re.sub(r'\D', '', aadhar_back) if aadhar_back else ""
            
            # Prefer matching values
            if front_digits == back_digits and front_digits:
                best_aadhar = front_digits
            # Prefer 12-digit values
            elif len(front_digits) == 12:
                best_aadhar = front_digits
            elif len(back_digits) == 12:
                best_aadhar = back_digits
            # Prefer longer values
            elif len(front_digits) > len(back_digits):
                best_aadhar = front_digits
            elif len(back_digits) > len(front_digits):
                best_aadhar = back_digits
            elif front_digits:
                best_aadhar = front_digits
            elif back_digits:
                best_aadhar = back_digits
            
            # Validate using strict method
            is_valid, clean_aadhar, rejection_reason = self._validate_aadhaar_number(best_aadhar)
            
            data['aadharnumber'] = clean_aadhar if clean_aadhar else best_aadhar
            
            if is_valid:
                data['aadhar_status'] = "aadhar_approved"
                logger.info(f"✅ APPROVED: {clean_aadhar}")
            else:
                data['aadhar_status'] = "aadhar_disapproved"
                data['aadhar_rejection_reason'] = rejection_reason
                logger.error(f"🚫 REJECTED: {rejection_reason} - '{data['aadharnumber']}'")
        
        logger.info("=" * 60)
        
        # === DOB Processing ===
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
        
        # === Gender Normalization ===
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