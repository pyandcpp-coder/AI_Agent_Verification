import logging
import math
import os
import re
import sys
import traceback
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image
from ultralytics import YOLO
from app.qwen_fallback import QwenFallbackAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class EntityAgent:
    def __init__(self, model1_path="models/best4.pt", model2_path="models/best.pt", other_lang_code='hin+tel+ben', enable_qwen_fallback=True, aadhaar_db_path="aadhaar_records.pkl"):
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA is available. Models will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("CUDA not available. Models will fall back to CPU.")

        # Duplicate checking setup
        self.aadhaar_db_path = aadhaar_db_path
        self.aadhaar_lock = threading.Lock()
        logger.info(f"Duplicate checking enabled with database: {self.aadhaar_db_path}")

        self.model1_path = model1_path
        self.model2_path = model2_path
        
        logger.info("Checking for YOLO models on filesystem...")
        if not Path(self.model1_path).exists():
             if Path(f"../{self.model1_path}").exists():
                 self.model1_path = f"../{self.model1_path}"
             else:
                 logger.warning(f"Model1 not found at {self.model1_path}. Assuming it will be available at runtime.")

        if not Path(self.model2_path).exists():
             if Path(f"../{self.model2_path}").exists():
                 self.model2_path = f"../{self.model2_path}"
             else:
                 logger.warning(f"Model2 not found at {self.model2_path}. Assuming it will be available at runtime.")

        logger.info("Loading models directly from filesystem...")
        try:
            self.model1 = YOLO(self.model1_path)
            self.model2 = YOLO(self.model2_path)
            
            # Clear cache after loading models
            clear_gpu_memory()
            logger.info("GPU memory cleared after model loading")
        except Exception as e:
            logger.error(f"Failed to load YOLO models: {e}")
            # Do not raise here to allow main.py to handle initialization errors if needed
            
        self.other_lang_code = other_lang_code

        self._check_tesseract()

        logger.info("YOLOv8 models loaded successfully.")

        self.enable_qwen_fallback = enable_qwen_fallback
        self.qwen_agent = None
        
        if self.enable_qwen_fallback:
            try:
                logger.info("Initializing Qwen3-VL fallback agent...")
                self.qwen_agent = QwenFallbackAgent()
                logger.info("Qwen3-VL fallback agent ready")
            except Exception as e:
                logger.warning(f"Could not initialize Qwen fallback: {e}")
                self.enable_qwen_fallback = False
        
        # Safely get names if models loaded
        if hasattr(self, 'model1'):
            self.card_classes = {i: name for i, name in self.model1.names.items()}
        if hasattr(self, 'model2'):
            self.entity_classes = {
                0: 'aadharnumber', 1: 'address', 2: 'address_other_lang', 3: 'city',
                4: 'dob', 5: 'gender', 6: 'gender_other_lang', 7: 'mobile_no',
                8: 'name', 9: 'name_otherlang', 10: 'pincode'
            }
        
        logger.info(f"Pipeline initialized to use '{self.other_lang_code}' for other language fields.")

    def _check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            logger.critical("Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.")
            raise RuntimeError("Tesseract not found")
        except Exception as e:
             logger.critical(f"An error occurred while checking Tesseract: {e}")
             raise RuntimeError(f"Error checking Tesseract: {e}")

    # ------------------ DUPLICATE CHECKING FUNCTIONS ------------------
    
    def _load_aadhaar_db(self):
        """Load existing Aadhaar records from pickle file"""
        if not os.path.exists(self.aadhaar_db_path):
            logger.info(f"No existing Aadhaar database found. Creating new one at {self.aadhaar_db_path}")
            return {}
        with open(self.aadhaar_db_path, "rb") as f:
            db = pickle.load(f)
            logger.info(f"Loaded {len(db)} records from Aadhaar database")
            return db

    def _save_aadhaar_db(self, db):
        """Save Aadhaar records to pickle file"""
        with open(self.aadhaar_db_path, "wb") as f:
            pickle.dump(db, f)
        logger.info(f"Saved {len(db)} records to Aadhaar database")

    def _check_and_save_aadhaar(self, aadharnumber, gender, dob):
        """
        Check if Aadhaar number exists in database.
        If duplicate, return existing record.
        If new, save to database.
        
        Returns:
          (is_duplicate: bool, existing_record: dict | None)
        """
        if not aadharnumber:
            logger.warning("Cannot check duplicate: Aadhaar number is empty")
            return False, None

        with self.aadhaar_lock:  # Thread-safe operation
            db = self._load_aadhaar_db()

            # Check if Aadhaar number already exists
            if aadharnumber in db:
                logger.warning(f"🚫 DUPLICATE AADHAAR DETECTED: {aadharnumber}")
                logger.warning(f"   Original submission: {db[aadharnumber]['timestamp']}")
                return True, db[aadharnumber]

            # Save new record
            db[aadharnumber] = {
                "aadharnumber": aadharnumber,
                "gender": gender,
                "dob": dob,
                "timestamp": datetime.now().isoformat()
            }
            self._save_aadhaar_db(db)
            logger.info(f"✅ New Aadhaar record saved: {aadharnumber}")

        return False, None

    # ------------------ VALIDATION FUNCTIONS ------------------

    def _is_garbage_text(self, text: str) -> bool:
        if not text: return True
        text_str = str(text).strip()
        if len(text_str) > 50 or '\n' in text_str: return True
        non_alphanum = sum(1 for c in text_str if not c.isalnum() and c not in [' ', '-'])
        if non_alphanum > 5: return True
        digits = sum(1 for c in text_str if c.isdigit())
        if len(text_str) > 0 and (digits / len(text_str) < 0.5): return True
        return False

    def _is_masked_aadhaar(self, text: str) -> bool:
        if not text: return False
        text_upper = str(text).upper()
        # Check for X's or asterisks
        if 'X' in text_upper or '*' in text_upper:
            logger.error(f"🚫 MASKED AADHAAR DETECTED: '{text}'")
            return True
        return False

    def _validate_aadhaar_number(self, text: str) -> tuple:
        if not text: return False, "", "not_detected"
        text_str = str(text).strip()
        
        if self._is_masked_aadhaar(text_str):
            return False, text_str, "masked_aadhar"
            
        if self._is_garbage_text(text_str):
            # Try to salvage digits
            digits_only = re.sub(r'\D', '', text_str)
            if len(digits_only) == 12:
                return True, digits_only, None
            return False, "", "invalid_format"
            
        digits_only = re.sub(r'\D', '', text_str)
        if len(digits_only) != 12:
            return False, digits_only, "invalid_length"
            
        return True, digits_only, None

    # ------------------ DETECTION AND PROCESSING METHODS ------------------

    def detect_and_crop_cards(self, image_paths: List[str], confidence_threshold: float) -> Dict[str, List[Dict[str, Any]]]:
        """Step 1: Detect Aadhaar front/back cards and pass cropped image data (np.array) forward."""
        logger.info(f"\nStep 1: Detecting Aadhaar cards in {len(image_paths)} images (Threshold: {confidence_threshold})")
        cropped_cards = {'front': [], 'back': []}
        for image_path in image_paths:
            logger.info(f"  Processing: {Path(image_path).name}")
            
            # Clear cache before processing each image
            clear_gpu_memory()
            
            results = self.model1(str(image_path), device=self.device, verbose=False)
            img = cv2.imread(str(image_path))
            input_filename = Path(image_path).stem
            detected = False
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.card_classes.get(int(box.cls[0]), "unknown")
                if class_name == 'print_aadhar':
                    raise ValueError("print_aadhar_detected")
                if class_name not in ['aadhar_front', 'aadhar_back']: continue
                detected = True
                crop = img[y1:y2, x1:x2]
                card_type = class_name.replace('aadhar_', '')
                card_data = {
                    "image": crop,
                    "name": f"{input_filename}_{class_name}_conf{int(float(box.conf[0])*100)}",
                    "type": card_type
                }
                cropped_cards[card_type].append(card_data)
                logger.info(f" Detected {class_name}")
            if not detected:
                logger.warning(f" No Aadhaar card detected in {Path(image_path).name}.")
        return cropped_cards

    def detect_entities_in_cards(self, cropped_cards: Dict[str, List[Dict[str, Any]]], confidence_threshold: float):
        """Step 2: Detect entities using in-memory card image data."""
        all_card_data = cropped_cards.get('front', []) + cropped_cards.get('back', [])
        logger.info(f"\nStep 2: Detecting entities in {len(all_card_data)} cards (Threshold: {confidence_threshold})")
        all_detections = {}

        for card_info in all_card_data:
            card_name = card_info['name']
            img = card_info['image']
            card_type = card_info['type']
            logger.info(f"  Processing: {card_name}")
            
            # Clear cache before inference to free memory
            clear_gpu_memory()
            
            results = self.model2(img, device=self.device, verbose=False)
            card_detections = []
            
            for box in results[0].boxes:
                if float(box.conf[0]) < confidence_threshold: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = self.entity_classes.get(int(box.cls[0]), "unknown")
                card_detections.append({'bbox': (x1, y1, x2, y2), 'class_name': class_name, 'confidence': float(box.conf[0])})
                
            logger.info(f"  Detected {len(card_detections)} entities in {card_name}")
            
            # Clear cache after each card to free memory
            clear_gpu_memory()
            all_detections[card_name] = {
                "card_image": img,
                "card_type": card_type,
                "detections": card_detections
            }
        return all_detections

    def crop_entities(self, all_detections: Dict[str, Dict[str, Any]]):
        """Step 3: Crop individual entities and add their image data to the detection dictionary."""
        logger.info(f"\nStep 3: Cropping individual entities")
        for card_name, card_data in all_detections.items():
            img = card_data['card_image']
            for i, detection in enumerate(card_data['detections']):
                x1, y1, x2, y2 = detection['bbox']
                
                # Bounds checking
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crop = img[y1:y2, x1:x2]
                detection['cropped_image'] = crop
                entity_key = f"{card_name}_{detection['class_name']}_{i}"
                detection['entity_key'] = entity_key
                logger.info(f"   Cropped entity: {entity_key}")
        return all_detections
    
    def _correct_entity_orientation_and_preprocess(self, entity_image: np.ndarray, entity_key: str, osd_confidence_threshold: float = 0.5) -> Optional[Image.Image]:
        try:
            img = entity_image
            if img is None or img.size == 0:
                logger.warning(f"  Entity image data for {entity_key} is empty, skipping.")
                return None
            
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
                    logger.warning(f" Both letter-based and OSD failed for {entity_key}. Assuming 0° rotation. Details: {e}")
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

            pil_img = Image.fromarray(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY))
            return pil_img
            
        except Exception as e:
            logger.error(f"   Unhandled error during entity orientation/preprocessing for {entity_key}: {e}")
            return None

    def _detect_orientation_by_letters(self, img: np.ndarray, entity_key: str) -> Optional[int]:
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
            logger.debug(f"Error calculating orientation score: {e}")
            return 0.0

    def _get_ocr_confidence_score(self, gray_img: np.ndarray) -> float:
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
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return 0.0
            
            upright_score = 0.0
            valid_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20: continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 5 or h < 5 or w > gray_img.shape[1] * 0.8 or h > gray_img.shape[0] * 0.8: continue
                
                aspect_ratio = h / w
                if 0.3 <= aspect_ratio <= 4.0:
                    valid_contours += 1
                    if 1.0 <= aspect_ratio <= 2.5: upright_score += 1.0
                    elif 0.5 <= aspect_ratio <= 3.5: upright_score += 0.7
                    else: upright_score += 0.3
            
            if valid_contours == 0: return 0.0
            return min(upright_score / valid_contours, 1.0)
            
        except Exception:
            return 0.0

    def _analyze_text_lines(self, gray_img: np.ndarray) -> float:
        try:
            _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            horizontal_pixels = cv2.countNonZero(horizontal_lines)
            vertical_pixels = cv2.countNonZero(vertical_lines)
            
            total_pixels = horizontal_pixels + vertical_pixels
            if total_pixels == 0: return 0.5
            
            horizontal_ratio = horizontal_pixels / total_pixels
            return horizontal_ratio
            
        except Exception:
            return 0.5

    def perform_multi_language_ocr(self, all_detections: Dict[str, Dict[str, Any]]):
        logger.info(f"\nStep 4: Correcting Entity Orientation & Performing Multi-Language OCR")
        # Only process these critical fields
        required_fields = {'aadharnumber', 'dob', 'gender', 'gender_other_lang'}
        ocr_results = {}
        processed_count = 0
        skipped_count = 0
        
        for card_name, card_data in all_detections.items():
            for detection in card_data['detections']:
                cropped_image = detection.get('cropped_image')
                entity_key = detection.get('entity_key')
                class_name = detection.get('class_name')

                if cropped_image is None or entity_key is None: 
                    continue
                
                # Skip processing if not a required field
                if class_name not in required_fields:
                    logger.debug(f"  Skipping non-essential entity: {entity_key} (Class: {class_name})")
                    skipped_count += 1
                    continue

                logger.info(f"  Processing entity: {entity_key} (Class: {class_name})")
                lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                
                processed_pil_img = self._correct_entity_orientation_and_preprocess(cropped_image, entity_key)

                if processed_pil_img:
                    try:
                        text = pytesseract.image_to_string(processed_pil_img, lang=lang_to_use, config='--psm 6')
                        extracted_text = ' '.join(text.split()).strip()
                        ocr_results[entity_key] = extracted_text
                        processed_count += 1
                    except Exception as e:
                        logger.error(f" OCR failed for {entity_key}: {e}")
                        ocr_results[entity_key] = None
        
        logger.info(f"  OCR Summary: Processed {processed_count} entities, Skipped {skipped_count} non-essential entities")
        return ocr_results

    def organize_results_by_card_type(self, all_detections: Dict[str, Dict[str, Any]], ocr_results: Dict[str, str], confidence_threshold: float):
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
            
    def process_images(self, image_paths, user_id: str, task_id: str, confidence_threshold: float, verbose=True):
        """Main pipeline function to process images in-memory without saving any intermediate or final files."""
        try:
            if verbose:
                logger.info(f"[user_id={user_id}] Starting In-Memory Pipeline for task {task_id}")

            # Step 1: Detect and Crop Cards
            cropped_cards = self.detect_and_crop_cards(image_paths, confidence_threshold)
            if not cropped_cards.get('front', []) and not cropped_cards.get('back', []):
                logger.warning(f"[user_id={user_id}] No Aadhaar card detected in any provided images.")
                return {'error': 'no_aadhar_detected', 'success': False}

            # Step 2: Detect Entities
            all_detections = self.detect_entities_in_cards(cropped_cards, confidence_threshold)

            # Step 3: Crop Entities
            self.crop_entities(all_detections)

            # Step 4: OCR
            ocr_results = self.perform_multi_language_ocr(all_detections)

            # Step 5: Organize
            organized_results = self.organize_results_by_card_type(all_detections, ocr_results, confidence_threshold)

            # Step 6: Extract Main Fields
            final_data = self.extract_main_fields(organized_results)

            # --- QWEN FALLBACK LOGIC START ---
            if self.enable_qwen_fallback and self.qwen_agent:
                # 1. Validate Aadhaar
                aadhaar = final_data.get('aadharnumber', '')
                is_valid_aadhaar, clean_aadhaar, rejection_reason = self._validate_aadhaar_number(aadhaar)
                
                # 2. Check Masking - Immediate Rejection
                if rejection_reason == 'masked_aadhar':
                    final_data['aadhar_status'] = 'masked_aadhar'
                    logger.info("🚫 Masked Aadhaar detected in OCR - Skipping Qwen to avoid false positives")
                
                else:
                    # 3. Determine if Qwen is needed
                    # We need Qwen if: Aadhaar is invalid OR DOB missing OR Gender missing
                    missing_fields = []
                    if not is_valid_aadhaar: missing_fields.append('aadharnumber')
                    
                    dob = final_data.get('dob', '')
                    if dob in ['Invalid Format', 'Not Detected', '', None]: missing_fields.append('dob')
                    
                    gender = final_data.get('gender', '')
                    if gender in ['Other', 'Not Detected', '', None]: missing_fields.append('gender')

                    if missing_fields:
                        logger.info(f"🔄 Triggering Qwen Fallback for: {missing_fields}")
                        # Use the first image (Front) for Qwen
                        front_image_path = image_paths[0] if image_paths else None
                        
                        if front_image_path:
                            try:
                                qwen_results = self.qwen_agent.extract_fields(front_image_path, missing_fields, 'front')
                                
                                # Merge Qwen Results
                                if 'aadharnumber' in qwen_results:
                                    q_aadhaar = qwen_results['aadharnumber']
                                    if self._is_masked_aadhaar(q_aadhaar):
                                        final_data['aadhar_status'] = 'masked_aadhar'
                                        final_data['aadharnumber'] = q_aadhaar
                                    else:
                                        # Validate Qwen Aadhaar
                                        q_valid, q_clean, _ = self._validate_aadhaar_number(q_aadhaar)
                                        if q_valid:
                                            final_data['aadharnumber'] = q_clean
                                            # If OCR failed but Qwen succeeded, status is good
                                            if final_data.get('aadhar_status') != 'masked_aadhar':
                                                final_data['aadhar_status'] = 'extracted'

                                if 'dob' in qwen_results and qwen_results['dob']:
                                    final_data['dob'] = qwen_results['dob']
                                
                                if 'gender' in qwen_results and qwen_results['gender']:
                                    final_data['gender'] = qwen_results['gender']
                                    
                            except Exception as e:
                                logger.error(f"❌ Qwen Fallback Failed: {e}")

            # --- DUPLICATE CHECK START ---
            # Only check duplicates if we have a valid Aadhaar number
            aadhaar_num = final_data.get('aadharnumber', '')
            if aadhaar_num and final_data.get('aadhar_status') != 'masked_aadhar':
                is_duplicate, existing_record = self._check_and_save_aadhaar(
                    aadhaar_num,
                    final_data.get('gender', ''),
                    final_data.get('dob', '')
                )
                
                if is_duplicate:
                    # REJECT THE SUBMISSION - Return error response
                    logger.error(f"❌ SUBMISSION REJECTED: Duplicate Aadhaar detected - {aadhaar_num}")
                    return {
                        'error': 'duplicate_aadhar_detected',
                        'success': False,
                        'duplicate': True,
                        'aadharnumber': aadhaar_num,
                        'existing_record': existing_record,
                        'message': f'This Aadhaar number ({aadhaar_num}) has already been submitted on {existing_record["timestamp"]}. Duplicate submissions are not allowed.'
                    }
                else:
                    # New record - mark as accepted
                    final_data['duplicate'] = False
                    final_data['existing_record'] = None
                    logger.info(f"✅ Submission accepted: New Aadhaar number")
            else:
                final_data['duplicate'] = False
                final_data['existing_record'] = None
            # --- DUPLICATE CHECK END ---

            if verbose:
                logger.info(f"[user_id={user_id}] Pipeline processing completed successfully in memory.")
            
            return {'organized_results': organized_results, 'data': final_data, 'success': True}

        except ValueError as ve:
            logger.error(f"[user_id={user_id}] SECURITY ERROR in pipeline: {ve}")
            return {'error': str(ve), 'security_flagged': True, 'step': 'card_detection', 'success': False}
        except Exception as e:
            logger.error(f"[user_id={user_id}] Unhandled error in pipeline: {e}\n{traceback.format_exc()}")
            return {'error': str(e), 'traceback': traceback.format_exc(), 'step': 'unknown', 'success': False}

    def extract_main_fields(self, organized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts only Aadhaar Number, DOB, and Gender."""
        fields = ['aadharnumber', 'dob', 'gender']
        data = {key: "" for key in fields}
        
        for side in ['front', 'back']:
            for card in organized_results.get(side, {}).values():
                for field in fields:
                    if field in card['entities'] and card['entities'][field]:
                        all_texts = [item.get('extracted_text', '') for item in card['entities'][field]]
                        first_valid_text = next((text for text in all_texts if text), '')
                        if first_valid_text:
                            data[field] = first_valid_text
                            
        if data.get('aadharnumber'):
            aad = data['aadharnumber']
            # detect masked aadhaar
            if re.search(r'X{4}', aad, re.IGNORECASE):
                data['aadhar_status'] = "masked_aadhar" 
            
            data['aadharnumber'] = aad.replace(" ", "")
        
        if data.get('dob'):
            digit_groups = re.findall(r'\d+', data['dob'])
            digits_only = ''.join(digit_groups)
            if len(digits_only) == 8:
                try:
                    parsed_date = datetime.strptime(digits_only, '%d%m%Y')
                    data['dob'] = parsed_date.strftime('%d-%m-%Y')
                    # Calculate age
                    current_year = datetime.now().year
                    birth_year = parsed_date.year
                    age = current_year - birth_year
                    # Adjust if birthday hasn't occurred this year
                    if datetime.now().month < parsed_date.month or \
                       (datetime.now().month == parsed_date.month and datetime.now().day < parsed_date.day):
                        age -= 1
                    data['age'] = age
                    data['age_status'] = 'age_approved' if age >= 18 else 'age_disapproved'
                except ValueError:
                    data['dob'] = 'Invalid Format'
                    data['age_status'] = 'age_disapproved'
            else:
                year = next((g for g in digit_groups if len(g) == 4), None)
                if year:
                    data['dob'] = year
                    # Calculate age from year only
                    try:
                        birth_year = int(year)
                        current_year = datetime.now().year
                        age = current_year - birth_year
                        data['age'] = age
                        data['age_status'] = 'age_approved' if age >= 18 else 'age_disapproved'
                    except ValueError:
                        data['age_status'] = 'age_disapproved'
                else:
                    data['dob'] = 'Invalid Format'
                    data['age_status'] = 'age_disapproved'
        else:
            data['age_status'] = 'age_disapproved'
        
        # Gender normalization
        if data['gender']:
            gender = data['gender'].strip().lower()
            if gender == 'male':
                data['gender'] = 'Male'
            elif gender == 'female':
                data['gender'] = 'Female'
            else:
                data['gender'] = 'Other'
            
        return data

if __name__ == "__main__":
    try:
        agent = EntityAgent(model1_path="models/best4.pt", model2_path="models/best.pt")
        print("Entity Agent initialized successfully")
    except Exception as e:
        print(f"Initialization failed: {e}")