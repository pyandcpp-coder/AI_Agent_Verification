import logging
import math 
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    def __init__(self, model_path="models/best.pt", other_lang_code='hin+tel+ben'):

        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"CUDA is available. Models will run on GPU ({torch.cuda.get_device_name(0)}).")
        else:
            self.device = "cpu"
            logger.info("CUDA not available. Models will fall back to CPU.")

        # --- Model Loading ---
        self.entity_model_path = model_path
        
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

    def extract_from_file(self, file_path: str, crop_coords: List[int] = None, confidence_threshold: float = 0.15):
        """
        Main Entry Point for the Orchestrator.
        1. Reads the file from 'file_path'.
        2. Crops it using 'crop_coords' [x1, y1, x2, y2].
        3. Runs your original logic on the crop.
        """
        try:
            # 1. Read Image
            img = cv2.imread(file_path)
            if img is None:
                logger.error(f"Failed to read image: {file_path}")
                return {"error": "failed_to_read_file"}

            # 2. Crop (If coords provided)
            if crop_coords:
                x1, y1, x2, y2 = crop_coords
                # Basic bounds checking
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                logger.info(f"Cropping image to: {crop_coords}")
                img = img[y1:y2, x1:x2]

            # 3. Detect Entities (Pass the numpy array directly)
            all_detections = self.detect_entities_in_image(img, confidence_threshold)
            
            if not all_detections:
                return {"success": False, "message": "no_entities_detected"}

            # 4. Crop & OCR
            self.crop_entities(all_detections)
            ocr_results = self.perform_multi_language_ocr(all_detections)
            
            # 5. Organize
            organized = self.organize_results_by_card_type(all_detections, ocr_results, confidence_threshold)
            
            # 6. Extract Main Fields & Validate
            final_data = self.extract_main_fields(organized)
            
            return {
                "success": True,
                "data": final_data
            }

        except Exception as e:
            logger.error(f"Error in extraction: {e}\n{traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    def detect_entities_in_image(self, image_input, confidence_threshold: float):
        """
        Modified slightly to accept either a Path (str) or a Numpy Array (cropped image).
        """
        logger.info(f"\nStep 1: Detecting entities in image (Threshold: {confidence_threshold})")
        
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
        
        # Only process these 3 entity types
        target_entities = {'aadharnumber', 'dob', 'gender'}
        
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

        # Wrap in your original structure
        all_detections = {
            input_name: {
                "card_image": img,
                "card_type": "front",
                "detections": card_detections
            }
        }
        
        return all_detections

    def crop_entities(self, all_detections: Dict[str, Dict[str, Any]]):
        """Step 3: Crop individual entities"""
        logger.info(f"\nStep 3: Cropping individual entities")
        for card_name, card_data in all_detections.items():
            img = card_data['card_image']
            for i, detection in enumerate(card_data['detections']):
                x1, y1, x2, y2 = detection['bbox']
                crop = img[y1:y2, x1:x2]
                detection['cropped_image'] = crop
                entity_key = f"{card_name}_{detection['class_name']}_{i}"
                detection['entity_key'] = entity_key
        return all_detections
    
    def _correct_entity_orientation_and_preprocess(self, entity_image: np.ndarray, entity_key: str, osd_confidence_threshold: float = 0.5) -> Optional[Any]:
        """
        Enhanced preprocessing and orientation correction with improved logging and error handling.
        """
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
        Step 4: Correcting orientation and perform OCR on in-memory entity images.
        Uses English for standard fields and other languages for '_other_lang' fields.
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
                lang_to_use = self.other_lang_code if class_name and class_name.endswith('_other_lang') else 'eng'
                
                processed_pil_img = self._correct_entity_orientation_and_preprocess(cropped_image, entity_key)

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
        print(" Entity Agent initialized successfully ")
    except Exception as e:
        print(f"Initialization failed: {e}")

