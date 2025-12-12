import logging
import re
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

logger = logging.getLogger(__name__)

class QwenFallbackAgent:
    """
    Third-level fallback using Qwen3-VL model for critical field extraction.
    Only used when OCR fails to extract valid aadharnumber, gender, or DOB year.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initialize Qwen3-VL model for fallback extraction.
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info("=" * 60)
        logger.info("🚀 Initializing Qwen3-VL Fallback Agent")
        logger.info("=" * 60)
        
        try:
            # Check device availability
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"✓ CUDA available - Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("⚠️ CUDA not available - Using CPU (this will be slower)")
            
            # Load model and processor
            logger.info(f"📥 Loading model: {model_name}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("✅ Qwen3-VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qwen3-VL model: {e}")
            raise RuntimeError(f"Qwen3-VL initialization failed: {e}")
    
    def should_use_fallback(self, data: Dict[str, Any]) -> Tuple[bool, list]:
        """
        Determine if Qwen fallback should be used based on data quality.
        
        Returns:
            Tuple of (should_use_fallback, list_of_missing_fields)
        """
        missing_fields = []
        
        # Check Aadhaar number validity
        aadhaar = data.get('aadharnumber', '')
        aadhaar_digits = re.sub(r'\D', '', str(aadhaar))
        
        if len(aadhaar_digits) != 12:
            missing_fields.append('aadharnumber')
            logger.info(f"  ⚠️ Aadhaar invalid: {len(aadhaar_digits)} digits (need 12)")
        
        # Check gender validity
        gender = data.get('gender', '').strip()
        if gender in ['Other', 'Not Detected', '', None]:
            missing_fields.append('gender')
            logger.info(f"  ⚠️ Gender invalid: '{gender}'")
        
        # Check DOB year validity
        dob = data.get('dob', '')
        if dob in ['Invalid Format', 'Not Detected', '', None]:
            missing_fields.append('dob')
            logger.info(f"  ⚠️ DOB invalid: '{dob}'")
        elif isinstance(dob, str):
            # Check if it contains a valid 4-digit year
            year_match = re.search(r'\b(19|20)\d{2}\b', dob)
            if not year_match:
                missing_fields.append('dob')
                logger.info(f"  ⚠️ DOB missing valid year: '{dob}'")
        
        should_use = len(missing_fields) > 0
        
        if should_use:
            logger.info(f"🔄 Qwen fallback needed for: {missing_fields}")
        
        return should_use, missing_fields
    
    def _prepare_image(self, image_input) -> Image.Image:
        """
        Convert various image formats to PIL Image.
        """
        if isinstance(image_input, str):
            # File path
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # OpenCV/numpy array
            if len(image_input.shape) == 2:
                # Grayscale
                return Image.fromarray(image_input).convert('RGB')
            else:
                # BGR to RGB conversion for OpenCV images
                return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def extract_fields(self, image_input, missing_fields: list, card_side: str = 'front') -> Dict[str, Any]:
        """
        Extract missing fields using Qwen3-VL model.
        
        Args:
            image_input: Image in various formats (path, numpy array, PIL Image)
            missing_fields: List of fields to extract ['aadharnumber', 'gender', 'dob']
            card_side: 'front' or 'back'
        
        Returns:
            Dictionary with extracted fields
        """
        logger.info("\n" + "=" * 60)
        logger.info("🤖 Qwen3-VL Fallback Extraction")
        logger.info("=" * 60)
        logger.info(f"Missing fields to extract: {missing_fields}")
        logger.info(f"Card side: {card_side}")
        
        try:
            # Prepare image
            image = self._prepare_image(image_input)
            
            # Build prompt based on missing fields
            prompt = self._build_extraction_prompt(missing_fields, card_side)
            
            logger.info(f"📝 Prompt: {prompt}")
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            logger.info("🔄 Generating extraction...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,  # Low temperature for factual extraction
                    do_sample=False
                )
            
            # Trim and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"📤 Raw model output: {output_text}")
            
            # Parse the output
            extracted_data = self._parse_model_output(output_text, missing_fields)
            
            logger.info("✅ Qwen extraction complete:")
            for field, value in extracted_data.items():
                logger.info(f"  {field}: {value}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Error in Qwen extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def _build_extraction_prompt(self, missing_fields: list, card_side: str) -> str:
        """
        Build a precise extraction prompt for the VLM model.
        """
        prompts = []
        
        if 'aadharnumber' in missing_fields:
            prompts.append(
                "Extract the 12-digit Aadhaar number. "
                "It may be formatted as XXXX XXXX XXXX or XXXX-XXXX-XXXX. "
                "Return ONLY the 12 digits, no spaces or dashes."
            )
        
        if 'gender' in missing_fields and card_side == 'front':
            prompts.append(
                "Extract the gender/sex field. "
                "Look for words like 'Male', 'Female', 'MALE', 'FEMALE', 'पुरुष', 'महिला'. "
                "Return ONLY: 'Male' or 'Female'."
            )
        
        if 'dob' in missing_fields and card_side == 'front':
            prompts.append(
                "Extract the date of birth (DOB) or birth year. "
                "Look for formats like DD/MM/YYYY, DD-MM-YYYY, or just YYYY. "
                "If full date is found, return in format: YYYY. "
                "If only year is visible, return just the 4-digit year."
            )
        
        base_prompt = "You are analyzing an Indian Aadhaar card image. Extract ONLY the following information:\n\n"
        base_prompt += "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
        base_prompt += "\n\nFormat your response EXACTLY as:\n"
        
        if 'aadharnumber' in missing_fields:
            base_prompt += "AADHAAR: [12 digits]\n"
        if 'gender' in missing_fields:
            base_prompt += "GENDER: [Male/Female]\n"
        if 'dob' in missing_fields:
            base_prompt += "DOB: [YYYY]\n"
        
        base_prompt += "\nIf you cannot find a field, write 'NOT_FOUND' for that field."
        
        return base_prompt
    
    def _parse_model_output(self, output_text: str, missing_fields: list) -> Dict[str, Any]:
        """
        Parse the VLM model output into structured data.
        """
        result = {}
        
        # Normalize output text
        output_upper = output_text.upper()
        
        if 'aadharnumber' in missing_fields:
            # Look for AADHAAR: pattern
            aadhaar_match = re.search(r'AADHAAR[:\s]+([0-9X\s-]{10,20})', output_upper)
            if aadhaar_match:
                aadhaar_raw = aadhaar_match.group(1)
                # Extract only digits (ignore X's which indicate masking)
                aadhaar_digits = re.sub(r'\D', '', aadhaar_raw)
                
                # Check for masking
                if 'X' in aadhaar_raw.upper():
                    logger.warning("⚠️ Qwen detected masked Aadhaar")
                    result['aadharnumber'] = aadhaar_raw  # Keep masked format for rejection
                    result['is_masked'] = True
                elif len(aadhaar_digits) == 12:
                    result['aadharnumber'] = aadhaar_digits
                    result['is_masked'] = False
                else:
                    logger.warning(f"⚠️ Invalid Aadhaar length from Qwen: {len(aadhaar_digits)}")
            else:
                # Try to find any 12-digit sequence
                digits = re.findall(r'\d{12}', output_text)
                if digits:
                    result['aadharnumber'] = digits[0]
                    result['is_masked'] = False
        
        if 'gender' in missing_fields:
            # Look for GENDER: pattern
            gender_match = re.search(r'GENDER[:\s]+(MALE|FEMALE|M|F)', output_upper)
            if gender_match:
                gender_raw = gender_match.group(1)
                if gender_raw in ['MALE', 'M']:
                    result['gender'] = 'Male'
                elif gender_raw in ['FEMALE', 'F']:
                    result['gender'] = 'Female'
            else:
                # Try to find gender keywords anywhere in output
                if 'MALE' in output_upper and 'FEMALE' not in output_upper:
                    result['gender'] = 'Male'
                elif 'FEMALE' in output_upper:
                    result['gender'] = 'Female'
        
        if 'dob' in missing_fields:
            # Look for DOB: pattern with year
            dob_match = re.search(r'DOB[:\s]+.*?(\d{4})', output_text)
            if dob_match:
                year = dob_match.group(1)
                year_int = int(year)
                current_year = datetime.now().year
                
                # Validate year is reasonable (1900-current year)
                if 1900 <= year_int <= current_year:
                    result['dob'] = year
                    # Calculate age
                    result['age'] = current_year - year_int
                else:
                    logger.warning(f"⚠️ Invalid year from Qwen: {year}")
            else:
                # Try to find any 4-digit year
                years = re.findall(r'\b(19|20)\d{2}\b', output_text)
                if years:
                    year_int = int(years[0])
                    current_year = datetime.now().year
                    if 1900 <= year_int <= current_year:
                        result['dob'] = years[0]
                        result['age'] = current_year - year_int
        
        return result
    
    def validate_and_merge(self, original_data: Dict[str, Any], 
                          qwen_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge Qwen results with original data, preferring valid Qwen results.
        
        Args:
            original_data: Original extraction results
            qwen_data: Results from Qwen fallback
        
        Returns:
            Merged data dictionary
        """
        merged = original_data.copy()
        
        logger.info("\n" + "=" * 60)
        logger.info("🔀 Merging Qwen results with original data")
        logger.info("=" * 60)
        
        # Merge Aadhaar number
        if 'aadharnumber' in qwen_data:
            qwen_aadhaar = qwen_data['aadharnumber']
            is_masked = qwen_data.get('is_masked', False)
            
            if is_masked:
                logger.warning("❌ Qwen detected masked Aadhaar - REJECTING")
                merged['aadharnumber'] = qwen_aadhaar
                merged['aadhar_status'] = 'aadhar_disapproved'
                merged['aadhar_rejection_reason'] = 'masked_aadhar_qwen'
            else:
                qwen_digits = re.sub(r'\D', '', str(qwen_aadhaar))
                if len(qwen_digits) == 12:
                    logger.info(f"✅ Using Qwen Aadhaar: {qwen_digits}")
                    merged['aadharnumber'] = qwen_digits
                    merged['aadhar_status'] = 'aadhar_approved'
                    if 'aadhar_rejection_reason' in merged:
                        del merged['aadhar_rejection_reason']
        
        # Merge gender
        if 'gender' in qwen_data:
            qwen_gender = qwen_data['gender']
            if qwen_gender in ['Male', 'Female']:
                logger.info(f"✅ Using Qwen gender: {qwen_gender}")
                merged['gender'] = qwen_gender
        
        # Merge DOB/age
        if 'dob' in qwen_data:
            logger.info(f"✅ Using Qwen DOB: {qwen_data['dob']}")
            merged['dob'] = qwen_data['dob']
            if 'age' in qwen_data:
                merged['age'] = qwen_data['age']
                # Update age status
                if qwen_data['age'] >= 18:
                    merged['age_status'] = 'age_approved'
                    logger.info(f"✅ Age approved: {qwen_data['age']} years")
                else:
                    merged['age_status'] = 'age_disapproved'
                    logger.info(f"❌ Age rejected: {qwen_data['age']} years")
        
        return merged