import re
from datetime import datetime

class VerificationScorer:
    def __init__(self):
        # Weights (Total 100) - Original Configuration
        self.weights = {
            "face": 20,      # Face Similarity (reduced from 40)
            "aadhar": 30,    # Valid Aadhaar Number (increased from 20)
            "dob": 30,       # Valid Age/DOB (increased from 20)
            "gender": 20     # Gender Match (unchanged)
        }
    
    def _normalize_dob(self, dob_string):
        """Standardizes date formats (dd-mm-yyyy, yyyy-mm-dd) to yyyy-mm-dd"""
        if not dob_string: return None
        formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d.%m.%Y', '%Y/%m/%d']
        for fmt in formats:
            try:
                return datetime.strptime(str(dob_string).strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    def calculate_score(self, face_data, entity_data, expected_gender, expected_dob=None):
        score = 0
        breakdown = {}
        rejection_reasons = []
        critical_failure = False  # If True, status is REJECTED regardless of score

        # --- 1. Face Similarity (0 - 20 points) ---
        face_sim = face_data.get("score", 0)
        low_face_sim = False
        
        if face_sim < 5:
            # Very low face similarity - Give 0 points and flag for review
            breakdown["face_score"] = 0
            low_face_sim = True
            rejection_reasons.append(f"Low Face Similarity (Similarity: {face_sim:.2f}%)")
        else:
            # Between 5% and 100% -> Scale linearly to 0-20 points
            # Formula: ((Score - 5) / 95) * 20, where 95 is the range (100-5)
            face_points = ((face_sim - 5) / 95) * self.weights["face"]
            score += face_points
            breakdown["face_score"] = round(face_points, 2)

        # --- 2. Aadhaar Validity & Masking (0 - 20 points) ---
        aadhaar_num = entity_data.get("aadharnumber", "").replace(" ", "")
        
        # Check for Masking
        if "X" in entity_data.get("aadharnumber", "").upper():
            # Masked Aadhaar is generally acceptable, but can be flagged if needed
            pass 
            
        if len(aadhaar_num) == 12 and aadhaar_num.isdigit():
            score += self.weights["aadhar"]
            breakdown["aadhar_score"] = self.weights["aadhar"]
        else:
            breakdown["aadhar_score"] = 0
            rejection_reasons.append("Invalid/Unreadable Aadhaar Number")

        # --- 3. DOB & Age Check (0 - 20 points) ---
        extracted_dob = entity_data.get("dob", "")
        age_status = entity_data.get("age_status") # Calculated in entity.py
        
        dob_score = 0
        # Rule A: Must be 18+
        if age_status == "age_approved":
            # Rule B: If Input DOB provided, it MUST match Aadhaar DOB
            if expected_dob and extracted_dob:
                norm_input = self._normalize_dob(expected_dob)
                norm_extracted = self._normalize_dob(extracted_dob)
                
                if norm_input and norm_extracted and norm_input == norm_extracted:
                    dob_score = self.weights["dob"]
                elif norm_input and norm_extracted:
                    # DOB Mismatch -> Critical Failure
                    dob_score = 0
                    critical_failure = True
                    rejection_reasons.append(f"DOB Mismatch (Input: {expected_dob} vs Aadhaar: {extracted_dob})")
                else:
                    # Format error but Age approved -> Pass
                    dob_score = self.weights["dob"]
            else:
                dob_score = self.weights["dob"]
        else:
            # Under 18 or Invalid DOB -> Critical Failure
            dob_score = 0
            critical_failure = True
            rejection_reasons.append("User is Under 18 or DOB Unreadable")
            
        score += dob_score
        breakdown["dob_score"] = dob_score

        # --- 4. Gender Check (0 - 20 points + Bonus) ---
        # Normalize Helper
        def clean_gender(g):
            if not g: return "unknown"
            g = g.lower()
            if "female" in g: return "female"
            if "male" in g: return "male"
            return "other"

        extracted_gen = entity_data.get("gender", "Other")
        input_gen = expected_gender if expected_gender else ""
        
        norm_extracted = clean_gender(extracted_gen)
        norm_input = clean_gender(input_gen)

        gender_score = 0
        
        # Case A: OCR detected Male/Female
        if norm_extracted in ["male", "female"]:
            if norm_input and norm_input == norm_extracted:
                gender_score = self.weights["gender"]
            elif norm_input:
                # Mismatch -> Critical Failure
                gender_score = 0
                critical_failure = True
                rejection_reasons.append(f"Gender Mismatch (Input: {norm_input} vs OCR: {norm_extracted})")
        
        # Case B: OCR is 'Other' or 'Not Detected' -> Gender should come from entity_data (via gender_pipeline in main.py)
        else:
            # If OCR failed, the main.py should have already tried gender_pipeline
            # and updated entity_data['gender'] with the fallback result
            # So if we're here and gender is still 'Other', it means all methods failed
            if norm_input:
                # We have expected gender but couldn't verify it
                critical_failure = True
                rejection_reasons.append("Gender could not be verified from document (OCR failed)")
            else:
                # No expected gender provided, can't verify
                # Don't fail, but give 0 points
                pass

        score += gender_score
        breakdown["gender_score"] = gender_score

        # --- FINAL DECISION ---
        if critical_failure:
            status = "REJECTED"
        elif score > 60.1:
            # High score -> APPROVED (even with low face sim)
            status = "APPROVED"
        elif low_face_sim and score >= 60:
            # Low face similarity but decent score -> REVIEW
            status = "REVIEW"
        elif 40 <= score <= 60.1:
            status = "REVIEW"
        else:
            status = "REJECTED"

        return {
            "total_score": round(score, 2),
            "status": status,
            "breakdown": breakdown,
            "rejection_reasons": rejection_reasons
        }