import re
from datetime import datetime

class VerificationScorer:
    def __init__(self):
        # Weights (Total 100) - Original Configuration
        self.weights = {
            "face": 40,      # Face Similarity
            "aadhar": 20,    # Valid Aadhaar Number & Masking Check
            "dob": 20,       # Valid Age/DOB
            "gender": 20     # Gender Match
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

        # --- 1. Face Similarity (0 - 40 points) ---
        face_sim = face_data.get("score", 0)
        low_face_sim = False
        
        if face_sim < 16:
            # Low face similarity - Flag it but don't auto-reject yet
            breakdown["face_score"] = 0
            low_face_sim = True
            rejection_reasons.append(f"Low Face Similarity (Similarity: {face_sim:.2f}%)")
        else:
            # Between 16% and 100% -> Scale linearly to 0-40 points
            # Formula: ((Score - 16) / 84) * 40, where 84 is the range (100-16)
            face_points = ((face_sim - 16) / 84) * 40
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
        
        # Case B: OCR is 'Other' or 'Not Detected' -> Use Face Analysis for Bonus
        else:
            # Fallback to Gender detected from Face (by InsightFace in face_sim.py)
            face_detected_gender = face_data.get("gender_img2") # Gender from Aadhaar Front image
            norm_face_gender = clean_gender(face_detected_gender)
            
            if norm_input and norm_face_gender in ["male", "female"]:
                if norm_input == norm_face_gender:
                    # MATCH! Give 10 Bonus Points (instead of full 20)
                    bonus_points = 10
                    score += bonus_points
                    breakdown["gender_bonus"] = bonus_points
                else:
                    # Mismatch on face gender too -> Reject
                    critical_failure = True
                    rejection_reasons.append(f"Gender Mismatch (Input: {norm_input} vs Face Analysis: {norm_face_gender})")
            else:
                # Could not determine gender from OCR or Face
                critical_failure = True
                rejection_reasons.append("Gender could not be verified (OCR & Face failed)")

        score += gender_score
        breakdown["gender_score"] = gender_score

        # --- FINAL DECISION ---
        if critical_failure:
            status = "REJECTED"
        elif low_face_sim and score >= 60:
            # Low face similarity but high overall score -> REVIEW
            status = "REVIEW"
        elif score >= 65:
            status = "APPROVED"
        elif 40 <= score < 65:
            status = "REVIEW"
        else:
            status = "REJECTED"

        return {
            "total_score": round(score, 2),
            "status": status,
            "breakdown": breakdown,
            "rejection_reasons": rejection_reasons
        }