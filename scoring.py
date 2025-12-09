# scoring.py
import re
from datetime import datetime

class VerificationScorer:
    def __init__(self):
        # Weights (Total 100)
        self.weights = {
            "face": 40,      # High weightage for Face
            "aadhar": 20,    # Valid Aadhaar Number
            "dob": 20,       # Valid Age/DOB
            "gender": 20     # Gender Match
        }
    
    def _normalize_dob(self, dob_string):
        """
        Normalize DOB to YYYY-MM-DD format for comparison.
        Handles formats like: DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, etc.
        """
        if not dob_string:
            return None
        
        # Try common formats
        formats = [
            '%d-%m-%Y',  # 06-12-2007
            '%d/%m/%Y',  # 06/12/2007
            '%Y-%m-%d',  # 2007-12-06
            '%d.%m.%Y',  # 06.12.2007
            '%Y/%m/%d',  # 2007/12/06
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(str(dob_string).strip(), fmt)
                return dt.strftime('%Y-%m-%d')  # Return in standard format
            except ValueError:
                continue
        
        return None

    def calculate_score(self, face_data, entity_data, expected_gender, expected_dob=None):
        score = 0
        breakdown = {}
        rejection_reasons = []
        critical_failure = False  # Track if any mandatory check fails

        # --- 1. Face Similarity (0 - 40 points) ---
        # Assumption: Face sim returns a score 0-100.
        # We map 0-100 similarity to 0-40 points.
        face_sim = face_data.get("similarity", 0)
        face_points = 0
        
        if face_sim >= 40: # Threshold for "Same Person"
            # Linear scaling: 40 sim -> 20 pts, 80+ sim -> 40 pts
            face_points = min(40, (face_sim / 100) * 40 * 1.5) 
            if face_sim > 70: face_points = 40 # Max points for very high match
        else:
            rejection_reasons.append(f"Face mismatch (Similarity: {face_sim:.2f}%)")
        
        score += face_points
        breakdown["face_score"] = round(face_points, 2)

        # --- 2. Aadhaar Validation (0 - 20 points) ---
        aadhaar_num = entity_data.get("aadharnumber", "").replace(" ", "")
        if len(aadhaar_num) == 12 and aadhaar_num.isdigit():
            # (Optional) Add Verhoeff algorithm here for extra strictness
            score += self.weights["aadhar"]
            breakdown["aadhar_score"] = self.weights["aadhar"]
        else:
            breakdown["aadhar_score"] = 0
            rejection_reasons.append("Invalid Aadhaar Number format")

        # --- 3. DOB / Age Validation (0 - 20 points) --- [MANDATORY CHECK]
        # RULE: User MUST be 18 or older to be approved
        extracted_dob = entity_data.get("dob", "")
        extracted_age = entity_data.get("age")
        
        # First check: Age must be 18+
        if entity_data.get("age_status") == "age_approved" and extracted_age and extracted_age >= 18:
            # Age is valid and 18+
            # Optional: If DOB is provided in request, verify it matches
            if expected_dob and extracted_dob:
                # Normalize both DOBs for comparison
                normalized_expected = self._normalize_dob(expected_dob)
                normalized_extracted = self._normalize_dob(extracted_dob)
                
                if normalized_expected and normalized_extracted:
                    if normalized_expected == normalized_extracted:
                        score += self.weights["dob"]
                        breakdown["dob_score"] = self.weights["dob"]
                        breakdown["dob_match"] = True
                    else:
                        # DOB mismatch - partial score for being 18+ but wrong DOB
                        score += 10  # Partial credit
                        breakdown["dob_score"] = 10
                        breakdown["dob_match"] = False
                        rejection_reasons.append(f"DOB mismatch (Provided: {expected_dob}, Aadhaar: {extracted_dob})")
                else:
                    # Give full score if comparison not possible but age is valid
                    score += self.weights["dob"]
                    breakdown["dob_score"] = self.weights["dob"]
            else:
                # No expected DOB provided, just verify age is 18+
                score += self.weights["dob"]
                breakdown["dob_score"] = self.weights["dob"]
        else:
            breakdown["dob_score"] = 0
            critical_failure = True  # Age verification is mandatory
            if extracted_age and extracted_age < 18:
                rejection_reasons.append(f"User is under 18 (Age: {extracted_age})")
            else:
                rejection_reasons.append("DOB invalid or user is under 18")

        # --- 4. Gender Check (0 - 20 points) --- [MANDATORY CHECK]
        extracted_gender = entity_data.get("gender", "").lower()
        expected_gender_input = expected_gender.lower() if expected_gender else ""
        
        # Normalize gender strings (e.g., 'm' -> 'male')
        if "male" in extracted_gender and "female" not in extracted_gender: 
            extracted_gender = "male"
        elif "female" in extracted_gender: 
            extracted_gender = "female"
        
        if expected_gender_input and extracted_gender:
            if expected_gender_input == extracted_gender:
                score += self.weights["gender"]
                breakdown["gender_score"] = self.weights["gender"]
            else:
                breakdown["gender_score"] = 0
                critical_failure = True  # Gender mismatch is mandatory rejection
                rejection_reasons.append(f"Gender mismatch (Provided: {expected_gender_input.capitalize()}, Aadhaar: {extracted_gender.capitalize()})")
        elif not expected_gender_input:
             # If no expected gender provided, cannot verify - treat as failure
             breakdown["gender_score"] = 0
             critical_failure = True
             rejection_reasons.append("Gender not provided for verification")
        else:
             # Gender not extracted from Aadhaar
             breakdown["gender_score"] = 0
             critical_failure = True
             rejection_reasons.append("Gender not readable from Aadhaar")

        # Final Verdict - STRICT ENFORCEMENT
        # If age < 18 OR gender mismatch -> AUTOMATIC REJECTION
        if critical_failure:
            status = "REJECTED"
        elif score >= 70:
            status = "APPROVED"
        else:
            status = "REJECTED"

        return {
            "total_score": round(score, 2),
            "status": status,
            "breakdown": breakdown,
            "rejection_reasons": rejection_reasons
        }