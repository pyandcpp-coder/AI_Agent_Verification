import re
from datetime import datetime

class VerificationScorer:
    def __init__(self):
        # Weights (Total 100)
        self.weights = {
            "face": 40,      # Face Similarity
            "aadhar": 20,    # Valid Aadhaar Number
            "dob": 20,       # Age 18+ & DOB Match
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
        face_sim = face_data.get("similarity", 0)
        
        # Critical: Face match below 15% -> Auto Reject
        if face_sim < 15:
            breakdown["face_score"] = 0
            critical_failure = True
            rejection_reasons.append(f"Face Not Matching (Similarity: {face_sim:.2f}%)")
        elif face_sim >= 40:
            face_points = min(40, (face_sim / 100) * 40 * 1.5)
            if face_sim > 70: face_points = 40
            score += face_points
            breakdown["face_score"] = round(face_points, 2)
        else:
            breakdown["face_score"] = 0
            rejection_reasons.append(f"Low Face Match ({face_sim:.2f}%)")

        # --- 2. Aadhaar Validity (0 - 20 points) ---
        aadhaar_num = entity_data.get("aadharnumber", "").replace(" ", "")
        # Check for XXXX masked numbers or invalid format
        if "XXXX" in entity_data.get("aadharnumber", "") or "xxxx" in entity_data.get("aadharnumber", ""):
            breakdown["aadhar_score"] = 0
            critical_failure = True
            rejection_reasons.append("Aadhaar Number is Masked (XXXX)")
        elif len(aadhaar_num) == 12 and aadhaar_num.isdigit():
            score += self.weights["aadhar"]
            breakdown["aadhar_score"] = self.weights["aadhar"]
        else:
            breakdown["aadhar_score"] = 0
            rejection_reasons.append("Invalid/Unreadable Aadhaar Number")

        # --- 3. DOB & Age Check ---
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
                    # DOB Mismatch -> Critical Failure (REJECT)
                    dob_score = 0
                    critical_failure = True
                    rejection_reasons.append(f"DOB Mismatch (Input: {expected_dob} vs Aadhaar: {extracted_dob})")
                else:
                    # Could not parse one of them, but Age is 18+ -> Give points
                    dob_score = self.weights["dob"]
            else:
                # No input DOB to cross-check, but Age is 18+ -> Give points
                dob_score = self.weights["dob"]
        else:
            # Under 18 -> Critical Failure (REJECT)
            dob_score = 0
            critical_failure = True
            rejection_reasons.append("User is Under 18 or DOB Unreadable")
            
        score += dob_score
        breakdown["dob_score"] = dob_score

        # --- 4. Gender Check (Non-Critical - Just affects score) ---
        # Normalize: "Male", "male", "M" -> "male"
        extracted_gen = entity_data.get("gender", "").lower()
        input_gen = expected_gender.lower() if expected_gender else ""
        
        # Helper to clean gender string
        def clean_gender(g):
            if "female" in g: return "female"
            if "male" in g: return "male"
            return "other"

        gender_score = 0
        if input_gen and extracted_gen:
            if clean_gender(input_gen) == clean_gender(extracted_gen):
                gender_score = self.weights["gender"]
            else:
                # Gender Mismatch -> Just 0 points, not critical
                gender_score = 0
                rejection_reasons.append(f"Gender Mismatch (Input: {input_gen} vs Aadhaar: {extracted_gen})")
        elif not extracted_gen:
            # Gender unreadable -> Just 0 points, not critical
            rejection_reasons.append("Gender Unreadable on Aadhaar")
        
        score += gender_score
        breakdown["gender_score"] = gender_score

        # --- FINAL DECISION BASED ON SCORE ---
        # Critical failures: DOB mismatch, Age < 18, XXXX in Aadhaar
        if critical_failure:
            status = "REJECTED"
        elif score >= 65:
            status = "APPROVED"
        elif 40 <= score < 65:
            status = "REVIEW"
        else:  # score < 40
            status = "REJECTED"

        return {
            "total_score": round(score, 2),
            "status": status,
            "breakdown": breakdown,
            "rejection_reasons": rejection_reasons
        }