# src/utils/validators.py

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class OutputValidator:
    """
    Validates extracted medical information for consistency and completeness
    """
    
    @staticmethod
    def validate_medical_summary(summary: Dict) -> Dict:
        """
        Validate medical summary
        
        Args:
            summary: Medical summary dictionary
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        # Check required fields
        required_fields = ['Symptoms', 'Diagnosis', 'Treatment']
        for field in required_fields:
            if field not in summary or summary[field] is None:
                warnings.append(f"Missing field: {field}")
            elif isinstance(summary[field], list) and len(summary[field]) == 0:
                warnings.append(f"Empty field: {field}")
        
        # Check diagnosis-symptom consistency
        diagnosis = summary.get('Diagnosis', {})
        symptoms = summary.get('Symptoms', [])
        
        if diagnosis and diagnosis.get('value'):
            diagnosis_value = diagnosis['value'].lower()
            symptom_values = [s.get('value', '').lower() for s in symptoms]
            
            # Check if diagnosis makes sense with symptoms
            if 'whiplash' in diagnosis_value:
                if not any('neck' in s for s in symptom_values):
                    warnings.append("Whiplash diagnosis but no neck symptoms mentioned")
        
        # Check treatment-diagnosis alignment
        treatments = summary.get('Treatment', [])
        if diagnosis and diagnosis.get('value') and not treatments:
            warnings.append("Diagnosis present but no treatment documented")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    @staticmethod
    def validate_soap_note(soap: Dict) -> Dict:
        """
        Validate SOAP note
        
        Args:
            soap: SOAP note dictionary
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        # Check required sections
        required_sections = ['Subjective', 'Objective', 'Assessment', 'Plan']
        for section in required_sections:
            if section not in soap:
                issues.append(f"Missing SOAP section: {section}")
        
        # Check section completeness
        if 'Subjective' in soap:
            subj = soap['Subjective']
            if not subj.get('Chief_Complaint'):
                warnings.append("Subjective: Missing Chief Complaint")
        
        if 'Assessment' in soap:
            assess = soap['Assessment']
            if not assess.get('Diagnosis'):
                warnings.append("Assessment: Missing Diagnosis")
        
        if 'Plan' in soap:
            plan = soap['Plan']
            if not plan.get('Treatment'):
                warnings.append("Plan: Missing Treatment")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    @staticmethod
    def validate_sentiment_analysis(sentiment: Dict) -> Dict:
        """
        Validate sentiment analysis
        
        Args:
            sentiment: Sentiment analysis dictionary
            
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        # Check required fields
        if 'overall_sentiment' not in sentiment:
            issues.append("Missing overall_sentiment")
        
        if 'confidence' not in sentiment:
            issues.append("Missing confidence score")
        elif sentiment['confidence'] < 0.3:
            warnings.append(f"Low confidence in sentiment analysis: {sentiment['confidence']}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }