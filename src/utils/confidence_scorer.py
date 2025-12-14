# src/utils/confidence_scorer.py

from typing import Dict, Any

class ConfidenceScorer:
    """
    Calculate and manage confidence scores for extracted information
    """
    
    @staticmethod
    def calculate_field_confidence(field_value: Any, source: str, 
                                   model_confidence: float = None) -> float:
        """
        Calculate confidence score for a field
        
        Args:
            field_value: The extracted value
            source: How the value was obtained ('stated', 'extracted', 'inferred')
            model_confidence: Model's confidence if available
            
        Returns:
            Confidence score (0.0-1.0)
        """
        if field_value is None:
            return 0.0
        
        # Base confidence by source
        base_confidence = {
            'stated': 0.95,
            'extracted': 0.85,
            'combined': 0.80,
            'inferred': 0.70,
            'predicted': 0.60,
            'not_mentioned': 0.0
        }
        
        confidence = base_confidence.get(source, 0.5)
        
        # Adjust by model confidence if available
        if model_confidence is not None:
            confidence = (confidence + model_confidence) / 2
        
        return round(confidence, 3)
    
    @staticmethod
    def calculate_overall_confidence(data: Dict) -> float:
        """
        Calculate overall confidence for entire extraction
        
        Args:
            data: Extracted data dictionary
            
        Returns:
            Overall confidence score
        """
        confidences = []
        
        def extract_confidences(obj):
            if isinstance(obj, dict):
                if 'confidence' in obj:
                    confidences.append(obj['confidence'])
                for value in obj.values():
                    extract_confidences(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_confidences(item)
        
        extract_confidences(data)
        
        if not confidences:
            return 0.0
        
        return round(sum(confidences) / len(confidences), 3)
    
    @staticmethod
    def get_confidence_level(confidence: float) -> str:
        """
        Get confidence level label
        
        Args:
            confidence: Confidence score
            
        Returns:
            Confidence level string
        """
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        elif confidence > 0.0:
            return "Low"
        else:
            return "None"