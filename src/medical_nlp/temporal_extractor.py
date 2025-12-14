# src/medical_nlp/temporal_extractor.py

import re
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TemporalExtractor:
    """
    Extracts temporal information from text
    - Dates
    - Durations
    - Quantities
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Patterns for temporal extraction
        self.patterns = {
            'duration': [
                r'(\d+)\s+(day|week|month|year)s?',
                r'(four|five|six|seven|eight|nine|ten)\s+(week|month|year)s?'
            ],
            'quantity': [
                r'(\d+)\s+(session|appointment|visit|time)s?',
                r'(ten|twenty|thirty)\s+(session|appointment|visit)s?'
            ],
            'date': [
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+',
                r'\d{1,2}/\d{1,2}/\d{2,4}',
                r'(last|this|next)\s+(week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'
            ]
        }
        
        # Number word mapping
        self.word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50
        }
    
    def extract_temporal_info(self, text: str) -> Dict:
        """
        Extract all temporal information from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with temporal information
        """
        return {
            'durations': self._extract_durations(text),
            'quantities': self._extract_quantities(text),
            'dates': self._extract_dates(text)
        }
    
    def _extract_durations(self, text: str) -> List[Dict]:
        """Extract duration mentions"""
        durations = []
        
        for pattern in self.patterns['duration']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                number = match.group(1)
                unit = match.group(2)
                
                # Convert word to number if needed
                if number.lower() in self.word_to_num:
                    number = str(self.word_to_num[number.lower()])
                
                durations.append({
                    'value': f"{number} {unit}s" if int(number) > 1 else f"{number} {unit}",
                    'raw': match.group(0),
                    'confidence': 0.9
                })
        
        return durations
    
    def _extract_quantities(self, text: str) -> List[Dict]:
        """Extract quantity mentions (e.g., '10 sessions')"""
        quantities = []
        
        for pattern in self.patterns['quantity']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                number = match.group(1)
                unit = match.group(2)
                
                # Convert word to number
                if number.lower() in self.word_to_num:
                    number = str(self.word_to_num[number.lower()])
                
                quantities.append({
                    'value': int(number),
                    'unit': unit + 's' if int(number) > 1 else unit,
                    'raw': match.group(0),
                    'confidence': 0.95
                })
        
        return quantities
    
    def _extract_dates(self, text: str) -> List[Dict]:
        """Extract date mentions"""
        dates = []
        
        for pattern in self.patterns['date']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append({
                    'value': match.group(0),
                    'confidence': 0.85
                })
        
        return dates
    
    def format_treatment_with_quantity(self, treatment: str, text: str) -> str:
        """
        Format treatment with quantity if found
        E.g., 'physiotherapy' + '10 sessions' -> '10 physiotherapy sessions'
        """
        quantities = self._extract_quantities(text)
        
        for qty in quantities:
            # Check if quantity is near treatment mention
            treatment_lower = treatment.lower()
            if treatment_lower in text.lower():
                # Find treatment position
                text_lower = text.lower()
                treatment_pos = text_lower.find(treatment_lower)
                qty_pos = text_lower.find(qty['raw'].lower())
                
                # If quantity is within 50 characters of treatment
                if abs(treatment_pos - qty_pos) < 50:
                    return f"{qty['value']} {treatment} {qty['unit']}"
        
        return treatment