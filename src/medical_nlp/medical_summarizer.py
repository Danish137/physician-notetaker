# src/medical_nlp/medical_summarizer.py

import os
import json
from typing import Dict, Optional
import logging
from groq import Groq

logger = logging.getLogger(__name__)

class MedicalSummarizer:
    """
    Extracts structured medical information using LLM or rule-based approach
    """
    
    def __init__(self, config: Dict, use_llm: bool = True):
        self.config = config
        self.use_llm = use_llm
        
        # Initialize Groq client if API key available
        self.client = None
        if use_llm:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.client = Groq(api_key=api_key)
                logger.info("Groq client initialized")
            else:
                logger.warning("No Groq API key found. Using rule-based extraction only.")
                self.use_llm = False
    
    def generate_summary(self, text: str, entities: Dict, temporal_info: Dict) -> Dict:
        """
        Generate structured medical summary
        
        Args:
            text: Full conversation text
            entities: Extracted entities
            temporal_info: Temporal information
            
        Returns:
            Structured medical summary
        """
        if self.use_llm and self.client:
            return self._llm_based_extraction(text, entities, temporal_info)
        else:
            return self._rule_based_extraction(text, entities, temporal_info)
    
    def _llm_based_extraction(self, text: str, entities: Dict, temporal_info: Dict) -> Dict:
        """Use LLM for structured extraction"""
        logger.info("Using LLM for medical information extraction...")
        
        prompt = self._create_extraction_prompt(text, entities, temporal_info)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config['models']['llm']['model'],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical information extraction assistant. Extract structured medical data from physician-patient conversations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config['models']['llm']['temperature'],
                max_tokens=self.config['models']['llm']['max_tokens']
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            # Remove markdown code blocks if present
            content = content.replace('```json', '').replace('```', '').strip()
            result = json.loads(content)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}. Falling back to rule-based extraction.")
            return self._rule_based_extraction(text, entities, temporal_info)
    
    def _create_extraction_prompt(self, text: str, entities: Dict, temporal_info: Dict) -> str:
        """Create prompt for LLM extraction"""
        prompt = f"""
Extract structured medical information from this physician-patient conversation.

CONVERSATION:
{text}

EXTRACTED ENTITIES:
Symptoms: {[e['value'] for e in entities.get('symptoms', [])]}
Body Parts: {[e['value'] for e in entities.get('body_parts', [])]}
Treatments: {[e['value'] for e in entities.get('treatments', [])]}
Diagnoses: {[e['value'] for e in entities.get('diagnoses', [])]}

TEMPORAL INFORMATION:
Durations: {temporal_info.get('durations', [])}
Quantities: {temporal_info.get('quantities', [])}

TASK:
Extract the following information in JSON format:

{{
  "Patient_Name": {{
    "value": "name or null if not mentioned",
    "confidence": 0.0-1.0,
    "source": "stated" or "not_mentioned"
  }},
  "Symptoms": [
    {{
      "value": "combined symptom (e.g., 'Neck pain')",
      "confidence": 0.0-1.0,
      "source": "stated" or "extracted"
    }}
  ],
  "Diagnosis": {{
    "value": "diagnosis or null",
    "confidence": 0.0-1.0,
    "source": "stated" or "inferred",
    "reasoning": "why this diagnosis was inferred (if applicable)"
  }},
  "Treatment": [
    {{
      "value": "treatment with quantity (e.g., '10 physiotherapy sessions')",
      "confidence": 0.0-1.0,
      "source": "stated" or "extracted"
    }}
  ],
  "Current_Status": {{
    "value": "current patient status",
    "confidence": 0.0-1.0,
    "source": "stated"
  }},
  "Prognosis": {{
    "value": "expected outcome or null",
    "confidence": 0.0-1.0,
    "source": "stated" or "not_mentioned"
  }}
}}

RULES:
1. Combine body parts with symptoms (e.g., "neck" + "pain" = "Neck pain")
2. Include quantities with treatments (e.g., "10 physiotherapy sessions")
3. Mark inferred information clearly with reasoning
4. Use null for missing information
5. Provide confidence scores based on how explicit the information is
6. For diagnosis, consider the full context (symptoms, accident description, treatment given)

Return ONLY the JSON object, no additional text.
"""
        return prompt
    
    def _rule_based_extraction(self, text: str, entities: Dict, temporal_info: Dict) -> Dict:
        """Rule-based extraction as fallback"""
        logger.info("Using rule-based medical information extraction...")
        
        text_lower = text.lower()
        
        # Extract patient name
        patient_name = self._extract_patient_name(text)
        
        # Process symptoms
        symptoms = self._process_symptoms(entities, text)
        
        # Infer diagnosis
        diagnosis = self._infer_diagnosis(text, symptoms, entities)
        
        # Process treatments
        treatments = self._process_treatments(entities, temporal_info, text)
        
        # Extract current status
        current_status = self._extract_current_status(text)
        
        # Extract prognosis
        prognosis = self._extract_prognosis(text)
        
        return {
            "Patient_Name": patient_name,
            "Symptoms": symptoms,
            "Diagnosis": diagnosis,
            "Treatment": treatments,
            "Current_Status": current_status,
            "Prognosis": prognosis
        }
    
    def _extract_patient_name(self, text: str) -> Dict:
        """Extract patient name from text"""
        # Look for name patterns
        import re
        
        # Pattern: Ms./Mr./Mrs. [Name]
        name_pattern = r'(Ms\.|Mr\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        match = re.search(name_pattern, text)
        
        if match:
            return {
                "value": match.group(2),
                "confidence": 0.95,
                "source": "stated"
            }
        
        return {
            "value": None,
            "confidence": 0.0,
            "source": "not_mentioned"
        }
    
    def _process_symptoms(self, entities: Dict, text: str) -> list:
        """Process and combine symptoms"""
        symptoms_list = []
        
        # Get symptoms from entities
        for symptom in entities.get('symptoms', []):
            symptoms_list.append({
                "value": symptom['value'],
                "confidence": symptom['confidence'],
                "source": symptom.get('source', 'extracted')
            })
        
        # Check for "head impact" in car accident context
        if 'car accident' in text.lower() and 'head' in text.lower():
            if 'hit' in text.lower() or 'impact' in text.lower():
                symptoms_list.append({
                    "value": "Head impact",
                    "confidence": 0.80,
                    "source": "extracted"
                })
        
        return symptoms_list
    
    def _infer_diagnosis(self, text: str, symptoms: list, entities: Dict) -> Dict:
        """Infer diagnosis from context"""
        text_lower = text.lower()
        
        # Check if diagnosis explicitly mentioned
        for diag in entities.get('diagnoses', []):
            if diag['value'].lower() in text_lower:
                return {
                    "value": diag['value'],
                    "confidence": 0.90,
                    "source": "stated"
                }
        
        # Inference rules
        # Rule 1: Car accident + neck/back pain = Whiplash
        if 'car accident' in text_lower or 'accident' in text_lower:
            symptom_values = [s['value'].lower() for s in symptoms]
            if any('neck' in s for s in symptom_values) and any('back' in s for s in symptom_values):
                return {
                    "value": "Whiplash injury",
                    "confidence": 0.75,
                    "source": "inferred",
                    "reasoning": "Car accident with rear impact causing neck and back pain is characteristic of whiplash injury"
                }
        
        # Check if whiplash mentioned in text
        if 'whiplash' in text_lower:
            return {
                "value": "Whiplash injury",
                "confidence": 0.90,
                "source": "stated"
            }
        
        return {
            "value": None,
            "confidence": 0.0,
            "source": "not_mentioned"
        }
    
    def _process_treatments(self, entities: Dict, temporal_info: Dict, text: str) -> list:
        """Process treatments with quantities"""
        treatments = []
        
        # Get treatments from entities
        treatment_entities = entities.get('treatments', [])
        quantities = temporal_info.get('quantities', [])
        
        # Match treatments with quantities
        for treatment in treatment_entities:
            treatment_value = treatment['value']
            treatment_lower = treatment_value.lower()
            
            # Try to find matching quantity
            matched = False
            for qty in quantities:
                if qty['unit'].lower() in ['session', 'sessions']:
                    # Check if treatment and quantity are related
                    if treatment_lower in text.lower():
                        treatments.append({
                            "value": f"{qty['value']} {treatment_value} {qty['unit']}",
                            "confidence": 0.90,
                            "source": "stated"
                        })
                        matched = True
                        break
            
            if not matched:
                treatments.append({
                    "value": treatment_value,
                    "confidence": treatment['confidence'],
                    "source": "stated"
                })
        
        # Check for medications
        if 'painkiller' in text.lower() or 'pain killer' in text.lower():
            treatments.append({
                "value": "Painkillers",
                "confidence": 0.85,
                "source": "stated"
            })
        
        return treatments
    
    def _extract_current_status(self, text: str) -> Dict:
        """Extract current patient status"""
        text_lower = text.lower()
        
        # Look for current status indicators
        current_indicators = ['now', 'currently', 'at present', 'occasional']
        
        for indicator in current_indicators:
            if indicator in text_lower:
                # Find the sentence with this indicator
                sentences = text.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        # Look for pain/symptom descriptors
                        if 'occasional' in sentence.lower() and 'back' in sentence.lower():
                            return {
                                "value": "Occasional backache",
                                "confidence": 0.85,
                                "source": "stated"
                            }
        # If no current status indicators matched, return not mentioned
        return {
            "value": None,
            "confidence": 0.0,
            "source": "not_mentioned"
        }

    def _extract_prognosis(self, text: str) -> Dict:
        """Extract prognosis/expected outcome"""
        text_lower = text.lower()

        # Look for prognosis indicators
        prognosis_keywords = ['recovery', 'expect', 'prognosis', 'outlook', 'within']

        if any(keyword in text_lower for keyword in prognosis_keywords):
            # Look for time-based recovery
            if 'full recovery' in text_lower:
                # Extract timeframe
                import re
                time_pattern = r'within\s+(\w+\s+\w+)'
                match = re.search(time_pattern, text_lower)

                if match:
                    timeframe = match.group(1)
                    return {
                        "value": f"Full recovery expected within {timeframe}",
                        "confidence": 0.90,
                        "source": "stated"
                    }

                return {
                    "value": "Full recovery expected",
                    "confidence": 0.85,
                    "source": "stated"
                }

        return {
            "value": None,
            "confidence": 0.0,
            "source": "not_mentioned"
        }
