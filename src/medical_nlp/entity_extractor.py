# src/medical_nlp/entity_extractor.py

import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts medical entities using NER models
    """
    
    def __init__(self, config: Dict):
        self.config = config
        model_name = config['models']['medical_ner']['name']
        
        # Check GPU availability
        self.device = self._get_device(config['models']['medical_ner']['device'])
        
        logger.info(f"Loading NER model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load NER pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy=config['models']['medical_ner']['aggregation_strategy'],
            device=0 if self.device == 'cuda' else -1
        )
        
        # Medical keyword patterns
        self.medical_keywords = config['processing']['medical_keywords']
    
    def _get_device(self, preferred_device: str) -> str:
        """Determine which device to use"""
        if preferred_device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extract medical entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with categorized entities
        """
        logger.info("Extracting medical entities...")
        
        # Run NER
        entities = self.ner_pipeline(text)
        
        # Categorize entities
        categorized = self._categorize_entities(entities)
        
        # Post-process and enhance
        enhanced = self._enhance_entities(categorized, text)
        
        return enhanced
    
    def _categorize_entities(self, entities: List[Dict]) -> Dict:
        """Categorize extracted entities"""
        categorized = {
            'symptoms': [],
            'treatments': [],
            'body_parts': [],
            'diagnoses': [],
            'medications': [],
            'procedures': []
        }
        
        for entity in entities:
            word = entity['word'].replace('##', '').strip()
            label = entity['entity_group']
            score = entity['score']
            
            entity_info = {
                'value': word,
                'confidence': float(round(score, 3)),  # Convert NumPy float to Python float
                'label': label
            }
            
            # Map entity types
            if label == 'Sign_symptom':
                categorized['symptoms'].append(entity_info)
            elif label == 'Therapeutic_procedure':
                categorized['treatments'].append(entity_info)
            elif label == 'Medication':
                categorized['medications'].append(entity_info)
            elif label == 'Biological_structure':
                categorized['body_parts'].append(entity_info)
            elif label == 'Diagnostic_procedure':
                categorized['procedures'].append(entity_info)
            elif label == 'Disease_disorder':
                categorized['diagnoses'].append(entity_info)
        
        return categorized
    
    def _enhance_entities(self, categorized: Dict, text: str) -> Dict:
        """Enhance entities with context and combinations"""
        enhanced = {}
        
        # Combine body parts with symptoms
        enhanced['symptoms'] = self._combine_symptoms_bodyparts(
            categorized['symptoms'],
            categorized['body_parts'],
            text
        )
        
        # Process treatments
        enhanced['treatments'] = self._deduplicate_and_clean(
            categorized['treatments'] + categorized['procedures']
        )
        
        # Process diagnoses
        enhanced['diagnoses'] = self._deduplicate_and_clean(
            categorized['diagnoses']
        )
        
        # Keep medications separate
        enhanced['medications'] = self._deduplicate_and_clean(
            categorized['medications']
        )
        
        # Store body parts for reference
        enhanced['body_parts'] = self._deduplicate_and_clean(
            categorized['body_parts']
        )
        
        return enhanced
    
    def _combine_symptoms_bodyparts(self, symptoms: List[Dict], 
                                   body_parts: List[Dict], 
                                   text: str) -> List[Dict]:
        """Combine body parts with symptoms to create phrases like 'neck pain'"""
        combined = []
        text_lower = text.lower()
        
        # Get unique body parts and symptoms
        body_part_values = [bp['value'].lower() for bp in body_parts]
        symptom_values = [s['value'].lower() for s in symptoms]
        
        # Check for combinations in text
        for bp in body_part_values:
            for symptom in self.medical_keywords['symptoms']:
                pattern = f"{bp} {symptom}"
                if pattern in text_lower:
                    combined.append({
                        'value': pattern.title(),
                        'confidence': 0.85,
                        'source': 'combined'
                    })
        
        # Add standalone symptoms
        for symptom in symptoms:
            if symptom['value'].lower() in self.medical_keywords['symptoms']:
                combined.append({
                    'value': symptom['value'],
                    'confidence': symptom['confidence'],
                    'source': 'extracted'
                })
        
        return self._deduplicate_and_clean(combined)
    
    def _deduplicate_and_clean(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicates and clean entities"""
        seen = set()
        cleaned = []
        
        for entity in entities:
            value = entity['value'].lower().strip()
            if value and value not in seen:
                seen.add(value)
                # Capitalize properly
                entity['value'] = ' '.join([w.capitalize() for w in value.split()])
                cleaned.append(entity)
        
        return cleaned