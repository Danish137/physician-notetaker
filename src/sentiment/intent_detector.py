# src/sentiment/intent_detector.py

import torch
from transformers import pipeline
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class IntentDetector:
    """
    Detects patient intent using zero-shot classification
    """
    
    def __init__(self, config: Dict):
        self.config = config
        model_name = config['models']['intent']['name']
        
        # Check GPU availability
        self.device = self._get_device(config['models']['intent']['device'])
        
        logger.info(f"Loading intent model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load zero-shot classification pipeline
        self.intent_pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.device == 'cuda' else -1
        )
        
        # Intent labels
        self.intent_labels = config.get('intent_labels', [
            "reporting symptoms",
            "seeking reassurance",
            "asking questions",
            "expressing concern",
            "providing medical history",
            "confirming understanding"
        ])
    
    def _get_device(self, preferred_device: str) -> str:
        """Determine which device to use"""
        if preferred_device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def detect(self, patient_utterances: List) -> Dict:
        """
        Detect patient intent from utterances
        
        Args:
            patient_utterances: List of patient utterances
            
        Returns:
            Dictionary with intent analysis
        """
        if not patient_utterances:
            return {
                'primary_intent': 'Unknown',
                'confidence': 0.0,
                'all_intents': []
            }
        
        logger.info("Detecting patient intent...")
        
        # Combine patient utterances
        combined_text = ' '.join([u.text for u in patient_utterances])
        
        # Truncate if too long
        if len(combined_text) > 1000:
            combined_text = combined_text[:1000]
        
        # Classify intent
        result = self.intent_pipeline(
            combined_text,
            candidate_labels=self.intent_labels,
            multi_label=True
        )
        
        # Get primary intent
        primary_intent = result['labels'][0]
        primary_score = result['scores'][0]
        
        # Get all significant intents (score > 0.3)
        all_intents = [
            {
                'intent': label,
                'confidence': float(round(score, 3))  # Convert NumPy float to Python float
            }
            for label, score in zip(result['labels'], result['scores'])
            if score > 0.3
        ]
        
        return {
            'primary_intent': primary_intent.title(),
            'confidence': float(round(primary_score, 3)),  # Convert NumPy float to Python float
            'all_intents': all_intents
        }