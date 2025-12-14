# src/sentiment/sentiment_analyzer.py

import torch
from transformers import pipeline
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes patient sentiment from conversations
    """
    
    def __init__(self, config: Dict):
        self.config = config
        model_name = config['models']['sentiment']['name']
        
        # Check GPU availability
        self.device = self._get_device(config['models']['sentiment']['device'])
        
        logger.info(f"Loading sentiment model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if self.device == 'cuda' else -1
        )
        
        # Sentiment mapping
        self.sentiment_mapping = config.get('sentiment_mapping', {
            'POSITIVE': 'Reassured',
            'NEGATIVE': 'Anxious',
            'NEUTRAL': 'Neutral'
        })
    
    def _get_device(self, preferred_device: str) -> str:
        """Determine which device to use"""
        if preferred_device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def analyze(self, patient_utterances: List) -> Dict:
        """
        Analyze sentiment from patient utterances
        
        Args:
            patient_utterances: List of patient utterances
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not patient_utterances:
            return {
                'overall_sentiment': 'Neutral',
                'sentiment_score': 0.5,
                'confidence': 0.0,
                'utterance_sentiments': []
            }
        
        logger.info("Analyzing patient sentiment...")
        
        # Analyze each utterance
        utterance_sentiments = []
        for utterance in patient_utterances:
            text = utterance.text
            
            # Skip very short utterances
            if len(text.split()) < 3:
                continue
            
            result = self.sentiment_pipeline(text[:512])[0]  # Limit length
            
            utterance_sentiments.append({
                'text': text[:100],  # Store preview
                'sentiment': self.sentiment_mapping.get(result['label'], result['label']),
                'score': float(round(result['score'], 3)),  # Convert NumPy float to Python float
                'raw_label': result['label']
            })
        
        # Calculate overall sentiment
        overall = self._calculate_overall_sentiment(utterance_sentiments)
        
        return {
            'overall_sentiment': overall['sentiment'],
            'sentiment_score': overall['score'],
            'confidence': overall['confidence'],
            'utterance_sentiments': utterance_sentiments,
            'sentiment_progression': self._track_sentiment_progression(utterance_sentiments)
        }
    
    def _calculate_overall_sentiment(self, utterance_sentiments: List[Dict]) -> Dict:
        """Calculate overall sentiment from utterances"""
        if not utterance_sentiments:
            return {
                'sentiment': 'Neutral',
                'score': 0.5,
                'confidence': 0.0
            }
        
        # Count sentiment types
        sentiment_counts = {}
        total_score = 0
        
        for us in utterance_sentiments:
            sentiment = us['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_score += us['score']
        
        # Get most common sentiment
        overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        avg_score = total_score / len(utterance_sentiments)
        
        # Calculate confidence based on consistency
        max_count = sentiment_counts[overall_sentiment]
        confidence = max_count / len(utterance_sentiments)
        
        return {
            'sentiment': overall_sentiment,
            'score': float(round(avg_score, 3)),  # Convert to native Python float
            'confidence': float(round(confidence, 3))  # Convert to native Python float
        }
    
    def _track_sentiment_progression(self, utterance_sentiments: List[Dict]) -> List[str]:
        """Track how sentiment changes through conversation"""
        return [us['sentiment'] for us in utterance_sentiments]