# src/pipeline.py

import os
import yaml
import json
from typing import Dict, Optional
from datetime import datetime
import logging
from pathlib import Path

from .preprocessing.text_processor import TextProcessor
from .medical_nlp.entity_extractor import EntityExtractor
from .medical_nlp.temporal_extractor import TemporalExtractor
from .medical_nlp.medical_summarizer import MedicalSummarizer
from .sentiment.sentiment_analyzer import SentimentAnalyzer
from .sentiment.intent_detector import IntentDetector
from .soap.soap_generator import SOAPGenerator
from .utils.confidence_scorer import ConfidenceScorer
from .utils.validators import OutputValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalNLPPipeline:
    """
    Main pipeline orchestrating all components
    """
    
    def __init__(self, config_path: str = "config/config.yaml", use_llm: bool = True):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
            use_llm: Whether to use LLM for extraction (requires API key)
        """
        logger.info("=" * 80)
        logger.info("Initializing Medical NLP Pipeline")
        logger.info("=" * 80)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.use_llm = use_llm
        
        # Check if API key is available
        if use_llm and not os.getenv('GROQ_API_KEY'):
            logger.warning("No Groq API key found. Disabling LLM features.")
            self.use_llm = False
        
        # Initialize components
        logger.info("Loading models and components...")
        self.text_processor = TextProcessor(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.temporal_extractor = TemporalExtractor(self.config)
        self.medical_summarizer = MedicalSummarizer(self.config, use_llm=self.use_llm)
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.intent_detector = IntentDetector(self.config)
        self.soap_generator = SOAPGenerator(self.config, use_llm=self.use_llm)
        
        logger.info("✓ Pipeline initialization complete")
        logger.info("=" * 80)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def process(self, text: str, include_reasoning: bool = True) -> Dict:
        """
        Process medical transcript through entire pipeline
        
        Args:
            text: Raw transcript text
            include_reasoning: Whether to include reasoning in output
            
        Returns:
            Complete analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting Medical Transcript Analysis")
        logger.info("=" * 80 + "\n")
        
        try:
            # Step 1: Preprocessing
            logger.info("[1/6] Text Preprocessing...")
            processed = self.text_processor.process(text)
            logger.info(f"  ✓ Identified {len(processed['doctor_utterances'])} doctor utterances")
            logger.info(f"  ✓ Identified {len(processed['patient_utterances'])} patient utterances")
            
            # Step 2: Entity Extraction
            logger.info("\n[2/6] Medical Entity Extraction...")
            entities = self.entity_extractor.extract_entities(processed['full_text'])
            logger.info(f"  ✓ Extracted {len(entities['symptoms'])} symptoms")
            logger.info(f"  ✓ Extracted {len(entities['treatments'])} treatments")
            logger.info(f"  ✓ Extracted {len(entities['body_parts'])} body parts")
            logger.info(f"  ✓ Extracted {len(entities['symptoms'])} symptoms")
            logger.info(f"  ✓ Extracted {len(entities['treatments'])} treatments")
            logger.info(f"  ✓ Extracted {len(entities['body_parts'])} body parts")
        
            # Step 3: Temporal Information
            logger.info("\n[3/6] Temporal Information Extraction...")
            temporal_info = self.temporal_extractor.extract_temporal_info(processed['full_text'])
            logger.info(f"  ✓ Extracted {len(temporal_info['durations'])} durations")
            logger.info(f"  ✓ Extracted {len(temporal_info['quantities'])} quantities")
            
            # Step 4: Medical Summary
            logger.info("\n[4/6] Medical Summary Generation...")
            medical_summary = self.medical_summarizer.generate_summary(
                processed['full_text'],
                entities,
                temporal_info
            )
            logger.info("  ✓ Medical summary generated")
            
            # Step 5: Sentiment & Intent Analysis
            logger.info("\n[5/6] Sentiment & Intent Analysis...")
            sentiment = self.sentiment_analyzer.analyze(processed['patient_utterances'])
            intent = self.intent_detector.detect(processed['patient_utterances'])
            logger.info(f"  ✓ Overall sentiment: {sentiment['overall_sentiment']}")
            logger.info(f"  ✓ Primary intent: {intent['primary_intent']}")
            
            # Step 6: SOAP Note Generation
            logger.info("\n[6/6] SOAP Note Generation...")
            soap_note = self.soap_generator.generate(
                processed['full_text'],
                medical_summary,
                processed['doctor_utterances'],
                processed['patient_utterances']
            )
            logger.info("  ✓ SOAP note generated")
            
            # Calculate overall confidence
            overall_confidence = ConfidenceScorer.calculate_overall_confidence(medical_summary)
            
            # Validate outputs
            logger.info("\n[Validation] Validating outputs...")
            validation_results = {
                'medical_summary': OutputValidator.validate_medical_summary(medical_summary),
                'soap_note': OutputValidator.validate_soap_note(soap_note),
                'sentiment': OutputValidator.validate_sentiment_analysis(sentiment)
            }
            
            # Log validation warnings
            for component, validation in validation_results.items():
                if validation['warnings']:
                    for warning in validation['warnings']:
                        logger.warning(f"  {component}: {warning}")
            
            # Compile results
            results = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': '1.0.0',
                    'llm_used': self.use_llm,
                    'overall_confidence': overall_confidence
                },
                'medical_summary': medical_summary,
                'sentiment_analysis': {
                    'overall_sentiment': sentiment['overall_sentiment'],
                    'sentiment_score': sentiment['sentiment_score'],
                    'confidence': sentiment['confidence'],
                    'intent': intent['primary_intent'],
                    'intent_confidence': intent['confidence'],
                    'all_intents': intent['all_intents']
                },
                'soap_note': soap_note,
                'validation': validation_results
            }
            
            # Remove reasoning if not requested
            if not include_reasoning:
                results = self._remove_reasoning(results)
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ Analysis Complete")
            logger.info("=" * 80 + "\n")
            
            return results
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}")
            raise

    def _remove_reasoning(self, results: Dict) -> Dict:
        """Remove reasoning fields from results"""
        def remove_field(obj, field_name):
            if isinstance(obj, dict):
                obj.pop(field_name, None)
                for value in obj.values():
                    remove_field(value, field_name)
            elif isinstance(obj, list):
                for item in obj:
                    remove_field(item, field_name)
        
        remove_field(results, 'reasoning')
        return results

    def process_file(self, input_path: str, output_dir: str = "data/output") -> str:
        """
        Process transcript from file and save results
        
        Args:
            input_path: Path to input transcript file
            output_dir: Directory to save outputs
            
        Returns:
            Path to output file
        """
        logger.info(f"Processing file: {input_path}")
        
        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process
        results = self.process(text)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"analysis_{timestamp}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Results saved to: {output_path}")
        
        return output_path