import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_processor import TextProcessor

@pytest.fixture
def config():
    return {
        'processing': {
            'speaker_labels': {
                'doctor': ['Doctor', 'Physician', 'Dr.'],
                'patient': ['Patient', 'Pt.']
            }
        }
    }

@pytest.fixture
def text_processor(config):
    return TextProcessor(config)

def test_speaker_separation(text_processor):
    text = """
    Doctor: How are you?
    Patient: I'm fine.
    Doctor: Good to hear.
    """
    
    result = text_processor.process(text)
    
    assert len(result['doctor_utterances']) == 2
    assert len(result['patient_utterances']) == 1

def test_text_cleaning(text_processor):
    text = """
    Doctor:  How   are   you?  
    Patient: I'm  fine.
    """
    
    result = text_processor.process(text)
    
    # Check that extra spaces are removed
    assert '  ' not in result['full_text']

def test_empty_input(text_processor):
    text = ""
    
    result = text_processor.process(text)
    
    assert len(result['all_utterances']) == 0
    assert result['full_text'] == ""