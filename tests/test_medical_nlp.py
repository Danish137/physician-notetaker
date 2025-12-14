import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.medical_nlp.temporal_extractor import TemporalExtractor

@pytest.fixture
def config():
    return {}

@pytest.fixture
def temporal_extractor(config):
    return TemporalExtractor(config)

def test_duration_extraction(temporal_extractor):
    text = "The pain lasted four weeks and required treatment."
    
    result = temporal_extractor.extract_temporal_info(text)
    
    assert len(result['durations']) > 0
    assert any('4' in str(d['value']) or 'four' in str(d['value']).lower() 
               for d in result['durations'])

def test_quantity_extraction(temporal_extractor):
    text = "I had ten physiotherapy sessions last month."
    
    result = temporal_extractor.extract_temporal_info(text)
    
    assert len(result['quantities']) > 0
    assert any(q['value'] == 10 for q in result['quantities'])