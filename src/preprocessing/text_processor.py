# src/preprocessing/text_processor.py

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Utterance:
    """Represents a single utterance in the conversation"""
    speaker: str
    text: str
    index: int

class TextProcessor:
    """
    Handles text preprocessing for medical transcripts
    - Speaker diarization
    - Text cleaning
    - Sentence segmentation
    """
    
    def __init__(self, config: Dict):
        self.doctor_labels = config['processing']['speaker_labels']['doctor']
        self.patient_labels = config['processing']['speaker_labels']['patient']
    
    def process(self, text: str) -> Dict:
        """
        Main processing function
        
        Args:
            text: Raw transcript text
            
        Returns:
            Dict with separated utterances and cleaned text
        """
        # Separate speakers
        utterances = self._separate_speakers(text)
        
        # Clean text
        cleaned_utterances = [self._clean_text(u) for u in utterances]
        
        # Separate by speaker type
        doctor_utterances = [u for u in cleaned_utterances if u.speaker == "Doctor"]
        patient_utterances = [u for u in cleaned_utterances if u.speaker == "Patient"]
        
        return {
            'all_utterances': cleaned_utterances,
            'doctor_utterances': doctor_utterances,
            'patient_utterances': patient_utterances,
            'full_text': self._reconstruct_text(cleaned_utterances),
            'patient_text': self._reconstruct_text(patient_utterances),
            'doctor_text': self._reconstruct_text(doctor_utterances)
        }
    
    def _separate_speakers(self, text: str) -> List[Utterance]:
        """Separate text into speaker utterances"""
        utterances = []
        lines = text.strip().split('\n')
        
        current_speaker = None
        current_text = []
        index = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with speaker label
            speaker = self._identify_speaker(line)
            
            if speaker:
                # Save previous utterance
                if current_speaker and current_text:
                    utterances.append(Utterance(
                        speaker=current_speaker,
                        text=' '.join(current_text),
                        index=index
                    ))
                    index += 1
                    current_text = []
                
                # Start new utterance
                current_speaker = speaker
                # Remove speaker label from text
                text_part = self._remove_speaker_label(line)
                if text_part:
                    current_text.append(text_part)
            else:
                # Continue previous utterance
                if current_speaker:
                    current_text.append(line)
        
        # Add last utterance
        if current_speaker and current_text:
            utterances.append(Utterance(
                speaker=current_speaker,
                text=' '.join(current_text),
                index=index
            ))
        
        return utterances
    
    def _identify_speaker(self, line: str) -> str:
        """Identify speaker from line"""
        line_lower = line.lower()
        
        # Check for doctor labels
        for label in self.doctor_labels:
            if line_lower.startswith(label.lower() + ':') or \
               line_lower.startswith(label.lower() + ' :'):
                return "Doctor"
        
        # Check for patient labels
        for label in self.patient_labels:
            if line_lower.startswith(label.lower() + ':') or \
               line_lower.startswith(label.lower() + ' :'):
                return "Patient"
        
        return None
    
    def _remove_speaker_label(self, line: str) -> str:
        """Remove speaker label from beginning of line"""
        # Find first colon and remove everything before it
        if ':' in line:
            parts = line.split(':', 1)
            return parts[1].strip()
        return line
    
    def _clean_text(self, utterance: Utterance) -> Utterance:
        """Clean text of utterance"""
        text = utterance.text
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\']', '', text)
        
        # Normalize quotes
        text = text.replace('"', '').replace('"', '')
        
        return Utterance(
            speaker=utterance.speaker,
            text=text.strip(),
            index=utterance.index
        )
    
    def _reconstruct_text(self, utterances: List[Utterance]) -> str:
        """Reconstruct text from utterances"""
        return ' '.join([u.text for u in utterances])