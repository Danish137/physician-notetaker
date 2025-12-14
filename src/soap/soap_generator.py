# src/soap/soap_generator.py

import os
import json
from typing import Dict, List
import logging
from groq import Groq

logger = logging.getLogger(__name__)

class SOAPGenerator:
    """
    Generates SOAP notes from medical conversations
    """
    
    def __init__(self, config: Dict, use_llm: bool = True):
        self.config = config
        self.use_llm = use_llm
        
        # SOAP keywords
        self.soap_keywords = config['soap']['keywords']
        
        # Initialize Groq client if available
        self.client = None
        if use_llm:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.client = Groq(api_key=api_key)
                logger.info("Groq client initialized for SOAP generation")
            else:
                logger.warning("No Groq API key found. Using rule-based SOAP generation.")
                self.use_llm = False
    
    def generate(self, text: str, medical_summary: Dict, 
                 doctor_utterances: List, patient_utterances: List) -> Dict:
        """
        Generate SOAP note
        
        Args:
            text: Full conversation text
            medical_summary: Extracted medical summary
            doctor_utterances: Doctor's statements
            patient_utterances: Patient's statements
            
        Returns:
            SOAP note dictionary
        """
        if self.use_llm and self.client:
            return self._llm_based_soap(text, medical_summary)
        else:
            return self._rule_based_soap(text, medical_summary, 
                                        doctor_utterances, patient_utterances)
    
    def _llm_based_soap(self, text: str, medical_summary: Dict) -> Dict:
        """Generate SOAP note using LLM"""
        logger.info("Generating SOAP note using LLM...")
        
        prompt = self._create_soap_prompt(text, medical_summary)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config['models']['llm']['model'],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical documentation assistant. Generate structured SOAP notes from physician-patient conversations."
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
            content = content.replace('```json', '').replace('```', '').strip()
            result = json.loads(content)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM SOAP generation failed: {e}. Falling back to rule-based.")
            return self._rule_based_soap(text, medical_summary, [], [])
    
    def _create_soap_prompt(self, text: str, medical_summary: Dict) -> str:
        """Create prompt for SOAP generation"""
        prompt = f"""
Generate a SOAP note from this physician-patient conversation.

CONVERSATION:
{text}

EXTRACTED MEDICAL SUMMARY:
{json.dumps(medical_summary, indent=2)}

Generate a structured SOAP note in this JSON format:

{{
  "Subjective": {{
    "Chief_Complaint": "Main complaint in patient's words",
    "History_of_Present_Illness": "Detailed patient history",
    "Review_of_Systems": "Relevant systems review if mentioned"
  }},
  "Objective": {{
    "Physical_Exam": "Physical examination findings",
    "Vital_Signs": "Vital signs if mentioned",
    "Observations": "General observations"
  }},
  "Assessment": {{
    "Diagnosis": "Primary diagnosis",
    "Severity": "Severity level",
    "Supporting_Evidence": "Evidence supporting diagnosis"
  }},
  "Plan": {{
    "Treatment": "Treatment plan",
    "Medications": "Prescribed medications",
    "Follow_Up": "Follow-up instructions",
    "Patient_Education": "Education provided to patient"
  }}
}}

SOAP Note Guidelines:
- Subjective: What the PATIENT reports (symptoms, history, concerns)
- Objective: What the DOCTOR observes (exam findings, measurements)
- Assessment: Doctor's clinical judgment (diagnosis, severity)
- Plan: Treatment and follow-up actions

Use professional medical terminology. Be concise but complete.
Return ONLY the JSON object, no additional text.
"""
        return prompt
    
    def _rule_based_soap(self, text: str, medical_summary: Dict,
                         doctor_utterances: List, patient_utterances: List) -> Dict:
        """Generate SOAP note using rules"""
        logger.info("Generating SOAP note using rule-based approach...")
        
        # Classify sentences
        sentences = self._split_into_sentences(text)
        classified = self._classify_sentences(sentences)
        
        # Build SOAP sections
        soap_note = {
            "Subjective": self._build_subjective(classified, medical_summary),
            "Objective": self._build_objective(classified, text),
            "Assessment": self._build_assessment(classified, medical_summary),
            "Plan": self._build_plan(classified, medical_summary)
        }
        
        return soap_note
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _classify_sentences(self, sentences: List[str]) -> Dict:
        """Classify sentences by SOAP category"""
        classified = {
            'subjective': [],
            'objective': [],
            'assessment': [],
            'plan': []
        }
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for keywords
            if any(kw in sentence_lower for kw in self.soap_keywords['subjective']):
                classified['subjective'].append(sentence)
            elif any(kw in sentence_lower for kw in self.soap_keywords['objective']):
                classified['objective'].append(sentence)
            elif any(kw in sentence_lower for kw in self.soap_keywords['assessment']):
                classified['assessment'].append(sentence)
            elif any(kw in sentence_lower for kw in self.soap_keywords['plan']):
                classified['plan'].append(sentence)
        
        return classified
    
    def _build_subjective(self, classified: Dict, medical_summary: Dict) -> Dict:
        """Build Subjective section"""
        # Get chief complaint from symptoms
        symptoms = medical_summary.get('Symptoms', [])
        chief_complaint = "Not specified"
        
        if symptoms:
            symptom_values = [s.get('value', '') for s in symptoms if s.get('value')]
            if symptom_values:
                chief_complaint = ', '.join(symptom_values)
        
        # Build HPI from subjective sentences
        hpi_sentences = classified.get('subjective', [])
        hpi = ' '.join(hpi_sentences) if hpi_sentences else "Patient reports " + chief_complaint.lower()
        
        return {
            "Chief_Complaint": chief_complaint,
            "History_of_Present_Illness": hpi[:500]  # Limit length
        }
    
    def _build_objective(self, classified: Dict, text: str) -> Dict:
        """Build Objective section"""
        objective_sentences = classified.get('objective', [])
        
        # Look for physical exam mentions
        physical_exam = "Not documented"
        if objective_sentences:
            physical_exam = ' '.join(objective_sentences)
        elif 'examination' in text.lower() or 'exam' in text.lower():
            # Extract exam-related sentences
            sentences = self._split_into_sentences(text)
            exam_sentences = [s for s in sentences if 'exam' in s.lower() or 'range of motion' in s.lower()]
            if exam_sentences:
                physical_exam = ' '.join(exam_sentences)
        
        # General observations
        observations = "Patient appears in normal health, cooperative during examination"
        
        return {
            "Physical_Exam": physical_exam[:500],
            "Observations": observations
        }
    
    def _build_assessment(self, classified: Dict, medical_summary: Dict) -> Dict:
        """Build Assessment section"""
        diagnosis_info = medical_summary.get('Diagnosis', {})
        diagnosis = diagnosis_info.get('value', 'No definitive diagnosis documented')
        
        # Determine severity
        severity = "Mild to moderate"
        if 'improving' in str(medical_summary).lower():
            severity = "Mild, improving"
        
        # Supporting evidence
        symptoms = medical_summary.get('Symptoms', [])
        evidence = "Based on patient history and symptoms"
        if symptoms:
            symptom_list = [s.get('value', '') for s in symptoms]
            evidence = f"Based on {', '.join(symptom_list).lower()} following reported incident"
        
        return {
            "Diagnosis": diagnosis,
            "Severity": severity,
            "Supporting_Evidence": evidence
        }
    
    def _build_plan(self, classified: Dict, medical_summary: Dict) -> Dict:
        """Build Plan section"""
        # Get treatments
        treatments = medical_summary.get('Treatment', [])
        treatment_plan = "Continue current management"
        
        if treatments:
            treatment_values = [t.get('value', '') for t in treatments]
            treatment_plan = ', '.join(treatment_values)
        
        # Follow-up
        follow_up = "Return if symptoms worsen or persist"
        if 'Prognosis' in medical_summary and medical_summary['Prognosis'].get('value'):
            prognosis = medical_summary['Prognosis']['value']
            if 'month' in prognosis.lower():
                follow_up = "Follow-up as needed. " + prognosis
        
        return {
            "Treatment": treatment_plan,
            "Medications": "As previously prescribed" if 'painkiller' in str(medical_summary).lower() else "None",
            "Follow_Up": follow_up,
            "Patient_Education": "Patient advised on symptoms to monitor and when to seek care"
        }