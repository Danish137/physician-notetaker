# Physician Notetaker - Medical NLP System

An AI-powered system for medical transcription analysis, extracting structured medical information, performing sentiment analysis, and generating SOAP notes from physician-patient conversations.

## 🎯 Features

### 1. Medical NLP Summarization
- **Named Entity Recognition (NER)**: Extracts symptoms, treatments, diagnoses, and body parts
- **Structured Medical Reports**: Converts conversations to JSON format
- **Temporal Information Extraction**: Captures dates, durations, and quantities
- **Confidence Scoring**: Every extracted field includes confidence metrics

### 2. Sentiment & Intent Analysis
- **Patient Sentiment Detection**: Classifies as Anxious, Neutral, or Reassured
- **Intent Recognition**: Identifies patient intentions (seeking reassurance, reporting symptoms, etc.)
- **Conversation Flow Analysis**: Tracks sentiment changes throughout the dialogue

### 3. SOAP Note Generation (Bonus)
- **Automated SOAP Formatting**: Converts transcripts to clinical documentation
- **Section Classification**: Accurately maps content to Subjective, Objective, Assessment, Plan
- **Professional Clinical Language**: Transforms conversational text to medical terminology

## 📋 Requirements

- **Python**: 3.9+
- **GPU**: GTX 3050 or equivalent (4GB+ VRAM recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models and dependencies

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Danish137/physician-notetaker.git
cd physician-notetaker
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 4. Configure Environment

Copy the example environment file and add your API keys:
```bash
# Copy the example file
cp .envexample .env

# Edit .env and add your Groq API key
# Get your API key from: https://console.groq.com
```

The `.env` file should contain:
```bash
# Groq API Key (Required for LLM-based extraction)
GROQ_API_KEY=your_groq_api_key_here

# Optional: For Hugging Face gated models
HUGGINGFACE_TOKEN=your_hf_token_here

# GPU Configuration
USE_GPU=true
```


### 5. Run the System

#### Option 1: Using Jupyter Notebook
```bash
# Open the notebook
jupyter notebook notebooks/01_data_exploration.ipynb

# Connect the kernel and run all cells
# Results will be saved as JSON in data/output directory
```

#### Option 2: Using Command Line
```bash
# Process sample transcript
python main.py --input data/input/sample_transcript.txt

# Process with custom output directory
python main.py --input your_transcript.txt --output data/output/

# Use only local models (no API calls)
python main.py --input transcript.txt --local-only
```

## 📊 Usage Examples

### Example 1: Basic Medical Summarization
```python
from src.pipeline import MedicalNLPPipeline

# Initialize pipeline
pipeline = MedicalNLPPipeline(config_path="config/config.yaml")

# Input transcript
transcript = """
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
"""

# Process
results = pipeline.process(transcript)

# Access results
print(results['medical_summary'])
print(results['sentiment_analysis'])
print(results['soap_note'])
```

### Example 2: Custom Confidence Threshold
```python
pipeline = MedicalNLPPipeline(
    config_path="config/config.yaml",
    confidence_threshold=0.7
)

results = pipeline.process(transcript, include_reasoning=True)
```

## 📁 Project Structure
```
physician-notetaker/
├── README.md          # Project documentation
├── .envexample        # Example environment file (copy to .env and add your keys)
├── requirements.txt   # Python dependencies
├── config/            # Configuration files
├── data/              # Input/output data
├── src/               # Source code
│   ├── preprocessing/ # Text processing
│   ├── medical_nlp/   # Medical information extraction
│   ├── sentiment/     # Sentiment and intent analysis
│   ├── soap/          # SOAP note generation
│   └── utils/         # Utilities
├── notebooks/         # Jupyter notebooks for testing
├── tests/             # Unit tests
└── main.py            # Entry point
```

## 🧪 Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_medical_nlp.py
```

## 📈 Expected Output Format

### Medical Summary
```json
{
  "Patient_Name": {
    "value": "Janet Jones",
    "confidence": 0.95,
    "source": "stated"
  },
  "Symptoms": [
    {
      "value": "Neck pain",
      "confidence": 0.85,
      "source": "extracted"
    }
  ],
  "Diagnosis": {
    "value": "Whiplash injury",
    "confidence": 0.70,
    "source": "inferred",
    "reasoning": "Car accident + neck/back pain pattern"
  }
}
```

### Sentiment Analysis
```json
{
  "overall_sentiment": "Reassured",
  "sentiment_score": 0.75,
  "intent": "Seeking reassurance",
  "confidence": 0.82
}
```

### SOAP Note
```json
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Patient reports car accident with subsequent pain for four weeks"
  },
  "Objective": {
    "Physical_Exam": "Full range of motion, no tenderness"
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury and lower back strain"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy as needed",
    "Follow_Up": "Return if symptoms worsen"
  }
}
```

## 🎓 Model Information

### Local Models (No API Required)
1. **Medical NER**: `d4data/biomedical-ner-all` (400MB)
2. **Sentiment**: `distilbert-base-uncased-finetuned-sst-2-english` (260MB)
3. **Intent**: `facebook/bart-large-mnli` (1.6GB)

### Optional API Models (Enhanced Accuracy)
1. **Groq (Llama 3.3 70B)**: For structured extraction and SOAP generation
   - Fast inference with Groq's optimized infrastructure
   - Model: `llama-3.3-70b-versatile`
   - Get API key from: https://console.groq.com

**Total local storage**: ~2.5GB

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
- Model selection
- Confidence thresholds
- Output format
- Processing parameters
- SOAP note keywords

## 🔧 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode in .env
USE_GPU=false
```

### Out of Memory Error
```bash
# Reduce batch size in config.yaml
models:
  medical_ner:
    batch_size: 4  # Reduce from 8
```

### Model Download Issues
```bash
# Manually download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('d4data/biomedical-ner-all')"
```

## 📝 Key Design Decisions

### Handling Ambiguous/Missing Data
- **Confidence Scoring**: All fields include confidence metrics (0.0-1.0)
- **Source Tracking**: Mark data as "stated", "extracted", or "inferred"
- **Explicit Nulls**: Missing data clearly marked with reasoning
- **Multiple Hypotheses**: For ambiguous cases, provide alternatives

### Pre-trained Model Selection
- **PubMedBERT alternatives**: Used for medical domain knowledge
- **DistilBERT**: Lightweight, efficient for sentiment analysis
- **BART-MNLI**: Zero-shot classification without fine-tuning
- **Groq (Llama 3.3 70B)**: For complex structured extraction and SOAP generation

### Resource Optimization
- **Model Quantization**: Reduce memory footprint
- **Batch Processing**: Efficient GPU utilization
- **Lazy Loading**: Models loaded only when needed
- **CPU Fallback**: Automatic switching if GPU unavailable

## 🚧 Future Improvements

- [ ] Add multi-language support
- [ ] Implement active learning for continuous improvement
- [ ] Add web interface (Gradio/Streamlit)
- [ ] Add physician validation workflow


---
