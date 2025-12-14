import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from colorama import init, Fore, Style

from src.pipeline import MedicalNLPPipeline

# Initialize colorama for colored output
init(autoreset=True)

def print_header():
    """Print application header"""
    print("\n" + "=" * 80)
    print(Fore.CYAN + Style.BRIGHT + "  PHYSICIAN NOTETAKER - Medical NLP System")
    print("=" * 80 + "\n")

def print_results(results: dict):
    """Pretty print results"""
    print("\n" + Fore.GREEN + "=" * 80)
    print(Fore.GREEN + "RESULTS")
    print(Fore.GREEN + "=" * 80 + "\n")
    
    # Medical Summary
    print(Fore.YELLOW + Style.BRIGHT + "📋 MEDICAL SUMMARY")
    print(Fore.YELLOW + "-" * 80)
    print(json.dumps(results['medical_summary'], indent=2))
    
    # Sentiment Analysis
    print("\n" + Fore.YELLOW + Style.BRIGHT + "😊 SENTIMENT & INTENT ANALYSIS")
    print(Fore.YELLOW + "-" * 80)
    print(json.dumps(results['sentiment_analysis'], indent=2))
    
    # SOAP Note
    print("\n" + Fore.YELLOW + Style.BRIGHT + "📝 SOAP NOTE")
    print(Fore.YELLOW + "-" * 80)
    print(json.dumps(results['soap_note'], indent=2))
    
    # Metadata
    print("\n" + Fore.CYAN + Style.BRIGHT + "ℹ️  METADATA")
    print(Fore.CYAN + "-" * 80)
    print(f"Timestamp: {results['metadata']['timestamp']}")
    print(f"Overall Confidence: {results['metadata']['overall_confidence']}")
    print(f"LLM Used: {results['metadata']['llm_used']}")
    
    print("\n" + Fore.GREEN + "=" * 80 + "\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Medical NLP System for physician-patient conversation analysis'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input transcript file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/output',
        help='Output directory (default: data/output)'
    )
    parser.add_argument(
        '--local-only',
        action='store_true',
        help='Use only local models (no API calls)'
    )
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Exclude reasoning from output'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Print header
    print_header()
    
    # Initialize pipeline
    use_llm = not args.local_only
    pipeline = MedicalNLPPipeline(use_llm=use_llm)
    
    # Get input
    if args.input:
        # Process from file
        if not Path(args.input).exists():
            print(Fore.RED + f"✗ Error: Input file not found: {args.input}")
            sys.exit(1)
        
        output_path = pipeline.process_file(args.input, args.output)
        
        # Load and display results
        with open(output_path, 'r') as f:
            results = json.load(f)
        
        print_results(results)
        print(Fore.GREEN + f"✓ Full results saved to: {output_path}\n")
        
    else:
        # Use default sample
        print(Fore.CYAN + "No input file specified. Using sample transcript...\n")
        
        sample_text = """
Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.
"""
        
        print(Fore.WHITE + "INPUT TRANSCRIPT:")
        print(Fore.WHITE + "-" * 80)
        print(sample_text)
        print(Fore.WHITE + "-" * 80 + "\n")
        
        results = pipeline.process(sample_text, include_reasoning=not args.no_reasoning)
        
        print_results(results)
        
        # Save to file
        Path(args.output).mkdir(parents=True, exist_ok=True)
        output_path = Path(args.output) / "sample_output.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(Fore.GREEN + f"✓ Results saved to: {output_path}\n")

if __name__ == "__main__":
    main()
