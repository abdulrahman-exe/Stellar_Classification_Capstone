"""
Run the complete stellar classification training pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipelines import train_pipeline
from src.pipelines.train_pipeline import TrainingPipeline
import logging

# Setup logging to see output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stellar_classification.log')
    ]
)

def main():
    print("🚀 Starting Stellar Classification Training Pipeline...")
    print("=" * 60)
    
    try:
        # Create pipeline
        pipeline = TrainingPipeline()
        
        # Validate pipeline first
        print("🔍 Validating pipeline components...")
        if not pipeline.validate_pipeline():
            print("❌ Pipeline validation failed!")
            return
        
        print("✅ Pipeline validation passed!")
        print()
        
        # Run pipeline
        print("🏃 Running complete training pipeline...")
        results = pipeline.run_pipeline()
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"📊 Best model: {results['best_model']}")
        print(f"📈 Best accuracy: {results['best_accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
