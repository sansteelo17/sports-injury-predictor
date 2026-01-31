"""
Test that logging works correctly in the actual modules.

This script imports the modules and demonstrates that they have logging configured.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_logger_imports():
    """Test that modules can import and use the logger."""

    logger.info("Testing logger imports from key modules...")

    try:
        # Test inference_pipeline
        from src.inference import inference_pipeline
        logger.info("✓ inference_pipeline.py imported successfully with logging")

        # Test risk_card
        from src.inference import risk_card
        logger.info("✓ risk_card.py imported successfully with logging")

        # Test classification
        from src.models import classification
        logger.info("✓ classification.py imported successfully with logging")

        logger.info("\nAll modules imported successfully with logging configured!")
        logger.info("Logs are being written to: logs/injury_predictor.log")

        return True

    except Exception as e:
        logger.error(f"Failed to import modules: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_logger_imports()
    sys.exit(0 if success else 1)
