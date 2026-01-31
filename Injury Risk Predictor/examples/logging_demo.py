"""
Example demonstrating the logging framework in action.

This script shows how logging is integrated throughout the codebase.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

# Create logger for this module
logger = get_logger(__name__)


def main():
    """Demonstrate the logging framework."""

    logger.info("=" * 60)
    logger.info("Injury Risk Predictor - Logging Framework Demo")
    logger.info("=" * 60)

    # Simulate different log levels
    logger.debug("DEBUG: Detailed information for diagnosing problems")
    logger.info("INFO: General informational messages about progress")
    logger.warning("WARNING: Something unexpected happened, but still working")
    logger.error("ERROR: A serious problem occurred")

    logger.info("\nLogging is configured to:")
    logger.info("  - Console output: INFO level and above")
    logger.info("  - File output: DEBUG level and above (logs/injury_predictor.log)")
    logger.info("  - Format: timestamp - module - level - message")

    logger.info("\nExample usage in key modules:")
    logger.info("  1. inference_pipeline.py: Logs prediction start/end, errors")
    logger.info("  2. risk_card.py: Logs risk card generation for each player")
    logger.info("  3. classification.py: Logs model training start/end")

    logger.info("\nCheck logs/injury_predictor.log for the full log history!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
