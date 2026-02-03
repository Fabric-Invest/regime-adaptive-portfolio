#!/usr/bin/env python3
"""
Regime-Adaptive Portfolio Strategy

Main entry point for the Fabric SDK strategy.
This strategy implements dynamic portfolio allocation based on
detected market regimes (Bull/Neutral/Bear).
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.strategy import RegimeAdaptiveStrategy

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the strategy."""
    logger.info("=" * 60)
    logger.info("Starting Regime-Adaptive Portfolio Strategy")
    logger.info("=" * 60)
    
    try:
        # Initialize strategy
        strategy = RegimeAdaptiveStrategy()
        
        # Run the strategy
        strategy.run()
        
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
    except Exception as e:
        logger.exception(f"Strategy error: {e}")
        raise


if __name__ == "__main__":
    main()
