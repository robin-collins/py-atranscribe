#!/usr/bin/env python3
"""Demo script to test the enhanced transcription system.

This script demonstrates the enhanced whisper transcription with flash attention
and distil-whisper models.
"""

import asyncio
import logging

from src.config import AppConfig
from src.transcription.enhanced_whisper import EnhancedWhisperTranscriber


async def demo_enhanced_transcription() -> None:
    """Demo the enhanced transcription functionality."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting Enhanced Transcription Demo")

    # Create a minimal config for testing
    config = AppConfig()

    # Initialize enhanced transcriber
    transcriber = EnhancedWhisperTranscriber(config.transcription.whisper)

    try:
        # Initialize the transcriber
        logger.info("Initializing enhanced transcriber...")
        await transcriber.initialize()

        logger.info("✅ Enhanced transcriber initialized successfully!")
        logger.info("Model: %s", config.transcription.whisper.enhanced_model)
        logger.info(
            "Flash Attention: %s", config.transcription.whisper.use_flash_attention
        )

        # Note: For actual transcription, you would need an audio file
        logger.info("Demo completed - transcriber is ready for audio files")

    except Exception as e:
        logger.exception("❌ Demo failed: %s", e)
        raise
    finally:
        # Cleanup
        await transcriber.cleanup()
        logger.info("Demo cleanup completed")


if __name__ == "__main__":
    asyncio.run(demo_enhanced_transcription())
