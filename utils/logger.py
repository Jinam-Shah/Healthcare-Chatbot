import logging
import colorlog
from datetime import datetime
import os
import json


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup colored logger with file output

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    Returns:
        logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Color formatter
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # File handler (if log file specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Simple formatter for file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class ConversationLogger:
    """
    Log chatbot conversations to JSON
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create new log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"conversation_{timestamp}.json")

        self.conversations = []

    def log_interaction(self, user_input, bot_response, tag, confidence, metadata=None):
        """
        Log a single conversation turn

        Args:
            user_input: User's message
            bot_response: Bot's response
            tag: Predicted intent tag
            confidence: Prediction confidence
            metadata: Additional data (e.g., attention weights)
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'bot': bot_response,
            'intent': tag,
            'confidence': float(confidence),
            'metadata': metadata or {}
        }

        self.conversations.append(interaction)

        # Save after each interaction
        self.save()

    def save(self):
        """Save conversations to JSON file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)

    def get_stats(self):
        """Get conversation statistics"""
        if not self.conversations:
            return {}

        avg_confidence = sum(c['confidence'] for c in self.conversations) / len(self.conversations)
        intents = [c['intent'] for c in self.conversations]
        intent_counts = {intent: intents.count(intent) for intent in set(intents)}

        return {
            'total_interactions': len(self.conversations),
            'average_confidence': avg_confidence,
            'intent_distribution': intent_counts
        }
