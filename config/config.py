#!/usr/bin/env python3
"""
Healthcare Chatbot - Active Configuration
"""

import os

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
LOG_DIR = os.path.join(BASE_DIR, 'outputs', 'logs')
METRICS_DIR = os.path.join(BASE_DIR, 'outputs', 'metrics')

# ===== DATA FILES =====
INTENTS_FILE = os.path.join(DATA_DIR, 'intents.json')
SYNONYMS_FILE = os.path.join(DATA_DIR, 'medical_synonyms.json')
AUGMENTED_INTENTS = os.path.join(DATA_DIR, 'augmented_intents.json')

# ===== MODEL FILES =====
MODEL_CHECKPOINT = os.path.join(MODEL_DIR, 'best_model.pth')
TRAINING_DATA = os.path.join(MODEL_DIR, 'training_data.pkl')
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, 'embeddings.pkl')

# ===== MODEL HYPERPARAMETERS =====
MODEL_CONFIG = {
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 3,
    'dropout': 0.7,
    'bidirectional': True,
    'attention': True
}

# ===== TRAINING PARAMETERS =====
TRAINING_CONFIG = {
    'num_epochs': 1000,
    'batch_size': 16,
    'learning_rate': 0.003,
    'weight_decay': 1e-5,
    'early_stopping_patience': 30,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}

# ===== INFERENCE PARAMETERS =====
INFERENCE_CONFIG = {
    'confidence_threshold': 0.75,
    'max_response_length': 200,
    'context_window': 5,
    'fallback_responses': [
        "I'm not sure I understand. Could you rephrase that?",
        "I don't have enough information about that. Can you be more specific?",
        "That's outside my current knowledge. Please consult a healthcare professional."
    ]
}

# ===== LOGGING =====
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_conversations': True,
    'log_predictions': True,
    'max_log_size_mb': 100
}

# ===== API SETTINGS =====
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False
}

# ===== NLP SETTINGS =====
NLP_CONFIG = {
    'use_stemming': True,
    'use_lemmatization': False,
    'min_word_length': 2,
    'remove_stopwords': False,
    'lowercase': True
}

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
