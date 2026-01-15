#!/usr/bin/env python3
"""
Healthcare Chatbot - Training Script
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import LSTMAttentionChatbot, DataProcessor
from models.embeddings import Word2VecTrainer
from models.data_processor import ChatDataset
from utils.evaluation import (
    ModelEvaluator, TrainingMonitor, split_data, evaluate_model
)
from utils.logger import setup_logger

from config.config import (
    INTENTS_FILE, MODEL_CHECKPOINT, TRAINING_DATA, EMBEDDINGS_FILE,
    MODEL_CONFIG, TRAINING_CONFIG, MODEL_DIR, METRICS_DIR
)


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels, lengths in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        if model.use_attention:
            outputs, _ = model(inputs, lengths)
        else:
            outputs = model(inputs, lengths)

        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, lengths in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            if model.use_attention:
                outputs, _ = model(inputs, lengths)
            else:
                outputs = model(inputs, lengths)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    logger = setup_logger('Training', log_file=os.path.join(MODEL_DIR, 'training.log'))
    logger.info("=" * 60)
    logger.info("Healthcare Chatbot - FIXED Training")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ===== STEP 1: Data Processing =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Data Processing")
    logger.info("=" * 60)

    processor = DataProcessor(
        max_seq_length=50,
        use_stemming=True
    )

    intents_file = INTENTS_FILE.replace('intents.json', 'intents_working.json')
    intents = processor.load_intents(intents_file)
    xy = processor.build_vocabulary(intents)

    # ===== CRITICAL FIX: Save original vocabulary BEFORE Word2Vec =====
    original_word2idx = processor.word2idx.copy()
    original_idx2word = processor.idx2word.copy()

    logger.info(f"\n Original vocabulary saved: {len(original_word2idx)} words")

    # ===== STEP 2: Train Word2Vec Embeddings =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Training Word2Vec Embeddings")
    logger.info("=" * 60)

    sentences = processor.get_tokenized_sentences(intents)
    w2v_trainer = Word2VecTrainer(
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        window=5,
        min_count=1
    )

    embedding_matrix_w2v = w2v_trainer.train(sentences, save_path=EMBEDDINGS_FILE)

    # ===== CRITICAL FIX: Create embedding matrix using ORIGINAL vocabulary =====
    logger.info("\n Aligning embeddings with original vocabulary...")

    vocab_size = len(original_word2idx)
    embedding_matrix = np.zeros((vocab_size, MODEL_CONFIG['embedding_dim']))

    # Map embeddings from Word2Vec to original vocabulary
    matched = 0
    for word, idx in original_word2idx.items():
        if word in ['<PAD>', '<UNK>']:
            # Keep as zeros or random
            if word == '<UNK>':
                embedding_matrix[idx] = np.random.normal(0, 0.1, MODEL_CONFIG['embedding_dim'])
        elif word in w2v_trainer.model.wv:
            embedding_matrix[idx] = w2v_trainer.model.wv[word]
            matched += 1
        else:
            # Random for words not in Word2Vec
            embedding_matrix[idx] = np.random.normal(0, 0.1, MODEL_CONFIG['embedding_dim'])

    logger.info(f" Matched {matched}/{vocab_size} words to Word2Vec embeddings")

    # ===== CRITICAL FIX: Restore original vocabulary =====
    processor.word2idx = original_word2idx
    processor.idx2word = original_idx2word

    logger.info(" Using original vocabulary for training")

    # ===== STEP 3: Prepare Training Data =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Preparing Training Data")
    logger.info("=" * 60)

    X, y, lengths = processor.prepare_training_data(xy)

    train_data, val_data, test_data = split_data(
        X, y, lengths,
        train_size=TRAINING_CONFIG['train_split'],
        val_size=TRAINING_CONFIG['val_split'],
        test_size=TRAINING_CONFIG['test_split']
    )

    X_train, y_train, len_train = train_data
    X_val, y_val, len_val = val_data
    X_test, y_test, len_test = test_data

    train_dataset = ChatDataset(X_train, y_train, len_train)
    val_dataset = ChatDataset(X_val, y_val, len_val)
    test_dataset = ChatDataset(X_test, y_test, len_test)

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)

    # ===== STEP 4: Model Initialization =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Model Initialization")
    logger.info("=" * 60)

    vocab_size = len(processor.word2idx)
    num_classes = len(processor.tags)

    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Number of classes: {num_classes}")

    model = LSTMAttentionChatbot(
        vocab_size=vocab_size,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        num_classes=num_classes,
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout'],
        bidirectional=MODEL_CONFIG['bidirectional'],
        pretrained_embeddings=embedding_matrix,  # Use aligned embeddings
        use_attention=MODEL_CONFIG['attention']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # ===== STEP 5: Training Setup =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Training Setup")
    logger.info("=" * 60)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    early_stopping = EarlyStopping(patience=30)  # Fixed: 30 instead of 150
    monitor = TrainingMonitor(save_dir=METRICS_DIR)

    # ===== STEP 6: Training Loop =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Training Model")
    logger.info("=" * 60)

    best_val_acc = 0
    num_epochs = TRAINING_CONFIG['num_epochs']

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        monitor.update(epoch + 1, train_loss, val_loss, train_acc, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"[BEST] New best model saved! Val Acc: {val_acc:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered!")
            model.load_state_dict(early_stopping.best_model)
            break

        if (epoch + 1) % 10 == 0:
            monitor.plot_training_history()

    # ===== STEP 7: Final Evaluation =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Final Evaluation")
    logger.info("=" * 60)

    monitor.plot_training_history()

    evaluator = ModelEvaluator(model, device, processor.tags, save_dir=METRICS_DIR)
    logger.info("\nEvaluating on test set...")
    test_metrics = evaluator.evaluate_batch(test_loader)
    evaluator.print_metrics(test_metrics, dataset_name='Test')
    evaluator.generate_full_report()

    # ===== STEP 8: Save Final Model =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Saving Model")
    logger.info("=" * 60)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'processor': processor,
        'model_config': MODEL_CONFIG,
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'tags': processor.tags,
        'word2idx': processor.word2idx,
        'idx2word': processor.idx2word,
        'tag2idx': processor.tag2idx,
        'idx2tag': processor.idx2tag,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1_weighted']
    }

    torch.save(checkpoint, MODEL_CHECKPOINT)
    logger.info(f"Model saved to {MODEL_CHECKPOINT}")

    processor.save(TRAINING_DATA)

    summary = {
        'model': 'LSTMAttentionChatbot',
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_weighted': float(test_metrics['f1_weighted']),
        'best_val_accuracy': float(best_val_acc),
        'epochs_trained': epoch + 1
    }

    summary_path = os.path.join(MODEL_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training summary saved to {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {test_metrics['f1_weighted']:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()