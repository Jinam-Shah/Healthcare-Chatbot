#!/usr/bin/env python3
"""
Healthcare Chatbot - Interactive Interface
"""

import torch
import json
import random
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.neural_net import LSTMAttentionChatbot
from utils.nltk_utils import tokenize, stem

# Paths
MODEL_PATH = Path('outputs/models/best_model.pth')
INTENTS_PATH = Path('data/intents_working.json')


class HealthcareChatbot:
    def __init__(self):
        print("\n" + "=" * 60)
        print("Loading Healthcare Chatbot...")
        print("=" * 60)

        # Set device
        self.device = torch.device('cpu')

        try:
            # Load model checkpoint
            print("Loading model checkpoint...")
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)

            # Extract vocabulary from checkpoint
            print("Loading vocabulary...")
            self.word2idx = checkpoint['word2idx']
            self.idx2word = checkpoint['idx2word']
            self.tag2idx = checkpoint['tag2idx']
            self.idx2tag = checkpoint['idx2tag']

            # Load intents
            print("Loading intents...")
            with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
                intents_data = json.load(f)
                self.intents = intents_data['intents']

            # Initialize model
            print("Initializing model...")
            vocab_size = checkpoint['vocab_size']
            num_classes = checkpoint['num_classes']

            # Get model config from checkpoint
            model_config = checkpoint.get('model_config', {})

            self.model = LSTMAttentionChatbot(
                vocab_size=vocab_size,
                embedding_dim=model_config.get('embedding_dim', 256),
                hidden_dim=model_config.get('hidden_dim', 512),
                num_classes=num_classes,
                num_layers=model_config.get('num_layers', 3),
                dropout=model_config.get('dropout', 0.5),
                bidirectional=model_config.get('bidirectional', True),
                use_attention=model_config.get('attention', True)
            )

            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            print("\n" + "=" * 60)
            print("Healthcare Chatbot Loaded Successfully!")
            print("=" * 60)
            print(f"  Vocabulary Size: {vocab_size}")
            print(f"  Intent Categories: {num_classes}")
            print(f"  Test Accuracy: {checkpoint.get('test_accuracy', 0):.2%}")
            print(f"  Test F1 Score: {checkpoint.get('test_f1', 0):.2%}")
            print("=" * 60 + "\n")

        except FileNotFoundError as e:
            print(f"\nError: Model file not found!")
            print(f"   {e}")
            print("\nPlease train the model first:")
            print("   python run.py train\n")
            sys.exit(1)
        except Exception as e:
            print(f"\nError loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def preprocess_input(self, text):
        """
        Preprocess user input exactly like training

        Args:
            text: User input string
        Returns:
            List of stemmed tokens
        """
        # Tokenize
        tokens = tokenize(text.lower())

        # Stem each token
        stemmed = [stem(token) for token in tokens]

        return stemmed

    def encode_sentence(self, text, max_len=50):
        """
        Encode sentence to tensor

        Args:
            text: User input string
            max_len: Maximum sequence length
        Returns:
            Tensor of word indices, actual length
        """
        # Preprocess
        words = self.preprocess_input(text)

        # Convert words to indices
        indices = []
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                # Use UNK token for unknown words
                indices.append(self.word2idx.get('<UNK>', 1))

        # Store actual length before padding
        actual_length = len(indices)

        # Pad or truncate
        if len(indices) < max_len:
            # Pad with zeros (PAD token)
            indices += [0] * (max_len - len(indices))
        else:
            # Truncate
            indices = indices[:max_len]
            actual_length = max_len

        # Convert to tensor
        return torch.tensor([indices], dtype=torch.long), actual_length

    def predict(self, sentence):
        """
        Predict intent and get response

        Args:
            sentence: User input string
        Returns:
            (predicted_tag, response, confidence)
        """
        try:
            # Encode input (now returns tensor AND length)
            input_tensor, actual_length = self.encode_sentence(sentence)
            input_tensor = input_tensor.to(self.device)

            # Create lengths tensor
            lengths = torch.tensor([actual_length], dtype=torch.long).to(self.device)

            # Predict with lengths parameter (CRITICAL FIX!)
            with torch.no_grad():
                output = self.model(input_tensor, lengths)

                # Handle tuple output (with attention)
                if isinstance(output, tuple):
                    output, attention_weights = output
                else:
                    attention_weights = None

                # Get probabilities
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

                # Convert to Python values
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()

                # Get tag name
                predicted_tag = self.idx2tag[predicted_idx]

            # Find response
            response = self.get_response(predicted_tag)

            return predicted_tag, response, confidence

        except Exception as e:
            print(f"\nPrediction error: {e}")
            import traceback
            traceback.print_exc()
            return "error", "I encountered an error processing your message.", 0.0

    def get_response(self, tag):
        """
        Get random response for predicted tag

        Args:
            tag: Predicted intent tag
        Returns:
            Response string
        """
        for intent in self.intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

        # Fallback
        return "I'm not sure how to respond to that. Could you rephrase?"

    def chat(self):
        """
        Run interactive chat loop
        """
        print("Healthcare Chatbot is ready!")
        print("Type 'quit', 'exit', or 'bye' to stop")
        print("Type 'debug' to see detailed prediction info")
        print("=" * 60 + "\n")

        debug_mode = False

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Check for empty input
                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\nBot: Goodbye! Take care of your health!\n")
                    break

                # Toggle debug mode
                if user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    status = "ON" if debug_mode else "OFF"
                    print(f"\nDebug mode: {status}\n")
                    continue

                # Predict
                tag, response, confidence = self.predict(user_input)

                # Select emoji based on confidence
                if confidence > 0.9:
                    conf_emoji = "🟢"  # High confidence
                elif confidence > 0.7:
                    conf_emoji = "🟡"  # Medium confidence
                else:
                    conf_emoji = "🟠"  # Low confidence

                # Display response
                print(f"\nBot: {response}")

                # Show debug info if enabled
                if debug_mode:
                    print(f"     Intent: {tag}")
                    print(f"     Confidence: {confidence:.4f}")
                    print(f"     Tokens: {self.preprocess_input(user_input)}")
                else:
                    print(f"     {conf_emoji} [{tag} | {confidence:.1%}]")

                print()

            except KeyboardInterrupt:
                print("\n\nBot: Goodbye! Stay healthy!\n")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                continue


def test_chatbot():
    """
    Quick test with predefined inputs
    """
    print("\n" + "=" * 60)
    print("Running Chatbot Tests")
    print("=" * 60 + "\n")

    chatbot = HealthcareChatbot()

    test_cases = [
        "Hello",
        "Hi there",
        "I have a fever",
        "I'm feeling sick",
        "My throat hurts",
        "I've been coughing",
        "I feel congested",
        "I have a headache",
        "How long does the flu last?",
        "Is this serious?",
        "Thank you",
        "Goodbye"
    ]

    print("Testing with predefined inputs:\n")
    print("=" * 60)

    for user_input in test_cases:
        tag, response, confidence = chatbot.predict(user_input)

        conf_emoji = "🟢" if confidence > 0.9 else "🟡" if confidence > 0.7 else "🟠"

        print(f"\nUser: {user_input}")
        print(f"Bot:  {response}")
        print(f"      {conf_emoji} [{tag} | {confidence:.1%}]")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_chatbot()
    else:
        try:
            chatbot = HealthcareChatbot()
            chatbot.chat()
        except Exception as e:
            print(f"\nFatal Error: {e}\n")
            import traceback

            traceback.print_exc()
            sys.exit(1)