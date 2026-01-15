#!/usr/bin/env python3
"""
Healthcare Chatbot - Main Entry Point
Simple interface to run different components
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description='Healthcare Chatbot - NLP Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train      # Train the model
  python run.py chat       # Run interactive chatbot
  python run.py test       # Quick test with sample inputs
        """
    )

    parser.add_argument(
        'command',
        choices=['train', 'chat', 'test'],
        help='Command to execute'
    )

    args = parser.parse_args()

    # Execute command
    if args.command == 'train':
        print("\n Starting model training...\n")
        # Try to import from either train_FIXED or train
        try:
            import train_FIXED as train_module
        except ImportError:
            import train as train_module
        train_module.main()

    elif args.command == 'chat':
        print("\n Starting interactive chatbot...\n")
        from chatbot import HealthcareChatbot
        try:
            chatbot = HealthcareChatbot()
            chatbot.chat()
        except FileNotFoundError:
            print("\n Error: Model not found!")
            print("Please train the model first: python run.py train\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n Error: {e}\n")
            sys.exit(1)

    elif args.command == 'test':
        print("\n Running quick test...\n")
        from chatbot import test_chatbot
        try:
            test_chatbot()
        except FileNotFoundError:
            print("\n Error: Model not found!")
            print("Please train the model first: python run.py train\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n Error: {e}\n")
            sys.exit(1)


if __name__ == '__main__':
    main()