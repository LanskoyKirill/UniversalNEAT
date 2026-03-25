#!/usr/bin/env python3
"""
NEAT Model Inference Script
Loads and runs the trained NEAT neural network model for text generation.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import re


class NEATModel:
    """Load and run inference on a trained NEAT model."""

    def __init__(self, model_path):
        """Initialize model from JSON file."""
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'r') as f:
            # Read and fix locale-based decimal separators (,950000 -> .950000)
            content = f.read()
            # Replace comma decimals with period decimals in number contexts
            content = re.sub(r'(\d),(\d{6})', r'\1.\2', content)
            self.data = json.loads(content)

        # Extract model parameters
        self.vocabulary = self.data['vocabulary']
        self.vocab_size = self.data['vocabSize']
        self.neuron_count = self.data['neuronCount']
        self.output_start = self.data['outputStart']
        self.rnn_decay = self.data.get('rnnDecay', 0.95)
        self.input_start = self.data.get('inputStart', 2)
        self.bias_index = self.data.get('biasIndex', 0)

        # Special tokens
        self.pad_token = self.data.get('padToken', 0)
        self.sos_token = self.data.get('sosToken', 1)
        self.eos_token = self.data.get('eosToken', 2)
        self.unk_token = self.data.get('unkToken', 3)

        # Evaluation order (topological sort)
        self.order = self.data['order']

        # Build network structure
        self._build_network()

    def _build_network(self):
        """Build adjacency list from connections."""
        self.adj_list = [dict() for _ in range(self.neuron_count)]
        self.rnn_connections = []

        for conn_data in self.data['connections']:
            from_idx = conn_data['from']
            to_idx = conn_data['to']
            weight = conn_data['weight']
            enabled = conn_data['enabled']
            recurrent = conn_data.get('recurrent', False)

            # Only add enabled connections
            if enabled:
                if from_idx < len(self.adj_list):
                    self.adj_list[from_idx][to_idx] = weight

                # Track RNN connections
                if recurrent:
                    self.rnn_connections.append((from_idx, to_idx, weight))

    def tokenize(self, text):
        """Simple word-level tokenization."""
        words = text.lower().split()
        tokens = []
        for word in words:
            # Try to find exact match
            if word in self.vocabulary:
                tokens.append(self.vocabulary.index(word))
            else:
                # Use UNK token for unknown words
                tokens.append(self.unk_token)
        return tokens

    def detokenize(self, tokens):
        """Convert token IDs back to text."""
        words = []
        for token_id in tokens:
            if 0 <= token_id < len(self.vocabulary):
                word = self.vocabulary[token_id]
                # Skip special tokens in output
                if word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                    words.append(word)
        return ' '.join(words)

    def _reset_neurons(self):
        """Reset neuron activations."""
        self.neurons = np.zeros(self.neuron_count, dtype=np.float32)
        self.rnn_neurons = np.zeros(self.neuron_count, dtype=np.float32)
        # Bias neuron is always 1
        self.neurons[self.bias_index] = 1.0
        self.rnn_neurons[self.bias_index] = 1.0

    def _run_network(self):
        """Forward pass through the network."""
        for neuron_idx in self.order:
            # Apply activation to hidden neurons
            if neuron_idx >= self.input_start + self.vocab_size:
                self.neurons[neuron_idx] = np.tanh(self.neurons[neuron_idx])

            # Propagate to downstream neurons
            if neuron_idx in range(len(self.adj_list)):
                for to_idx, weight in self.adj_list[neuron_idx].items():
                    if to_idx < self.neuron_count:
                        self.neurons[to_idx] += weight * self.neurons[neuron_idx]

    def _update_rnn_state(self):
        """Update RNN hidden state."""
        # Update RNN neurons from connections
        for from_idx, to_idx, weight in self.rnn_connections:
            self.rnn_neurons[to_idx] += self.neurons[from_idx] * weight

        # Decay RNN state
        self.rnn_neurons *= self.rnn_decay

        # Update neuron activations with RNN state
        self.neurons = self.rnn_neurons.copy()
        self.neurons[self.bias_index] = 1.0

    def _get_predicted_token(self, temperature=0.5):
        """Get predicted token from output neurons."""
        output_start = self.output_start
        output_end = min(output_start + self.vocab_size, self.neuron_count)

        # Get output activations
        output_activations = self.neurons[output_start:output_end]

        # Find max activation
        max_activation = np.max(output_activations)

        # If all activations are very negative, return UNK
        if max_activation <= -0.9:
            return self.unk_token

        predicted_idx = np.argmax(output_activations)
        return predicted_idx

    def generate(self, seed_text, max_length=50, temperature=0.5):
        """
        Generate text starting from seed_text.

        Restarts the network from scratch for each token, processing the entire
        sequence (seed + all generated tokens so far) as context.

        Complexity: O(n²) - process 1 + 2 + 3 + ... + n tokens total

        Args:
            seed_text: Initial text to start generation
            max_length: Maximum tokens to generate
            temperature: Softmax temperature for sampling

        Returns:
            Generated text string
        """
        # Tokenize seed
        seed_tokens = self.tokenize(seed_text)
        if not seed_tokens:
            seed_tokens = [self.sos_token]

        # Build sequence: start with seed
        sequence = list(seed_tokens)
        generated_tokens = []

        # Generate new tokens
        for step in range(max_length):
            # Reset network for this iteration
            self._reset_neurons()

            # Process entire sequence through the network
            for token in sequence:
                self._feed_token(token)
                self._run_network()
                self._update_rnn_state()

            # Signal generation phase (token index = 1, value = 1.0)
            self.neurons[1] = 1.0
            self._run_network()

            # Get prediction
            predicted = self._get_predicted_token(temperature)
            generated_tokens.append(predicted)

            # Stop if EOS token
            if predicted == self.eos_token:
                break

            # Add predicted token to sequence for next iteration
            sequence.append(predicted)

        # Convert to text, filtering out seed tokens
        result_tokens = seed_tokens + generated_tokens
        return self.detokenize(result_tokens)

    def _feed_token(self, token_id):
        """Feed a token to the input neurons."""
        # Clear input neurons
        for i in range(self.input_start, self.input_start + self.vocab_size):
            self.neurons[i] = 0.0

        # Set the corresponding input neuron
        if 0 <= token_id < self.vocab_size:
            input_idx = self.input_start + token_id
            if input_idx < self.neuron_count:
                self.neurons[input_idx] = 1.0

    def predict_next(self, text):
        """
        Predict the next token given some context text.

        Args:
            text: Context text

        Returns:
            Predicted next word
        """
        tokens = self.tokenize(text)
        if not tokens:
            tokens = [self.sos_token]

        self._reset_neurons()

        # Process all tokens
        for token in tokens:
            self._feed_token(token)
            self._run_network()
            self._update_rnn_state()

        # Prepare for next prediction
        self.neurons[1] = 1.0
        self._run_network()

        # Get prediction
        predicted_idx = self._get_predicted_token()

        if 0 <= predicted_idx < len(self.vocabulary):
            word = self.vocabulary[predicted_idx]
            if word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                return word

        return '<UNK>'


def main():
    """Command-line interface for model inference."""
    parser = argparse.ArgumentParser(
        description='Run inference with trained NEAT model'
    )
    parser.add_argument(
        'model',
        help='Path to model JSON file'
    )
    parser.add_argument(
        '--seed',
        default='i have',
        help='Seed text for generation (default: "i have")'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=20,
        help='Maximum tokens to generate (default: 20)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5,
        help='Softmax temperature (default: 0.5)'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run in prediction mode (predict next word given context)'
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = NEATModel(args.model)
    print(f"✓ Model loaded: {model.vocab_size} vocab, {model.neuron_count} neurons")

    if args.predict:
        # Prediction mode
        print(f"\nPredicting next word after: '{args.seed}'")
        next_word = model.predict_next(args.seed)
        print(f"Predicted next word: {next_word}")
    else:
        # Generation mode
        print(f"\nGenerating text from seed: '{args.seed}'")
        generated = model.generate(args.seed, max_length=args.length, temperature=args.temperature)
        print(f"Generated text:\n{generated}")


if __name__ == '__main__':
    main()
