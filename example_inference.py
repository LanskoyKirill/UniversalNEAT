#!/usr/bin/env python3
"""
Quick examples of using the NEAT model through Python
"""

from run_model import NEATModel


def main():
    """Run example inferences with the trained model."""

    # Load the model
    print("=" * 60)
    print("NEAT Model Inference Examples")
    print("=" * 60)

    model = NEATModel('logs/models/best_model.json')
    print(f"\n✓ Loaded model with {model.vocab_size} vocabulary and {model.neuron_count} neurons\n")

    # Example 1: Predict next word
    print("Example 1: Predict Next Word")
    print("-" * 60)
    context = "i have a"
    next_word = model.predict_next(context)
    print(f"Context: '{context}'")
    print(f"Predicted next word: {next_word}\n")

    # Example 2: Generate text with different seeds
    print("Example 2: Generate Text from Different Seeds")
    print("-" * 60)
    seeds = [
        "i have",
        "my dog",
        "we go",
        "the park",
    ]

    for seed in seeds:
        generated = model.generate(seed, max_length=15)
        print(f"Seed: '{seed}'")
        print(f"Generated: '{generated}\n'")

    # Example 3: Temperature effect (same seed, different temperature)
    print("Example 3: Temperature Effect on Generation")
    print("-" * 60)
    seed = "I go outside"
    temperatures = [0.3, 0.5, 0.8]

    for temp in temperatures:
        generated = model.generate(seed, max_length=10, temperature=temp)
        print(f"Temperature={temp}: '{generated}'")

    print("\n" + "=" * 60)
    print("For more options, run: python3 run_model.py --help")
    print("=" * 60)


if __name__ == '__main__':
    main()
