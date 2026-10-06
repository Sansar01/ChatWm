import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim


sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.transformer import TransformerLanguageModel


# Small training dataset
text = """
hello world
hello ai
hello developer
ai can write code
ai can learn code
developer writes code
code is useful
"""


# Build a simple character vocabulary
characters = sorted(set(text))

vocab = {
    character: index
    for index, character in enumerate(characters)
}

inverse_vocab = {
    index: character
    for character, index in vocab.items()
}


def encode(text):
    return [
        vocab[character]
        for character in text
    ]


data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# Model configuration
vocab_size = len(vocab)

model = TransformerLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
    max_sequence_length=64,
)


# Training configuration
sequence_length = 32
learning_rate = 0.001
epochs = 100

loss_function = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


print("Starting training...")
print("Vocabulary size:", vocab_size)
print("Training characters:", len(data))


for epoch in range(epochs):

    total_loss = 0.0

    for start in range(0, len(data) - sequence_length - 1):

        input_sequence = data[
            start:start + sequence_length
        ]

        target_sequence = data[
            start + 1:start + sequence_length + 1
        ]

        input_sequence = input_sequence.unsqueeze(0)
        target_sequence = target_sequence.unsqueeze(0)

        optimizer.zero_grad()

        output = model(input_sequence)

        loss = loss_function(
            output.reshape(-1, vocab_size),
            target_sequence.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {total_loss:.4f}"
        )


print("Training completed.")

torch.save(
    model.state_dict(),
    "model_weights.pth"
)

print("Model saved to model_weights.pth")