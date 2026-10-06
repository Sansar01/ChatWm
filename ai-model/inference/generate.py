import os
import sys

import torch


# Allow importing from the ai-model directory
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model.transformer import TransformerLanguageModel


# The same training text used to create the vocabulary
text = """
hello world
hello ai
hello developer
ai can write code
ai can learn code
developer writes code
code is useful
"""


# Build the same character vocabulary
characters = sorted(set(text))

vocab = {
    character: index
    for index, character in enumerate(characters)
}

inverse_vocab = {
    index: character
    for character, index in vocab.items()
}


# Create the model
vocab_size = len(vocab)

model = TransformerLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    num_layers=4,
    max_sequence_length=64,
)


# Load trained weights
model.load_state_dict(
    torch.load(
        "model_weights.pth",
        map_location="cpu"
    )
)

model.eval()


def encode(value):
    return [
        vocab[character]
        for character in value
    ]


def decode(token_ids):
    return "".join(
        inverse_vocab[token_id]
        for token_id in token_ids
    )


def generate(prompt, max_new_tokens=50):

    input_ids = torch.tensor(
        [encode(prompt)],
        dtype=torch.long
    )

    for _ in range(max_new_tokens):

        # Keep only the latest tokens
        input_ids = input_ids[
            :, -64:
        ]

        with torch.no_grad():
            logits = model(input_ids)

        # Get predictions for the last token
        next_token_logits = logits[:, -1, :]

        # Select the most likely token
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

    return decode(
        input_ids[0].tolist()
    )


prompt = "hello"

result = generate(
    prompt,
    max_new_tokens=50
)

print("\nGenerated text:")
print(result)