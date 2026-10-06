import torch
import torch.nn as nn


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_sequence_length: int = 256,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dim
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(
            embedding_dim,
            vocab_size
        )

        self.max_sequence_length = max_sequence_length

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum length {self.max_sequence_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            sequence_length
        )

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        x = self.transformer(x)

        logits = self.output_layer(x)

        return logits


if __name__ == "__main__":
    vocab_size = 100

    model = TransformerLanguageModel(
        vocab_size=vocab_size
    )

    input_ids = torch.randint(
        0,
        vocab_size,
        (2, 10)
    )

    output = model(input_ids)

    print("Input shape:", input_ids.shape)
    print("Output shape:", output.shape)