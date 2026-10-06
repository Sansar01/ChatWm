import re


class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    def build_vocab(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text)

        unique_tokens = sorted(set(tokens))

        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
        ]

        all_tokens = special_tokens + unique_tokens

        self.vocab = {
            token: index
            for index, token in enumerate(all_tokens)
        }

        self.inverse_vocab = {
            index: token
            for token, index in self.vocab.items()
        }

    def encode(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text)

        unk_id = self.vocab["<UNK>"]

        return [
            self.vocab.get(token, unk_id)
            for token in tokens
        ]

    def decode(self, token_ids):
        return " ".join(
            self.inverse_vocab.get(token_id, "<UNK>")
            for token_id in token_ids
        )


if __name__ == "__main__":
    text = "function add(a, b) { return a + b; }"

    tokenizer = Tokenizer()
    tokenizer.build_vocab(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("Vocabulary:")
    print(tokenizer.vocab)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(decoded)