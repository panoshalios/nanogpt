# "Performs byte pair encoding"

import json
from pathlib import Path


class GPT2Tokenizer:
    def __init__(self):
        self.pairs_to_replace: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.max_vocab_size: int | None = None

    # Loops through the pairs_to_replace dictionary and merges the pairs
    def tokenize(self, text: str) -> list[int]:
        byte_sequence = text.encode("utf-8")
        int_sequence = list(map(int, byte_sequence))

        # need to replace in order of the pairs_to_replace
        for pair, replace_with in self.pairs_to_replace.items():
            int_sequence = self._merge_pair(int_sequence, pair, replace_with)
        return int_sequence

    # Finds the pair with the lowest value in the pairs_to_replace dictionary and merges it.
    # Repeats until we can't merge any more pairs.
    def tokenize_2(self, text: str) -> list[int]:
        byte_sequence = text.encode("utf-8")
        tokens = list(map(int, byte_sequence))

        while len(tokens) >= 2:
            frequency_table = self._get_frequency_table(tokens)

            # Select the earliest learned merge that appears in the tokens.
            pair_to_merge = min(
                frequency_table,
                key=lambda pair: self.pairs_to_replace.get(pair, float("inf")),
            )
            if pair_to_merge not in self.pairs_to_replace:
                break

            tokens = self._merge_pair(tokens, pair_to_merge, self.pairs_to_replace[pair_to_merge])

        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        # need to convert the tokens to bytes sequences
        byte_sequence = b"".join(map(lambda token: self.vocab[token], tokens))
        text = byte_sequence.decode("utf-8", errors="replace")
        return text

    def save(self, path: str | Path) -> None:
        data = {
            "max_vocab_size": self.max_vocab_size,
            "merges": [
                [token1, token2, replacement]
                for (token1, token2), replacement in self.pairs_to_replace.items()
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "GPT2Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        tokenizer = cls()
        tokenizer.max_vocab_size = data["max_vocab_size"]
        tokenizer.pairs_to_replace = {
            (token1, token2): replacement for token1, token2, replacement in data["merges"]
        }
        tokenizer.vocab = tokenizer._build_vocab()
        return tokenizer

    # "Learn the vocabulary and the byte pair encoding from the texts"
    def train(self, text: str, max_vocab_size: int) -> dict[tuple[int, int], int]:
        # Build a frequency table of adjacent token pairs
        # Find the most frequent pair
        # Merge that pair into a new token id
        # Record the merge (add the token to the vocab)
        # Repeat until the desired vocabulary size is reached

        assert max_vocab_size > 256, "Max vocab size must be greater than 256"

        # We convert the list into the byte sequence
        byte_sequence = text.encode("utf-8")
        # convert to int
        int_sequence = list(map(int, byte_sequence))

        # Loop through the byte sequence and merge the most frequent pair of characters
        num_merges = 0
        total_merges = max_vocab_size - 256
        byte_to_replace = num_merges + 256  # 256 is the first byte that is not a valid character
        pair_to_replace: dict[tuple[int, int], int] = {}
        while num_merges < total_merges:
            frequency_table = self._get_frequency_table(int_sequence)
            if len(frequency_table) == 0:
                print("[GPT2Tokenizer.train] No more pairs to merge")
                break
            pair_to_merge = max(frequency_table, key=frequency_table.get)
            int_sequence = self._merge_pair(int_sequence, pair_to_merge, byte_to_replace)

            print(f"[GPT2Tokenizer.train] Merged pair {pair_to_merge} with {byte_to_replace}")

            pair_to_replace[pair_to_merge] = byte_to_replace
            num_merges += 1
            byte_to_replace += 1

        print("Pair to replacement: ", pair_to_replace)
        print("Int sequence: ", int_sequence)
        self.pairs_to_replace = pair_to_replace
        self.vocab = self._build_vocab()
        self.max_vocab_size = max_vocab_size
        return pair_to_replace

    def _build_vocab(self) -> dict[int, bytes]:
        # token to byte mapping
        vocab = {token: bytes([token]) for token in range(256)}
        for (token1, token2), replace_with in self.pairs_to_replace.items():
            # Token to the pair of bytes. Concatenate the bytes.
            vocab[replace_with] = vocab[token1] + vocab[token2]

        return vocab

    def _merge_pair(
        self, int_sequence: list[int], pair_to_merge: tuple[int, int], replace_with: int
    ) -> list[int]:
        new_sequence = []
        i = 0
        while i < len(int_sequence):
            if (
                i < len(int_sequence) - 1
                and int_sequence[i] == pair_to_merge[0]
                and int_sequence[i + 1] == pair_to_merge[1]
            ):
                new_sequence.append(replace_with)
                i += 2
            else:
                new_sequence.append(int_sequence[i])
                i += 1
        return new_sequence

    def _get_frequency_table(self, int_sequence: list[int]) -> dict[tuple[int, int], int]:
        pair_count: dict[tuple[int, int], int] = {}

        for pair in zip(int_sequence, int_sequence[1:]):
            pair_count[pair] = pair_count.get(pair, 0) + 1

        return pair_count


test_text = "Hello, world! My nane is Panos"

tokenizer = GPT2Tokenizer()
tokenizer.train(test_text, 266)
tokenized_text = tokenizer.tokenize(test_text)
detokenized_text = tokenizer.detokenize(tokenized_text)
print(f"[GPT2Tokenizer.test] Tokenized text: {tokenized_text}")
print(f"[GPT2Tokenizer.test] Detokenized text: {detokenized_text}")
print(f"Tokenized text matches original text: {detokenized_text == test_text}")
