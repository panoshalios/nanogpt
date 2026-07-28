# Loads the Shakespeare dateset and prepares it for training
import os

import numpy as np

from tokenizer.gpt2_tokenizer import GPT2Tokenizer


def main():
    tokenizer = GPT2Tokenizer.load("output/shakespeare_tokenizer.json")
    print("Tokenizer loaded")

    train_split = 0.9

    with open("input/shakespeare/input.txt", "r") as f:
        text = f.read()
        print("Text loaded")
        num_tokens = len(text)

        # Train, val split
        train_data = text[: int(num_tokens * train_split)]
        val_data = text[int(num_tokens * train_split) :]

        # Tokenize the data
        train_tokens = tokenizer.tokenize(train_data)
        val_tokens = tokenizer.tokenize(val_data)

        print(f"Train tokens: {len(train_tokens)}")
        print(f"Val tokens: {len(val_tokens)}")

        # load to numpy arrays
        train_tokens = np.array(train_tokens, dtype=np.uint16)
        val_tokens = np.array(val_tokens, dtype=np.uint16)

        print(f"Train tokens shape: {train_tokens.shape}")
        print(f"Val tokens shape: {val_tokens.shape}")

        # save to numpy arrays
        train_tokens.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
        val_tokens.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

        print("Data saved")


if __name__ == "__main__":
    main()
