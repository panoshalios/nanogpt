# Trains the GPT2 tokenizer on the input text and serializes the tokenizer to a file
from tokenizer import GPT2Tokenizer


def main():
    tokenizer = GPT2Tokenizer()
    max_vocab_size = 400
    input_path = "input/shakespeare/input.txt"
    output_path = "output/shakespeare_tokenizer.json"

    # Read from input/shakespeare/input.txt
    with open(input_path, "r") as f:
        text = f.read()
        tokenizer.train(text, max_vocab_size)
        print(f"Tokenizer trained. Saving to {output_path}")
        tokenizer.save(output_path)
        print("Tokenizer saved")


if __name__ == "__main__":
    main()
