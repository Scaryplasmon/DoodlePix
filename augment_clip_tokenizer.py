import argparse
import logging
from transformers import CLIPTokenizer

def generate_web_safe_hex_codes():
    """
    Generate a list of web safe hex codes.
    Web safe colors are produced by taking each channel (R, G, B) from
    the set {0, 51, 102, 153, 204, 255}. This yields 6^3 = 216 color codes.
    """
    levels = [0, 51, 102, 153, 204, 255]
    hex_codes = []
    for r in levels:
        for g in levels:
            for b in levels:
                hex_code = f"#{r:02x}{g:02x}{b:02x}"
                hex_codes.append(hex_code)
    return hex_codes

def test_tokenization(tokenizer, test_colors):
    """Test tokenization of a list of hex codes and print the results."""
    print("Tokenization results:")
    for code in test_colors:
        # Tokenizing the code and getting token ids
        tokens = tokenizer.tokenize(code)
        token_ids = tokenizer(code)['input_ids']
        print(f"{code}: tokens: {tokens} | token ids: {token_ids}")

def main():
    parser = argparse.ArgumentParser(
        description="Augment CLIP Tokenizer with additional HEX color codes as special tokens."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path or identifier for the pretrained CLIP model (used to load the tokenizer)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the augmented tokenizer will be saved."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Loading original CLIPTokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Test a few hex codes BEFORE adding special tokens.
    test_colors = ["#fb1212", "#0088ff", "#c0ffee", "#abcdef", "#123456"]
    logger.info("Tokenization BEFORE adding special HEX tokens:")
    test_tokenization(tokenizer, test_colors)

    # Generate a curated list of hex codes (216 web-safe colors in our case)
    hex_tokens = generate_web_safe_hex_codes()
    logger.info(f"Generated {len(hex_tokens)} web safe HEX tokens for addition.")

    # Add these hex tokens as additional special tokens.
    special_tokens_dict = {'additional_special_tokens': hex_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added_tokens} additional special tokens to the tokenizer.")

    # Test the tokenization AFTER adding the special tokens.
    logger.info("Tokenization AFTER adding special HEX tokens:")
    test_tokenization(tokenizer, test_colors)

    # Save the augmented tokenizer so that it can be loaded in your training script.
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Augmented tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()