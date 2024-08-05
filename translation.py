from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import json

# Initialize the model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to perform translation
def translate_text(input_text, input_lang_code, output_lang_code):
    try:
        print(input_text)

        input_lang = input_lang_code
        output_lang = output_lang_code

        if input_lang is None:
            return "Invalid input language code"

        if output_lang is None:
            return "Invalid output language code"

        tokenizer.src_lang = input_lang
        encoded_text = tokenizer(input_text, return_tensors="pt").to(device)  # Move tensors to GPU
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.lang_code_to_id[output_lang]
        )
        output_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return {
            "text": input_text,
            "translation": output_text
        }

    except KeyError as e:
        # Handle the case where the language code is not found in the tokenizer
        print(f"KeyError: {e}")
        return "Error: Language code not found in tokenizer."

    except Exception as e:
        # Handle any other exceptions
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred during translation."
