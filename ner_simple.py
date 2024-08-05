from transformers import BertTokenizer, BertForTokenClassification
import pickle
import torch
import re

model_location = 'finetuned-bert-ner/simpleentities'

# Load label mappings
with open(f'{model_location}/label_mapping.pkl', 'rb') as f:
    loaded_label2id, loaded_id2label = pickle.load(f)

# Load model and tokenizer
model_loaded = BertForTokenClassification.from_pretrained(
    f'{model_location}',
    num_labels=len(loaded_id2label),
    id2label=loaded_id2label,
    label2id=loaded_label2id
)

tokenizer_loaded = BertTokenizer.from_pretrained(f'{model_location}')

# Maximum length of words per chunk
max_length = 350

# Function to split the paragraph into sentences
def split_paragraph(paragraph):
    # Split on period followed by space or newline
    sentences = re.split(r'(?<=[\.\n])\s+', paragraph)
    return sentences

# Function to create chunks from sentences containing dictionary words
def create_chunks(paragraph):
    sentences = split_paragraph(paragraph)
    chunks = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        merged_sentence = sentence
        word_count = len(sentence.split())
        j = i + 1

        while j < len(sentences) and word_count < max_length:
            next_sentence = sentences[j].strip()
            if word_count + len(next_sentence.split()) <= max_length:
                merged_sentence += ' ' + next_sentence
                word_count += len(next_sentence.split())
            else:
                break
            j += 1
        chunks.append(merged_sentence)            
                
        i = j      

    return chunks

def ner_simple(sentence):
    chunks = create_chunks(sentence)

    result = []
    for chunk in chunks:
        print(chunk)
        r = process(chunk)
        print('------------and now chunk details ---------')
        for q in r:
           print(q)
           result.append(q)        
        print('processed')
    return result

def process(sentence):    
    # Tokenize the sentence
    inputs = tokenizer_loaded(sentence, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)

    # Forward pass
    outputs = model_loaded(ids, attention_mask=mask)
    logits = outputs[0]

    # Get the active logits
    active_logits = logits.view(-1, model_loaded.num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size*seq_len,) - predictions at the token level

    # Convert IDs to tokens
    tokens = tokenizer_loaded.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [loaded_id2label[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

    # Post-processing logic
    last_b_entity = None
    last_i_token_index = -21  # Initialize to -11 to avoid negative index issues in the first 10 tokens
    for i in range(len(wp_preds)):
        token, label = wp_preds[i]
        if label.startswith('B-'):
            last_b_entity = label
            last_i_token_index = i  # Update last I-token index since B-token marks the start
        elif label.startswith('I-'):
            if last_b_entity is None or (last_i_token_index < i - 20):  # No B-token encountered or I-token is too far from the last one
                wp_preds[i] = (token, 'O')
            else:
                last_i_token_index = i  # Update last I-token index
        elif label == 'O':
            continue
        else:
            last_b_entity = None

    # Prepare word-level predictions
    word_level_predictions = []
    for pair in wp_preds:
        if (pair[0].startswith("##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
            # skip prediction
            continue
        else:
            word_level_predictions.append(pair)

    # Join tokens to form the original sentence representation
    str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")

    return word_level_predictions
