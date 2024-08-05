from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
import torch
import re
from transformers import pipeline

model_location = 'finetuned-bart-summarization'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loaded = BartForConditionalGeneration.from_pretrained(model_location)
tokenizer_loaded = BartTokenizer.from_pretrained(model_location)
model_loaded.to(device)

max_length = 2000

def process_summarize(text, max_length=2000):
    inputs = tokenizer_loaded.encode(text, return_tensors="pt", max_length=max_length, truncation=False)
    inputs = inputs.to(device)
    summary_ids = model_loaded.generate(inputs, max_length=2000, num_beams=30, early_stopping=False)
    summary = tokenizer_loaded.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def split_paragraph(paragraph):
    # Split on period followed by space or newline
    sentences = re.split(r'(?<=[\.\n])\s+', paragraph)
    return sentences

# Function to create chunks from sentences containing dictionary words
def create_chunks(paragraph):
    sentences = split_paragraph(paragraph)
    chunks = []
    i = 0
    running_sentence = ''
    
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
        running_sentence = ''
                
        i = j      

    if(len(running_sentence) > 0):
        chunks.append(running_sentence)
        
    return chunks

def is_match(sentence, word, index):
    i = index
    j = 0
    while j < len(word) and i < len(sentence) and word[j] == sentence[i]:
        i += 1
        j += 1
    return j == len(word)

def summarize_it(sentence):
    chunks = create_chunks(sentence)

    result = ""
    for chunk in chunks:
        print(chunk)
        r = process_summarize(chunk)
        result += r + "."
        
    return result.strip(".")

def summarize_text(sentence):    
    output = summarize_it(sentence)
    return {
        "text": sentence,
        "summarization": output
    }
