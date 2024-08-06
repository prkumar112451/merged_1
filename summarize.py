from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

class SummarizationModelManager:
    def __init__(self, model_paths):
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models(model_paths)
    
    def load_models(self, model_paths):
        for name, (model_path, tokenizer_path) in model_paths.items():
            if 'flan' in model_path or 't5' in model_path:
                # Load FLAN-T5 or similar models
                self.models[name] = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(self.device)
                self.tokenizers[name] = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Load BART or other models
                self.models[name] = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
                self.tokenizers[name] = BartTokenizer.from_pretrained(model_path)
    
    def summarize(self, model_name, text, max_length=2000):
        model = self.models.get(model_name)
        tokenizer = self.tokenizers.get(model_name)
        
        if model is None or tokenizer is None:
            raise ValueError(f"Model {model_name} not found.")
        
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=False).to(self.device)
        summary_ids = model.generate(inputs, max_length=2000, num_beams=30, early_stopping=False)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# Example usage
model_paths = {
    "bart": ("finetuned-bart-summarization", "finetuned-bart-summarization"),  # BART model
    "flant5": ("truocpham/flan-dialogue-summary-checkpoint", "google/flan-t5-base")  # FLAN-T5 model
}

model_manager = SummarizationModelManager(model_paths)

def split_paragraph(paragraph):
    sentences = re.split(r'(?<=[\.\n])\s+', paragraph)
    return sentences

def create_chunks(paragraph, max_length=2000):
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

def summarize(text, model_name):
    chunks = create_chunks(text)
    result = ""
    
    for chunk in chunks:
        summary = model_manager.summarize(model_name, chunk)
        result += summary + "."
        
    return result.strip(".")


def summarize_text(sentence, model_name):    
    print('processing details')
    print('starting for model')
    print(model_name)
    output = summarize(sentence, model_name)
    return {
        "text": sentence,
        "summarization": output
    }
