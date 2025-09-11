import os
import re
import pandas as pd
import numpy as np
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import logging
import json

#--- configs ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

input_dir  = "all 10k pdfs from the banks must be in the same directory"
output_dir = "the script will save the embeddings in npy files and completely processed jsonl including the built-in embeddings "
EMBEDDINGS_DIR = os.path.join(output_dir, "embeddings")
OUTPUT_JSONL_PATH = os.path.join(output_dir, "10k_analysis.jsonl")

SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TARGET_SECTIONS = ['1A','2','7', '7A', '8']

def setup_directories():
    logging.info("Setting up output directories...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    logging.info(f"Output will be saved in '{output_dir}' directory.")

# --- pdf extractor & cleanser ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    extracts text from a pdf file using pdfplumber
    """
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "".join(page.extract_text() or "" for page in pdf.pages)
    return full_text

def clean_text(text: str) -> str:
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_all_sections(full_text: str) -> dict:
    section_pattern = re.compile(r"ITEM\s+(\d+[A-Z]?)\.?", re.IGNORECASE)
    matches = list(section_pattern.finditer(full_text))

    if not matches:
        logging.warning("No 'ITEM' sections found in the document.")
        return {}

    extracted_sections = {}
    for i, current_match in enumerate(matches):
        section_key = current_match.group(1).upper()
        start_pos = current_match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_text = full_text[start_pos:end_pos]
        extracted_sections[section_key] = section_text

    return extracted_sections

# --- sentiment analysis & embedder & other stuffs ---

def get_sentiment_scores(text: str, sentiment_pipeline, tokenizer) -> dict:
    id2label = sentiment_pipeline.model.config.id2label
    logging.info(f"Model label mapping: {id2label}")

    label_mapping = {}
    for idx, label_name in id2label.items():
        label_lower = label_name.lower()
        if 'positive' in label_lower or 'pos' in label_lower:
            label_mapping[label_name] = 'positive'
        elif 'negative' in label_lower or 'neg' in label_lower:
            label_mapping[label_name] = 'negative'
        else:
            label_mapping[label_name] = 'neutral'

    tokens = tokenizer.encode(text, add_special_tokens=False)

    max_chunk_length = 510

    text_chunks = []
    for i in range(0, len(tokens), max_chunk_length):
        chunk_tokens = tokens[i:i + max_chunk_length]
        text_chunks.append(tokenizer.decode(chunk_tokens))

    if not text_chunks:
        return {'sentiment_positive': 0.0, 'sentiment_negative': 0.0, 'sentiment_neutral': 0.0}

    results = sentiment_pipeline(text_chunks, padding=True, truncation=True, max_length=512)

    scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}

    for result_item in results:
        if isinstance(result_item, list):
            result = result_item[0]
        else:
            result = result_item

        label = result['label']
        score = result['score']

        if label in label_mapping:
            score_category = label_mapping[label]
            scores[score_category] += score
        else:
            logging.warning(f"Unknown sentiment label: {label}")

    num_chunks = len(text_chunks)
    final_scores = {
        'sentiment_positive': scores['positive'] / num_chunks if num_chunks > 0 else 0,
        'sentiment_negative': scores['negative'] / num_chunks if num_chunks > 0 else 0,
        'sentiment_neutral': scores['neutral'] / num_chunks if num_chunks > 0 else 0
    }

    logging.info(f"Final sentiment scores: {final_scores}")
    return final_scores

def generate_and_save_embedding(text: str, embedding_model, file_path: str):
    embedding = embedding_model.encode(text, show_progress_bar=False)
    np.save(file_path, embedding)
    return embedding

# --- main functions ---

def main():
    setup_directories()

    logging.info("Loading Hugging Face models this may take a moment...")
    device = 0 if torch.cuda.is_available() else -1

    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    sentiment_pipeline = pipeline(
        "text-classification",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        device=device
    )
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info("Models loaded successfully")

    all_results = []

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' not found please create it and add your pdf files")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logging.warning(f"no pdf files found in {input_dir}")
        return

    logging.info(f"found {len(pdf_files)} pdf files to process")

    for filename in pdf_files:
        file_match = re.match(r"10k-(\d{4})-(\w+)\.pdf", filename, re.IGNORECASE)
        if not file_match:
            logging.warning(f"skipping file with non-standard name: {filename}")
            continue

        year, bank = file_match.groups()
        bank = bank.lower()
        pdf_path = os.path.join(input_dir, filename)
        logging.info(f"--- processing: {filename} (Bank: {bank}, Year: {year}) ---")

        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            continue

        sections = extract_all_sections(full_text)

        for section_key in TARGET_SECTIONS:
            if section_key not in sections:
                logging.warning(f"section 'ITEM {section_key}' not found in {filename}")
                continue

            logging.info(f"Processing ITEM {section_key}...")
            section_text = sections[section_key]
            cleaned_text = clean_text(section_text)

            sentiment_scores = get_sentiment_scores(cleaned_text, sentiment_pipeline, sentiment_tokenizer)

            embedding_filename = f"10k-{year}-{bank}-item{section_key.lower()}.npy"
            embedding_path = os.path.join(EMBEDDINGS_DIR, embedding_filename)

            embedding_vector = generate_and_save_embedding(cleaned_text, embedding_model, embedding_path)

            sentiment_values = {
                'positive': sentiment_scores['sentiment_positive'],
                'negative': sentiment_scores['sentiment_negative'],
                'neutral': sentiment_scores['sentiment_neutral']
            }
            dominant_sentiment = max(sentiment_values, key=sentiment_values.get)

            result_row = {
                'bank': bank,
                'year': int(year),
                'section': f"item{section_key.lower()}",
                'sentiment_positive': sentiment_scores['sentiment_positive'],
                'sentiment_negative': sentiment_scores['sentiment_negative'],
                'sentiment_neutral': sentiment_scores['sentiment_neutral'],
                'dominant_sentiment': dominant_sentiment,
                'text_chunk': cleaned_text,
                'embeddings': embedding_vector.tolist() if embedding_vector is not None else None
            }
            all_results.append(result_row)

    if not all_results:
        logging.warning("No data was processed the output .jsonl file will not be created")
        return

    logging.info(f"Aggregating and saving results to {OUTPUT_JSONL_PATH}...")
    with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    logging.info("--- Script finished successfully! ---")

if __name__ == "__main__":
    main()
