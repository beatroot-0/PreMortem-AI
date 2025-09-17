!pip install PyGithub pdfplumber transformers torch sentence-transformers requests tqdm --quiet

import re
import json
import base64
from io import BytesIO
import github
import pdfplumber
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import requests
import tempfile
import os
from tqdm import tqdm 

#--- configs ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

GITHUB_TOKEN = "pat token insert here"
REPO_NAME = "repo name and folder insert here"

SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
TARGET_SECTIONS = ['1A', '2', '7', '7A', '8']

# --- pdf extractor & cleanser ---

def extract_text_from_pdf(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "".join(page.extract_text() or "" for page in pdf.pages)

def clean_text(text: str) -> str:
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_all_sections(full_text: str) -> dict:
    section_pattern = re.compile(r"ITEM\s+(\d+[A-Z]?)\.?", re.IGNORECASE)
    matches = list(section_pattern.finditer(full_text))
    if not matches:
        print("No 'ITEM' sections found in the document.")
        return {}
    extracted_sections = {}
    for i, current_match in enumerate(matches):
        section_key = current_match.group(1).upper()
        start_pos = current_match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_text = full_text[start_pos:end_pos]
        extracted_sections[section_key] = section_text
    return extracted_sections

# --- sentiment analysis & other stuffs ---

def get_sentiment_scores(text: str, sentiment_pipeline, tokenizer) -> dict:
    id2label = sentiment_pipeline.model.config.id2label
    print(f"--- Model label mapping: {id2label} ---")

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
        result = result_item[0] if isinstance(result_item, list) else result_item
        label = result['label']
        score = result['score']
        if label in label_mapping:
            score_category = label_mapping[label]
            scores[score_category] += score
        else:
            print(f"--- Unknown sentiment label: {label} ---")

    num_chunks = len(text_chunks)
    final_scores = {
        'sentiment_positive': scores['positive'] / num_chunks if num_chunks > 0 else 0,
        'sentiment_negative': scores['negative'] / num_chunks if num_chunks > 0 else 0,
        'sentiment_neutral': scores['neutral'] / num_chunks if num_chunks > 0 else 0
    }
    logging.info(f"Final sentiment scores: {final_scores}")
    return final_scores

# --- main functions ---

def main():
    print("--- Starting SEC 10-K processing pipeline ---")

    auth = github.Auth.Token(GITHUB_TOKEN)
    g = github.Github(auth=auth)
    repo = g.get_repo(REPO_NAME)

    print("--- Loading ProsusAI/finbert model this may take a moment ---")
    device = 0 if torch.cuda.is_available() else -1
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    sentiment_pipeline = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer, device=device)
    print("Models loaded successfully")

    all_results = []

    contents = repo.get_contents("10k")
    pdf_files = [content for content in contents if content.path.lower().endswith('.pdf')]
    print(f"--- Found {len(pdf_files)} PDF files in '10k' folder ---")

    progress_bar = tqdm(pdf_files, desc="Initializing...")
    for file_content_obj in progress_bar:
        filename = file_content_obj.name
        progress_bar.set_description(f"Processing {filename}")

        file_match = re.match(r"10k-(\d{4})-(\w+)\.pdf", filename, re.IGNORECASE)
        if not file_match:
            logging.warning(f"Skipping file with non-standard name: {filename}")
            continue

        year, bank = file_match.groups()
        bank = bank.lower()

        temp_pdf_path = None
        try:
            if file_content_obj.encoding == 'base64':
                file_bytes = file_content_obj.decoded_content
            else:
                download_url = file_content_obj.download_url
                headers = {'Authorization': f'token {GITHUB_TOKEN}'}
                response = requests.get(download_url, headers=headers)
                response.raise_for_status()
                file_bytes = response.content

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                temp_pdf_path = temp_f.name
                temp_f.write(file_bytes)

            full_text = extract_text_from_pdf(temp_pdf_path)
            if not full_text.strip():
                print(f"No text extracted from {filename}. Skipping.")
                continue

            sections = extract_all_sections(full_text)
            for section_key in TARGET_SECTIONS:
                if section_key not in sections:
                    print(f"Section 'ITEM {section_key}' not found in {filename}")
                    continue

                section_text = sections[section_key]
                cleaned_text = clean_text(section_text)

                sentiment_scores = get_sentiment_scores(cleaned_text, sentiment_pipeline, sentiment_tokenizer)

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
                }
                all_results.append(result_row)
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)


    if not all_results:
        print("--- No data was processed. Output file will not be created ---")
        return

    output_content = "\n".join(json.dumps(record) for record in all_results)
    output_path = "jsonl/10k_analysis.jsonl"

    try:
        file = repo.get_contents(output_path)
        repo.update_file(output_path, f"Update 10-K analysis", output_content, file.sha)
        print(f"Updated file in repo: {output_path}")
    except Exception:
        repo.create_file(output_path, f"Create 10-K analysis", output_content)
        print(f"Created new file in repo: {output_path}")

    print("--- 10-K script finished successfully! ---")

if __name__ == '__main__':
    main()

