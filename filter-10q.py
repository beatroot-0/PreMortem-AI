import os
import re
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# --- configs ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = r"C:\Users\gamin\OneDrive\Documents\shit\python\hackathon\datasets\master-10q"
OUTPUT_DIR = r"C:\Users\gamin\OneDrive\Documents\shit\python\hackathon\datasets\json-10q"

FINBERT_MODEL = "ProsusAI/finbert"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

MAX_SEQ_LEN = 512
CHAR_CHUNK_LEN = 1200
MDA_MIN_CHARS = 800

APOST = r"(?:'|\u2019|\x92)"
RE_PART_II = re.compile(r"\bPART\s+II\b", re.IGNORECASE)
RE_ITEM2 = re.compile(r"\bItem\s*2\b", re.IGNORECASE)
RE_MDA_CAPS = re.compile(r"\bMANAGEMENT" + APOST + r"S\s+DISCUSSION\s+AND\s+ANALYSIS\b", re.IGNORECASE)
RE_QFS = re.compile(r"\bQuarterly\s+Financial\s+Summary\b", re.IGNORECASE)
RE_LEGAL = re.compile(r"\bLegal\s+Notice\b", re.IGNORECASE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

#--- parsing the pdf ---

def parse_filename(pdf_path: str) -> Tuple[str, str, str]:
    fn = os.path.basename(pdf_path)
    m = re.match(r"[Qq](\d)[-_](\d{4})[-_](.+)\.pdf$", fn)
    if m:
        quarter = f"Q{m.group(1)}"
        year = m.group(2)
        abbr = m.group(3).lower().replace(" ", "-")
        mapping = {"boa": "boa", "bofa": "boa", "bank-of-america": "boa", "jpm": "jpm", "jp-morgan": "jpm", "gs": "gs", "goldman-sachs": "gs", "ms": "ms", "morgan-stanley": "ms"}
        company = mapping.get(abbr, re.sub(r"[^a-z0-9\-]+", "-", abbr))
        return company, quarter, year
    m = re.match(r"(.+)[-_ ]([Qq][1-4])[-_ ](\d{4})\.pdf$", fn)
    if m:
        company_raw = re.sub(r"\s+", "-", m.group(1).strip().lower())
        mapping = {"bank-of-america": "boa", "jp-morgan": "jpm", "goldman-sachs": "gs", "morgan-stanley": "ms"}
        company = mapping.get(company_raw, company_raw)
        quarter = m.group(2).upper()
        year = m.group(3)
        return company, quarter, year
    return Path(fn).stem.lower(), "Q?", "YYYY"

# --- cleaning the pdf text extracted ---

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"\f", "\n", s)
    s = re.sub(r"\n\s*Page\s+\d+(\s+of\s+\d+)?\s*\n", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", s)
    s = re.sub(r"\n\s+\n", "\n", s)
    s = re.sub(r"[\t\x0b\r]+", " ", s)
    return s.strip()

def extract_page_texts(pdf_path: str) -> List[str]:
    """
    extracts a list of text strings, one for each page, using pdfplumber
    """
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]

def extract_mda_text(pages: List[str], company: str, filename: str) -> str:
    full_text = "\n".join(pages)
    if company == "ms":
        qfs_match = RE_QFS.search(full_text)
        if not qfs_match:
            logging.warning(f"Could not find 'Quarterly Financial Summary' for MS in {filename}")
            return ""
        start_pos = qfs_match.start()
        legal_match = RE_LEGAL.search(full_text, pos=start_pos)
        end_pos = legal_match.start() if legal_match else len(full_text)
        return clean_text(full_text[start_pos:end_pos])
    else:
        item2_match = RE_ITEM2.search(full_text) or RE_MDA_CAPS.search(full_text)
        if not item2_match:
            logging.warning(f"Could not find start of MDA/Item 2 for {company} in {filename}")
            return ""
        start_pos = item2_match.end()
        end_pos = len(full_text)
        part2_match = RE_PART_II.search(full_text, pos=start_pos)
        if part2_match:
            end_pos = part2_match.start()
        item3_pattern = re.compile(r"\bItem\s*3\b", re.IGNORECASE)
        item3_match = item3_pattern.search(full_text, pos=start_pos)
        if item3_match and item3_match.start() < end_pos:
            end_pos = item3_match.start()
        return clean_text(full_text[start_pos:end_pos])

#--- sentiment analyser & embedder & other stuffs ---

def _load_finbert():
    tok = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    mdl.eval()
    id2label = getattr(mdl.config, 'id2label', {0: 'positive', 1: 'negative', 2: 'neutral'})
    logging.info(f"Model label mapping: {id2label}")
    return tok, mdl, id2label

def analyze_sentiment(text: str, tok, mdl, id2label: Dict[int, str]) -> Dict[str, float]:
    if not text:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    label_mapping = {}
    for idx, label_name in id2label.items():
        label_lower = label_name.lower()
        if 'positive' in label_lower or 'pos' in label_lower:
            label_mapping[idx] = 'positive'
        elif 'negative' in label_lower or 'neg' in label_lower:
            label_mapping[idx] = 'negative'
        else:
            label_mapping[idx] = 'neutral'

    chunks = [text[i:i+CHAR_CHUNK_LEN] for i in range(0, len(text), CHAR_CHUNK_LEN)]
    agg = np.zeros(3, dtype=np.float64)

    with torch.no_grad():
        for ch in chunks:
            inputs = tok(ch, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN, padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                mdl = mdl.to('cuda')
            logits = mdl(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

            for idx, prob in enumerate(probs):
                if idx in label_mapping:
                    category = label_mapping[idx]
                    if category == 'positive':
                        agg[0] += prob
                    elif category == 'negative':
                        agg[1] += prob
                    elif category == 'neutral':
                        agg[2] += prob

    agg /= max(1, len(chunks))
    return {"positive": float(agg[0]), "negative": float(agg[1]), "neutral": float(agg[2])}

def _build_embedder():
    emb = SentenceTransformer(EMBED_MODEL)
    logging.info(f"Embedding device: {emb._target_device}")
    return emb

def generate_embeddings(text: str, embedder) -> np.ndarray:
    if not text:
        return np.empty((0,), dtype=np.float32)
    sents = re.split(r"(?<=[\.\!?])\s+|\n{2,}", text)
    sents = [s.strip() for s in sents if s and len(s.strip()) > 2]
    if not sents:
        sents = [text]
    vecs = embedder.encode(sents, batch_size=64, convert_to_numpy=True, normalize_embeddings=False)
    return vecs

def save_outputs(company: str, quarter: str, year: str, mda_embeddings: np.ndarray) -> str:
    base = f"{company}_{quarter}_{year}"
    emb_path = os.path.join(OUTPUT_DIR, f"{base}_mda_embeddings.npy")
    if mda_embeddings.size:
        np.save(emb_path, mda_embeddings)
    else:
        np.save(emb_path, np.empty((0,), dtype=np.float32))
    return emb_path

def get_dominant_sentiment(sentiment_scores: Dict[str, float]) -> str:
    return max(sentiment_scores, key=sentiment_scores.get)

def detect_sections_from_toc(full_text: str, toc_search_len: int = 8000) -> List[Tuple[str,int,int]]:
    head = full_text[:toc_search_len]
    pat = re.compile(r"\bItem\s*(\d)\b(?:[\.:\-]|\s){0,6}[\s\S]{0,200}", re.IGNORECASE)
    matches = []
    for m in pat.finditer(head):
        try:
            num = int(m.group(1))
        except Exception:
            continue
        matches.append((num, m.start()))
    if len(matches) < 1:
        return []
    matches = sorted(matches, key=lambda x: x[1])
    sections: List[Tuple[str,int,int]] = []
    for i, (num, pos) in enumerate(matches):
        start = pos
        end = matches[i+1][1] if i+1 < len(matches) else len(full_text)
        label = f"item{num}"
        if num == 2:
            label = "mda"
        sections.append((label, start, end))
    return sections

# --- main functions ---

def main() -> None:
    logging.info("Starting SEC 10-Q processing pipeline...")
    tok, mdl, id2label = _load_finbert()
    embedder = _build_embedder()
    master_rows: List[Dict[str, object]] = []
    row_id_counter = 1
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logging.error(f"No PDF files found in '{INPUT_DIR}'.")
        return
    for fn in sorted(pdf_files):
        pdf_path = os.path.join(INPUT_DIR, fn)
        company, quarter, year = parse_filename(pdf_path)
        logging.info(f"--- Processing file: {fn} ({company}, {quarter} {year}) ---")
        pages = extract_page_texts(pdf_path)
        if not pages:
            continue
        full_text = "\n".join(pages)
        sections = detect_sections_from_toc(full_text)
        if sections:
            logging.info(f"Found TOC-based items for {fn}: {[s[0] for s in sections]}")
            for sec_label, start_pos, end_pos in sections:
                sec_text = clean_text(full_text[start_pos:end_pos])
                if not sec_text:
                    continue
                s_sec = analyze_sentiment(sec_text, tok, mdl, id2label)
                sec_embeddings = generate_embeddings(sec_text, embedder)
                save_outputs(f"{company}_{sec_label}", quarter, year, sec_embeddings)
                chunks = re.split(r"(?<=[\.\!?])\s+|\n{2,}", sec_text)
                chunks = [s.strip() for s in chunks if s and len(s.strip()) > 2]
                if not chunks:
                    chunks = [sec_text]
                for idx, chunk in enumerate(chunks):
                    emb_vec = sec_embeddings[idx].tolist() if sec_embeddings.size else []
                    master_rows.append({
                        'row_id': row_id_counter,
                        'bank': company,
                        'year': int(year) if year.isdigit() else year,
                        'quarter': quarter,
                        'filing_type': '10-Q',
                        'section': sec_label,
                        'chunk_index': idx,
                        'dominant_sentiment': get_dominant_sentiment(s_sec),
                        'sentiment_positive': s_sec['positive'],
                        'sentiment_negative': s_sec['negative'],
                        'sentiment_neutral': s_sec['neutral'],
                        'text_chunk': chunk,
                        'embeddings': emb_vec
                    })
                    row_id_counter += 1
        else:
            mda_text = extract_mda_text(pages, company, fn)
            if not mda_text or len(mda_text) < MDA_MIN_CHARS:
                logging.warning(f"MDA section not found or is too short in {fn}. Falling back to full document text.")
                mda_text = full_text
                section_label = 'full_text_fallback'
            else:
                section_label = 'mda'
            s_mda = analyze_sentiment(mda_text, tok, mdl, id2label)
            mda_embeddings = generate_embeddings(mda_text, embedder)
            save_outputs(company, quarter, year, mda_embeddings)
            chunks = re.split(r"(?<=[\.\!?])\s+|\n{2,}", mda_text)
            chunks = [s.strip() for s in chunks if s and len(s.strip()) > 2]
            if not chunks:
                chunks = [mda_text]
            for idx, chunk in enumerate(chunks):
                emb_vec = mda_embeddings[idx].tolist() if mda_embeddings.size else []
                master_rows.append({
                    'row_id': row_id_counter,
                    'bank': company,
                    'year': int(year) if year.isdigit() else year,
                    'quarter': quarter,
                    'filing_type': '10-Q',
                    'section': section_label,
                    'chunk_index': idx,
                    'dominant_sentiment': get_dominant_sentiment(s_mda),
                    'sentiment_positive': s_mda['positive'],
                    'sentiment_negative': s_mda['negative'],
                    'sentiment_neutral': s_mda['neutral'],
                    'text_chunk': chunk,
                    'embeddings': emb_vec
                })
                row_id_counter += 1
    if master_rows:
        master_json_path = os.path.join(OUTPUT_DIR, 'master_10q_analysis.jsonl')
        logging.info(f"Pipeline complete. Master JSON saved to {master_json_path}")
        with open(master_json_path, 'w', encoding='utf-8') as f:
            json.dump(master_rows, f, indent=4)
    else:
        logging.warning("No valid sections were extracted. The output JSON file will not be created.")

if __name__ == '__main__':
    main()