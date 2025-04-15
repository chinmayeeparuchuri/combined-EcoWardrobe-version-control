import sys
import os
import re

# Dynamically add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

import spacy
from spacy.matcher import Matcher
from rapidfuzz import process
import pandas as pd

from utils.color_to_dye import get_dye_from_color, color_to_dye

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load manufacturer names
manufacturer_path = "/Users/chinmayee/Documents/ecowardrobe-project/data/processed/clean_manufacturers.csv"
manufacturer_df = pd.read_csv(manufacturer_path)
manufacturer_names = manufacturer_df["Manufacturer Name"].dropna().unique().tolist()

# Fabric list
fabric_list = [
    "cotton", "linen", "hemp", "silk", "polyester", "nylon", "wool", "abaca",
    "coir", "seacell", "ramie", "modal", "tencel", "viscose", "bamboo", "jute"
]

# Colors (from color_to_dye keys)
color_list = list(color_to_dye.keys())

# Dyes from clean_dyes.csv
dye_path = "/Users/chinmayee/Documents/ecowardrobe-project/data/processed/clean_dyes.csv"
dye_df = pd.read_csv(dye_path)
dye_names = dye_df["Dye Name"].dropna().str.lower().unique().tolist()

# Dyes from fabric_dye_training_data.csv for fallback matching
fabric_dye_path = "/Users/chinmayee/Documents/ecowardrobe-project/data/processed/fabric_dye_training_data.csv"
fabric_dye_df = pd.read_csv(fabric_dye_path, low_memory=False)
fabric_dye_df.drop(columns=["Natural Dyes vs Synthetic Blending Sensitivity"], inplace=True, errors="ignore")
known_dyes_in_training = fabric_dye_df["Dye Name"].dropna().str.lower().unique().tolist()

# Matcher setup
matcher = Matcher(nlp.vocab)
for fabric in fabric_list:
    matcher.add("FABRIC", [[{"LOWER": fabric.lower()}]])
for color in color_list:
    matcher.add("COLOR", [[{"LOWER": color.lower()}]])

# Fuzzy match manufacturer
from rapidfuzz import process

manufacturer_df = pd.read_csv("data/processed/clean_manufacturers.csv")
manufacturer_names = manufacturer_df["Manufacturer Name"].dropna().tolist()

def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def fuzzy_match_manufacturer(text, threshold=70):
    if not text:
        return None

    normalized_text = normalize(text)
    normalized_choices = [normalize(name) for name in manufacturer_names]
    normalized_to_original = dict(zip(normalized_choices, manufacturer_names))

    result = process.extractOne(normalized_text, normalized_choices, score_cutoff=threshold)
    if result is None:
        print(f"[WARN] No manufacturer match found for: {text}")
        return None

    match, score = result
    return normalized_to_original[match]

def extract_manufacturer(text):
    # Split text into smaller spans (up to 4 words) to improve fuzzy match accuracy
    tokens = text.split()
    chunks = [' '.join(tokens[i:i+4]) for i in range(len(tokens))]
    
    best_score = 0
    best_match = None

    for chunk in chunks:
        result = process.extractOne(chunk, manufacturer_names, score_cutoff=70)
        if result:
            match, score, _ = result
            if score > best_score:
                best_score = score
                best_match = match

    if best_match:
        return best_match
    else:
        print(f"[WARN] No manufacturer match found in: {text}")
        return None


# Fuzzy match dye
def fuzzy_match_dye(text, threshold=85):
    result = process.extractOne(text.lower(), dye_names, score_cutoff=threshold)
    if result is None:
        return None
    match, score, _ = result
    return match if score >= threshold else None

# Extract explicitly mentioned dye name
def extract_explicit_dye_name(text):
    text_lower = text.lower()
    for dye in dye_names:
        if dye in text_lower:
            return dye
    return None

# Fallback to known dye in training data
def fallback_dye_to_known_dye(dye_candidate):
    if not dye_candidate:
        return None
    result = process.extractOne(dye_candidate.lower(), known_dyes_in_training, score_cutoff=80)
    if result:
        return result[0]
    return None

# Fabric extractor with blend percentage support
def extract_fabrics(text):
    fabric_data = {}
    blend_pattern = r"([A-Za-z\s]+?)(?:\s*\([^)]*\))?\s*\((\d+)%\)"
    matches = re.findall(blend_pattern, text)

    if matches:
        total_pct = sum(int(pct) for _, pct in matches)
        for raw_fabric, pct in matches:
            cleaned_fabric = raw_fabric.strip().lower()
            for known in fabric_list:
                if known in cleaned_fabric:
                    fabric_data[known] = int(pct) / total_pct
                    break
    else:
        detected = [fabric for fabric in fabric_list if fabric in text.lower()]
        if detected:
            weight = 1 / len(detected)
            fabric_data = {fabric: weight for fabric in detected}

    return fabric_data if fabric_data else None

# Main parser
def parse_user_input(text):
    doc = nlp(text)
    matches = matcher(doc)

    color = None
    dye = None

    # Manufacturer extraction (NEW logic below 👇)
    manufacturer = extract_manufacturer(text)  # uses fuzzy matching on chunks

    # Fabric extraction (you already have this)
    fabric_data = extract_fabrics(text)

    # Extract color from matcher
    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        if label == "COLOR" and color is None:
            color = span.text.lower()

    # Step 1: Explicit dye
    dye = extract_explicit_dye_name(text)

    # Step 2: Fuzzy dye
    if not dye:
        dye = fuzzy_match_dye(text)

    # Step 3: Fallback to known dye in training
    if dye and dye.lower() not in known_dyes_in_training:
        fallback = fallback_dye_to_known_dye(dye)
        if fallback:
            dye = fallback

    # Step 4: If no dye at all, fallback from color
    if not dye and color:
        fabric_keys = list(fabric_data.keys()) if fabric_data else []
        dye = get_dye_from_color(color, fabric_keys)

    return {
        "color": color,
        "dye": dye,
        "fabric": fabric_data,
        "manufacturer": manufacturer
    }

# Example usage
if __name__ == "__main__":
    sample = "natural madder linen shirt from Indian Needle Private Limited"
    parsed = parse_user_input(sample)
    print(parsed)

    if parsed["dye"]:
        dye_name = [parsed["dye"]]
    else:
        dye_name = get_dye_from_color(parsed["color"]) if parsed["color"] else None
    print(dye_name)