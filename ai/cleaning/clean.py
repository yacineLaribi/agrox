"""
add_wikipedia_images_fast.py

Fetches main image URLs for plant genera using Wikipedia REST API.
Saves progress after each row.
"""

import pandas as pd
import requests
from tqdm import tqdm
import time

INPUT_FILE = "data.csv"
OUTPUT_FILE = "data_clean.csv"

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")
if "image_url" not in df.columns:
    df["image_url"] = ""

def get_wikipedia_image(genus):
    """Return main image URL from Wikipedia summary API"""
    if not genus or genus == "":
        return "No image found"
    
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{genus}"
    headers = {"User-Agent": "AgroxBot/1.0 (contact: your_email@example.com)"}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            img_url = data.get("originalimage", {}).get("source")
            if img_url:
                return img_url
        return "No image found"
    except Exception:
        return "Error fetching image"

# Fetch images row by row and save progress
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching images"):
    genus = row.get("Genus", "")
    df.at[idx, "image_url"] = get_wikipedia_image(genus)
    
    # Save after each row
    df.to_csv(OUTPUT_FILE, index=False)
    time.sleep(0.2)  # small delay to be nice to Wikipedia

print(f"✅ Done. Output saved to {OUTPUT_FILE}")
