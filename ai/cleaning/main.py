"""
generate_plant_explanations.py
"""

import os
import time
import argparse
import json
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from google import genai

DEFAULT_MODEL = "gemini-2.5-flash-lite"
API_KEY_ENV = "GEMINI_API_KEY"
RATE_LIMIT_SLEEP = 0.5
MAX_RETRIES = 3

load_dotenv()


def clean(v):
    if v in ["", ".", "nan", None]:
        return None   # important so Gemini knows it's missing
    return str(v)


# ---------------------------
# FIXED PROMPT → STRICT JSON
# ---------------------------
def build_prompt_plant(row: dict):

    prompt = f"""
You will output ONLY VALID JSON. No text outside JSON. No markdown. No explanations.

For any field:
- If value is missing, generate a realistic guess using botanical knowledge.
- For perc_per, perc_wood, perc_ag, ALWAYS use numeric percentages (e.g., "92%") in the text like "About 92% of species are perennial".
- Never return numeric values for other fields, only descriptive text.

Return a JSON object with EXACTLY these keys:

"perc_per_text",
"perc_wood_text",
"perc_ag_text",
"floral_symm_text",
"mating_system_text",
"repro_syndrome_text",
"pollination_syndrome_text",
"redlist_text",
"tavg_text",
"cvalue_text",
"cv_cvalue_text"

Interpretation rules:
- perc_per, perc_wood, perc_ag: convert percentage → "About X% of species …"
- floral_symm: convert to "radial" or "bilateral" symmetry.
- mating_system: convert to human-readable (selfing, mixed, self-incompatible…)
- repro_syndrome: human-readable phrase.
- pollination_syndrome: convert to "wind-pollinated", "insect-pollinated", "bird-pollinated", etc.
- RedList score:
      <1  → low extinction risk
      1–3 → moderate risk
      >3  → high risk
- tavg:
      <0.3 → cold climates
      0.3–0.7 → temperate climates
      >0.7 → warm climates
- C_value:
      <2 → small genome
      2–8 → medium genome
      >8 → large genome
- CV_C_value:
      <5 → low variation
      5–15 → moderate variation
      >15 → high variation

Raw values (may be missing):
{{
 "perc_per": {clean(row.get("perc_per"))},
 "perc_wood": {clean(row.get("perc_wood"))},
 "perc_ag": {clean(row.get("perc_ag"))},
 "floral_symm": {clean(row.get("floral_symm"))},
 "mating_system": {clean(row.get("mating_system"))},
 "repro_syndrome": {clean(row.get("repro_syndrome"))},
 "pollination_syndrome": {clean(row.get("pollination_syndrome"))},
 "RedList": {clean(row.get("RedList"))},
 "tavg": {clean(row.get("tavg"))},
 "C_value": {clean(row.get("C_value"))},
 "CV_C_value": {clean(row.get("CV_C_value"))}
}}

Return ONLY the JSON. No other text.
"""
    return prompt


# -----------------------------------------------
def call_gemini(prompt: str, model: str = DEFAULT_MODEL):
    client = genai.Client()
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            txt = getattr(resp, "text", None) or str(resp)
            return txt.strip()
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini request failed: {last_exc}")


# -----------------------------------------------
def process_csv(input_csv: str, output_csv: str, model: str):
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    # required output columns
    new_cols = [
        "perc_per_text","perc_wood_text","perc_ag_text",
        "floral_symm_text","mating_system_text","repro_syndrome_text",
        "pollination_syndrome_text","redlist_text","tavg_text",
        "cvalue_text","cv_cvalue_text"
    ]

    for c in new_cols:
        if c not in df.columns:
            df[c] = ""

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing plants"):

        try:
            prompt = build_prompt_plant(row)
            raw = call_gemini(prompt, model=model)

            # Clean any accidental markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(raw)

            for key in new_cols:
                df.at[idx, key] = parsed.get(key, "data unavailable")

        except Exception as e:
            # fallback: no crash, but write error
            for key in new_cols:
                df.at[idx, key] = f"Error: {str(e)}"

        df.to_csv(output_csv, index=False)
        time.sleep(RATE_LIMIT_SLEEP)

    print("✔️ Completed:", output_csv)


# -----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate English plant trait fields")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if os.getenv(API_KEY_ENV) is None:
        print(f"⚠️ Warning: {API_KEY_ENV} not set")

    process_csv(args.input_csv, args.output_csv, args.model)
