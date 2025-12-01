import os, re, json

os.makedirs("clinical_case_simulator/i18n", exist_ok=True)

TEXTS = {}

def extract_text(content):
    matches = re.findall(r">(.*?)<", content)
    for m in matches:
        text = m.strip()
        if text and not re.match("^[0-9]+$", text):
            TEXTS[text] = ""

for base in ["main", "predict_disease", "predictproc"]:
    for root, dirs, files in os.walk(base + "/templates"):
        for file in files:
            if file.endswith(".html"):
                with open(os.path.join(root, file), "r", encoding="utf8") as f:
                    extract_text(f.read())

with open("clinical_case_simulator/i18n/vi.json", "w", encoding="utf8") as f:
    json.dump(TEXTS, f, indent=4, ensure_ascii=False)
