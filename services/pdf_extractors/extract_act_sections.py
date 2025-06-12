# services/pdf_extractors/extract_act_sections.py

import re
import pdfplumber
import pandas as pd
import os

PDF_PATH = "Data/raw_acts_pdf/Indian Penal Code Book (2).pdf"
CSV_PATH = "Data/actmetadata/ipc_sections.csv"

SECTION_PATTERN = re.compile(r"\[s\s*(\d+)\]")


def extract_sections_from_pdf(pdf_path):
    sections = []
    current_section_number = None
    current_section_title = None
    current_section_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            lines = text.split("\n")
            for line in lines:
                section_match = SECTION_PATTERN.match(line.strip())
                if section_match:
                    # Save previous section
                    if current_section_number is not None:
                        sections.append(
                            {
                                "section_number": current_section_number,
                                "section_name": current_section_title,
                                "section_text": "\n".join(current_section_text).strip(),
                            }
                        )

                    # Start new section
                    current_section_number = section_match.group(1)
                    current_section_title = None
                    current_section_text = []

                elif current_section_number is not None:
                    # First line after [s X] → section title
                    if current_section_title is None and line.strip():
                        current_section_title = line.strip()
                    else:
                        current_section_text.append(line.strip())

    # Save last section
    if current_section_number is not None:
        sections.append(
            {
                "section_number": current_section_number,
                "section_name": current_section_title,
                "section_text": "\n".join(current_section_text).strip(),
            }
        )

    return sections


if __name__ == "__main__":
    print(f"📄 Extracting sections from {PDF_PATH}...")
    sections = extract_sections_from_pdf(PDF_PATH)
    print(f"✅ Extracted {len(sections)} sections.")

    df = pd.DataFrame(sections)
    df.insert(0, "title", "Indian Penal Code")  # Consistent title column
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Saved CSV to {CSV_PATH}")
