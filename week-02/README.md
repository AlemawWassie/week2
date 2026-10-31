# Week 02 — Phone Repair FAQ Semantic Search

## What it does
This project loads phone repair FAQs from CSV, cleans the data, makes one text chunk from each question and answer, stores embeddings in Chroma, and returns up to three similar FAQs.

The answers are examples for learning, not a replacement for a technician's diagnosis.

## Files
- `task.py` — Python script
- `task.ipynb` — notebook with explanations and search examples
- `data/faqs.csv` — sample FAQ data
- `requirements.txt` — required packages

## Run on Windows
Open a terminal in this folder and run:
```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python task.py
```

The first run may download the embedding model, so internet access is needed. The local Chroma database is stored in `chroma_db/`.

## Try these questions
- Why does my battery lose power so fast?
- My phone is dead and will not start.
- I cannot hear sound from my phone.
- Why is my SIM not recognized?

A close match is not guaranteed to be correct. Read the result rather than treating its distance as a confidence percentage.

## Before submitting
Run the script and all notebook cells yourself. Update the notebook observation with what you actually saw, and make sure you can explain the main functions.
