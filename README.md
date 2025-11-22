# Recommender System

Simple Netflix-style recommender built with Flask, content+collaborative hybrids.

## Setup (Windows / VS Code)

1. Create virtualenv:
   python -m venv .venv
2. Activate:
   .venv\Scripts\Activate.ps1 (PowerShell)
   .venv\Scripts\activate.bat (CMD)
3. Install deps:
   pip install -r requirements.txt
4. Run:
   python app.py

## Notes

- Add your dataset `netflix_titles.csv` if needed (or remove it from .gitignore).
- Do not commit secrets or large datasets to public repos.
