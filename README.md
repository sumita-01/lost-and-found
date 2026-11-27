# Campus Lost and Found Matcher (Gemini)

## Run in VS Code
1. Unzip, then File > Open Folder > `lost_found_matcher`.
2. Open `.env` and replace `PASTE_YOUR_API_KEY_HERE` with your Gemini key (https://aistudio.google.com/apikey).
3. Open a terminal (Ctrl + `) and run:

```
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python main.py
```

Results appear in `output/`: matches.csv, no_match.csv, unclaimed.csv, summary.txt.

`python main.py --offline` runs the pipeline without Gemini (testing only).
Change the model in `.env` via GEMINI_MODEL if needed.

> Note: the files currently in `output/` were produced by the offline test mode. Running `python main.py` with your key overwrites them with Gemini results.
