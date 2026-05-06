#!/usr/bin/env python
"""
Persistent NER worker using en_core_sci_sm (scispaCy).
Runs under .venv310 (Python 3.10 + spaCy 3.7 + numpy 1.x).

Protocol: newline-delimited JSON on stdin/stdout.
  Request:  {"text": "..."}
  Response: {"entities": ["ent1", ...]}
Startup:   {"ready": true, "model": "en_core_sci_sm"}
"""
import sys
import json
import warnings

warnings.filterwarnings("ignore")


def main():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_sci_sm")
            model = "en_core_sci_sm"
        except OSError:
            nlp = spacy.load("en_core_web_sm")
            model = "en_core_web_sm"
    except Exception as exc:
        sys.stdout.write(json.dumps({"ready": False, "error": str(exc)}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    sys.stdout.write(json.dumps({"ready": True, "model": model}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text = req.get("text", "")
            doc = nlp(text) if text else nlp.make_doc("")
            entities = sorted({ent.text.lower() for ent in doc.ents})
            sys.stdout.write(json.dumps({"entities": entities}) + "\n")
        except Exception as exc:
            sys.stdout.write(json.dumps({"entities": [], "error": str(exc)}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
