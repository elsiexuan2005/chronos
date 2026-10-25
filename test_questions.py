import json
from pathlib import Path

path = Path(__file__).resolve().parent / "data" / "questions.json"

with open(path, "r", encoding="utf-8") as f:
    questions = json.load(f)

print("Total questions:", len(questions))

for question in questions[:3]:
    print(question["question"])
    print("Answer:", question["answer"])