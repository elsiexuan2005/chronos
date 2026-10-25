import json
import random
from pathlib import Path


class MathChallenge:
    def __init__(self, number_of_questions=5):
        if number_of_questions != 5:
            raise ValueError("Mixed mode requires exactly 5 questions")

        path = (
            Path(__file__).resolve().parent
            / "data"
            / "questions.json"
        )

        with open(path, "r", encoding="utf-8") as f:
            all_questions = json.load(f)

        self.questions = []

        # Pick one random question from each category
        for category in range(1, 6):
            pool = [
                q for q in all_questions
                if q["category"] == category
            ]

            if not pool:
                raise ValueError(
                    f"No questions in category {category}"
                )

            self.questions.append(random.choice(pool))

    def check_answer(self, index, answer):
        try:
            return (
                int(answer.strip())
                == int(self.questions[index]["answer"])
            )
        except (ValueError, TypeError, AttributeError):
            return False