import json
import random
import re
from pathlib import Path
from collections import Counter

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


MODEL = "gemini-3.5-flash-lite"
TARGET_PER_CATEGORY = 6

OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "questions.json"
)


# 1. Gemini response schema

class MathQuestion(BaseModel):
    question: str
    answer: int
    category: int = Field(ge=1, le=5)
    difficulty: int = Field(ge=1, le=5)


class QuestionBatch(BaseModel):
    questions: list[MathQuestion]


# 2. Verify mathematical answers

def solve_equation(question):
    """
    Solve basic arithmetic or linear equations
    using SymPy, restricted to simple maths notation.
    """
    from sympy import symbols, Eq, solve
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )

    x = symbols("x")

    expression = question.strip()
    expression = re.sub(
        r"^(what is|solve for x|solve)\s*:?\s*",
        "",
        expression,
        flags=re.IGNORECASE,
    )
    expression = expression.rstrip(" ?.")
    expression = expression.replace("^", "**").replace("×", "*")

    # Reject unsupported characters and notation.
    if not re.fullmatch(r"[0-9xX+\-*/()=.\s*]+", expression):
        raise ValueError("Unsupported mathematical expression")

    expression = expression.replace("X", "x")

    transformations = (
        standard_transformations
        + (implicit_multiplication_application,)
    )

    allowed = {"x": x}

    def parse(part):
        return parse_expr(
            part,
            local_dict=allowed,
            transformations=transformations,
            evaluate=True,
        )

    if "=" in expression:
        lhs, rhs = expression.split("=")
        solutions = solve(Eq(parse(lhs), parse(rhs)), x)

        if len(solutions) != 1:
            raise ValueError("Equation must have exactly one solution")

        result = solutions[0]
    else:
        result = parse(expression)

        if result.has(x):
            raise ValueError("Arithmetic question contains x")

    if result.is_integer is not True:
        raise ValueError("Answer must be an integer")

    return int(result)


def validate_question(q):
    try:
        expected_answer = solve_equation(q.question)

        if expected_answer != q.answer:
            return None

        key = re.sub(r"\s+", "", q.question.lower())

        return key, {
            "topic": f"Level {q.category}",
            "question": q.question,
            "code": "",
            "answer": str(expected_answer),
            "category": q.category,
            "difficulty": q.difficulty,
        }

    except Exception:
        return None


# 3. Gemini generation

def generate_with_gemini():
    prompt = """
    Generate exactly 30 unique mathematics questions.

    Create exactly 6 questions for each category:

    Category 1:
    Easy two-digit addition or subtraction.

    Category 2:
    Easy arithmetic involving carrying or borrowing.

    Category 3:
    Medium multiplication involving integers 11 to 19.

    Category 4:
    Medium linear equations of the form ax + b = c.

    Category 5:
    Hard linear equations with brackets or x on both sides.

    Requirements:
    - Every answer must be an integer.
    - Every equation must have exactly one solution.
    - Include question, answer, category and difficulty.
    - Set difficulty equal to category.
    - Use mathematical expressions only.
    - Do not include 'What is' or 'Solve for x'.
    - Use x as the variable.
    - Use * for explicit multiplication.
    - Use only integers, +, -, *, parentheses and =.
    - No fractions, decimals, exponents or division.
    - No duplicate questions.
    """

    client = genai.Client()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QuestionBatch,
            temperature=0.7,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    batch = QuestionBatch.model_validate_json(response.text)
    return batch.questions


# 4. Build final question set

def build_questions():
    questions = []
    seen = set()
    counts = Counter()

    print("Generating questions with Gemini...")

    generated = generate_with_gemini()

    for q in generated:
        result = validate_question(q)

        if result is None:
            print("Rejected:", q.question)
            continue

        key, question = result
        category = question["category"]

        if key in seen:
            continue

        if counts[category] >= TARGET_PER_CATEGORY:
            continue

        seen.add(key)
        questions.append(question)
        counts[category] += 1

    expected = {i: TARGET_PER_CATEGORY for i in range(1, 6)}

    if dict(counts) != expected:
        raise ValueError(
            f"Question distribution incomplete: {dict(counts)}. "
            "Regenerate the questions."
        )

    return questions


# 5. Save questions

def main():
    questions = build_questions()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    print("\nQuestion distribution:")
    print(dict(Counter(q["category"] for q in questions)))

    print(f"\nSaved {len(questions)} questions to {OUTPUT_PATH}")

    print("\nExample questions:")
    for q in questions[:5]:
        print(q["question"], "=>", q["answer"])


if __name__ == "__main__":
    main()