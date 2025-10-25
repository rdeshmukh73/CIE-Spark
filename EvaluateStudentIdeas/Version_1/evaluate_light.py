from google import genai
from dotenv import load_dotenv
from os import getenv
import pandas as pd
import ast
import re

load_dotenv()
client = genai.Client(api_key=getenv("GOOGLE_API_KEY"))

csv_file_path = "./dummySolutionGrades.csv"
df = pd.read_csv(csv_file_path)

problems = df['Problem Statement'].tolist()
solutions = df['Solution Hypothesis'].tolist()

problems_str = str(problems)
solutions_str = str(solutions)

prompt_text = f"""
You are an idea evaluator for a 6-week hackathon, 5 college students, 3 hours/week each.

Here are the problem statements:
{problems_str}

Here are the corresponding solution hypotheses:
{solutions_str}

Evaluate each solution based on these rubrics:
Timely, Importance, Profitable, Solvable, Contextual

STRICTLY return **only a Python dictionary** with two keys:
1. 'ratings': a list of dictionaries like {{"Timely": int, "Importance": int, "Profitable": int, "Solvable": int, "Contextual": int}}.
2. 'tokens': a dictionary like {{"input": int, "output": int, "total": int}}.

DO NOT include any explanations, text, or commentary. Output must be **valid Python code**.
"""

response = client.models.generate_content(
    model='gemini-2.5-flash-preview-09-2025',
    contents=[prompt_text]
)

output_text = response.candidates[0].content.parts[0].text.strip()
output_text = re.sub(r"^```(?:json)?\s*", "", output_text)
output_text = re.sub(r"\s*```$", "", output_text)

result = ast.literal_eval(output_text)

ratings_list = result['ratings']
for rubric in ["Timely", "Importance", "Profitable", "Solvable", "Contextual"]:
    df[rubric] = [r[rubric] for r in ratings_list]

output_path = "./dummySolutionGradesRated.csv"
df.to_csv(output_path, index=False)

print(f"Ratings written to {output_path}")

tokens = result['tokens']
print("Token usage (from Gemini):")
print(f"Input tokens: {tokens['input']}")
print(f"Output tokens: {tokens['output']}")
print(f"Total tokens: {tokens['total']}")
