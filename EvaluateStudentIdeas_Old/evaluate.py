'''
Evaluate dummy problem statements
- college students
- 6-week hackathon, 3 hours per week per person
- 5 member team
- Rate on scale of 1-5 with 1 lowest, 5 highest
'''

from google import genai
from dotenv import load_dotenv
from os import getenv

load_dotenv()

client = genai.Client(api_key=getenv("GOOGLE_API_KEY"))
csv_file_path = "./dummySolutionGrades.csv"

# Upload the CSV file
uploaded_file = client.files.upload(file=csv_file_path)
print(uploaded_file.uri)

# Generate content using the uploaded CSV
response = client.models.generate_content(
    model='gemini-2.5-flash-preview-09-2025',
    contents=[
        uploaded_file,
        '''
        You are an idea-evaluator, tasked with rating the desirability, feasibility and viability of ideas.
        In every row, read the column 'Problem Statement' and evaluate the contents of 'Solution Hypothesis' column based on knowledge that
        - Team members : 5 college students
        - Duration : 6-week hackathon, 3 hours per week per person
        The rubrics are: "Timely", "Importance", "Profitable", "Solvable", "Contextual"
        Rate every solution on every rubric on a scale of 1-5, 1 being lowest, 5 highest.
        Return your answer STRICTLY as a valid CSV with the same rows and columns:
        Name, Problem Statement, Solution Hypothesis, Timely, Importance, Profitable, Solvable, Contextual
        Create a new column estimating the input tokens, output tokens and total tokens used as well.
        '''
    ]
)
csv_output = response.candidates[0].content.parts[0].text

output_path = "./dummySolutionGradesRated.csv"
with open(output_path, "w") as f:
    f.write(csv_output)

