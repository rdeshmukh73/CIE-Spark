### PESU CIE - Spark - Idea valiation for Desirability, Viability, Feasibility
### Raghavendra Deshmukh, 1-Oct-2025

#The part 3 of the Problem Statement Evaluation for PESU CIE Spark
#This code reads the Problem Statements, Sub Categories, Creates an LLM Prompt and gets a polished Problem Statement
#Sends all same/similar grouped problem statements for a given Sub Category to LLM thereby reducing the # of overall problems
import re
from mistralai import Mistral
import pandas as pd
import time
from dotenv import load_dotenv
import os

load_dotenv()

MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key="MISTRAL_API_KEY")

#prob_cat_csv = f"/Users/raghavendradeshmukh/Deshmukh2025/PESU-CIE/Projects/CIE-Spark2025/DFVEvaluation/Top50/SubCategories/Health-Top50Percent-ProblemSubCategories.csv"
prob_cat_folder = f"/Users/raghavendradeshmukh/Deshmukh2025/PESU-CIE/Projects/CIE-Spark2025/DFVEvaluation/Top50/SubCategories/"
final_problems_folder = f"{prob_cat_folder}/FinalProblemStatements"
os.makedirs(final_problems_folder, exist_ok=True)

#If text has some junk/gibberish words.  We could do this step much earlier in the cycle though
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # remove \xa0, weird unicode, excessive spaces
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#The magic of the prompt engineering all done here
def build_prompt(row):
    label = clean_text(str(row["Label_KeyBERT"]))
    words = clean_text(str(row["Representation"]))
    docs = eval(row["Representative_Docs"]) if isinstance(row["Representative_Docs"], str) else row["Representative_Docs"]

    docs_cleaned = [clean_text(d) for d in docs]

    prompt = f"""
You are helping students learn entrepreneurship through problem-based learning.
Your task is to create ONE descriptive problem statement that synthesizes several related issues or topics.

Theme: {label}
Representative words: {words}
Representative problem statements:
- {"\n- ".join(docs_cleaned)}

Guidelines:
1. Write a descriptive problem statement of at least 3–5 sentences.
2. Explain the core issue clearly and why it matters.
3. Identify the primary people affected (stakeholders).
4. Include realistic boundary conditions (affordability, technology, social or infrastructural limitations).
5. Frame it as an open challenge that students can apply innovation and entrepreneurship frameworks to (e.g., NABC, DFV, Lean Canvas, BMC).

Output:
A single well-written problem statement.
"""
    print(prompt)
    return prompt


#Call the LLM (I am using Mistral AI with my own free API Key and can only use mistral-small) to get the Problem statement
def get_problem_statement(prompt):
    response = client.chat.complete(
        model="mistral-small",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


#This function loops through each sub-category of problem statement and creates an LLM polished single problem statement
#for Students to use during their team formation and CIE Spark activity
def rephraseProblemStatementsUsingCategories(prob_cat_folder):
    for file_name in os.listdir(prob_cat_folder):
        if not file_name.lower().endswith(".csv"):
            continue
        file_path = os.path.join(prob_cat_folder, file_name)
        only_file_name, ext = os.path.splitext(file_name)
        print(f"Rephrasing Problem Statements using Categories and LLM for: {only_file_name}")
        df = pd.read_csv(file_path)
        # Generate problem statements
        final_statements = []
        for _, row in df.iterrows():
            prompt = build_prompt(row)
            statement = get_problem_statement(prompt)
            final_statements.append(statement)
            time.sleep(5)
        
        # Save back to CSV
        df["Final_Problem_Statement"] = final_statements
        final_prob_df = df[["Final_Problem_Statement"]]
        problem_category = only_file_name.split("-")[0]
        final_problem_csv = f"{final_problems_folder}/{problem_category}-FinalProblemStatements.csv"
        final_prob_df.to_csv(final_problem_csv, index=False)
        print(f"Final Problem Statements for: {problem_category} created in: {final_problem_csv}")

    return

if __name__ == "__main__":
    rephraseProblemStatementsUsingCategories(prob_cat_folder)