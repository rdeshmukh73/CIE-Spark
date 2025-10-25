### PESU CIE - Spark - Idea valiation for Desirability, Viability, Feasibility
### Raghavendra Deshmukh, 28-Sep-2025
#### Evaluate all the problems submitted by Students for CIE Spark
## Step 1. Google AppScript to download each of the Problem Statements form the GDrive into individual CSVs
## Step 2. Integrate the Problem statement + Who is Affected + Pain Points + Initial Evidence into one mega column for DFV
## Step 3. Read individual CSV File into a Python DF and create a smaller Data Frame aka DF which has Name, Email ID, Department, 
#               Problem Statement which will be then used for processing the DFV
## Step 4. Each file is then processed by the DFV Algorithm using the Zero Shot Classification using hugging face library
## Step 5. The same DF will be used to update the DFV Score for each problem statement
####

import pandas as pd
import os
from transformers.pipelines import pipeline

# ----------------------------
# Setup Zero-Shot Classifier
# ----------------------------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#ALERT!!!  This folder needs to be read from the .env or config file and not hardcoded
problems_csv_folder = f"/Users/raghavendradeshmukh/Deshmukh2025/PESU-CIE/Projects/CIE-Spark2025/RDCopy-CIESpark-StudentSubmittedProblems"
#problems_csv_folder = f"/Users/raghavendradeshmukh/Deshmukh2025/PESU-CIE/Projects/CIE-Spark2025/ProblemStatementCSVs"

DESIRABILITY_WEIGHT = 0.4 #40%
FEASIBILITY_WEIGHT = 0.3 #30%
VIABILITY_WEIGHT = 0.3 #30%

## The old version of the simpler DFV is here and was used initially before using a newer more meaningful version below
DFV_LABELS_OLD = {
    "Desirability": [
        "1 - Very low desirability. Only a few people/customers/consumers care, no urgency or relevance.",
        "2 - Low desirability. Small audience, not a top priority problem.",
        "3 - Moderate desirability. Some people/customers/consumers care, but not critical or urgent.",
        "4 - High desirability. Many people care and having a solution is important to quality of life or work.",
        "5 - Very high desirability. Strong widespread demand across various customer segments, urgent societal need."
    ],
    "Feasibility": [
        "1 - Not feasible. Needs breakthroughs or unavailable resources.",
        "2 - Low feasibility. Needs highly specialized tech/resources, hard to access.",
        "3 - Moderately feasible. Possible with effort, partnerships or funding.",
        "4 - High feasibility. Achievable with existing tech and resources.",
        "5 - Very high feasibility. Straightforward to implement with available tools."
    ],
    "Viability": [
        "1 - No viable model. No clear revenue/funding path, unsustainable.",
        "2 - Low viability. Only donor/NGO dependent, no long-term model.",
        "3 - Moderate viability. Weak revenue or limited funding options.",
        "4 - High viability. Clear revenue/funding, good margins possible.",
        "5 - Very high viability. Multiple revenue streams, scalable and sustainable."
    ]
}

# The improved more verbose version of the DFV criteria is below and will be used.
DFV_LABELS = {
    "Desirability": [
        "1 - Very low desirability. Problem affects very few people or a niche context; little relevance to society, education, health, finance, or daily life.",
        "2 - Low desirability. Problem is recognized but affects a small group; not urgent or strongly felt in areas like campus life, agriculture, or transportation.",
        "3 - Moderate desirability. Clear need exists for certain groups or contexts (e.g., students, farmers, patients), but not yet widespread or urgent.",
        "4 - High desirability. Large and diverse groups care; solving the problem significantly improves quality of life, education, healthcare, or sustainability.",
        "5 - Very high desirability. Strong, widespread demand across multiple customer or societal segments; urgent need with broad impact on health, economy, and environment."
    ],
    "Feasibility": [
        "1 - Very low feasibility. Solution requires major scientific breakthroughs, unavailable infrastructure, or unrealistic resources.",
        "2 - Low feasibility. Technically possible but requires rare expertise, advanced infrastructure, or costly resources that are difficult to access.",
        "3 - Moderate feasibility. Solution is possible with effort, external support, or partnerships; some technical or resource barriers remain.",
        "4 - High feasibility. Solution is achievable using existing technology, resources, or processes with manageable effort.",
        "5 - Very high feasibility. Solution is straightforward to implement with readily available tools, skills, and infrastructure; low risk of technical failure."
    ],
    "Viability": [
        "1 - Very low viability. No sustainable funding or business model; idea is only conceptual or dependent on one-time grants.",
        "2 - Low viability. Relies heavily on donations, subsidies, or NGO/government support; limited long-term sustainability.",
        "3 - Moderate viability. Some potential for revenue, cost savings, or continued funding; model is weak or restricted to specific contexts.",
        "4 - High viability. Clear and sustainable funding or revenue model; good potential margins or cost-effectiveness in practice.",
        "5 - Very high viability. Multiple revenue streams or funding sources; highly scalable, sustainable, and attractive for long-term impact."
    ]
}

# This funtion adds Columns in the same DF to capture the DFV scores individually and the calculated overall DFV score
def add_dfv_columns(df):
    df["Desirability Score"] = None
    df["Feasibility Score"] = None
    df["Viability Score"] = None
    df["DFV Score"] = None
    return df

# Main function to evaluate the DFV for each problem statement found in the passed Dataframe
def evaluate_dfv(df):
    for idx, row in df.iterrows():
        #Extract the Problem Statement
        problem_text = row.get("Consolidated Problem Statement", "")
        if not problem_text:
            print(f"The Problem Text is Empty at row {idx}.  Moving to the next")
            continue

        #Loop through the DFV Labels and the 5 Point Criteria    
        for criterion, labels in DFV_LABELS.items():
            input_text = problem_text
            try:
                #Call the Classifier with the Input Text aka Problem Statement and the Labels
                output = classifier(input_text, candidate_labels=labels, multi_label=False)
                predicted_label = output["labels"][0]
                #model_score = int(predicted_label.split("-")[0].strip())
                #Get the model score and round it off to 2 decimal places instead of approximating it to the nearest highest int
                model_score = int(predicted_label.split("-")[0].strip()) + output["scores"][0] - 1
                model_score = round (model_score, 2) # type: ignore
                
            except Exception as e:
                print(f"ERROR!!! Classifying Problem Statement at {idx} for Criteria {criterion} - Error: {e}")                
                model_score = None
            df.at[idx, f"{criterion} Score"] = model_score
    return df            


#Calculating the final DFV Score to stack rank
# D=40%, F and V = 30% weightage and then multiply by 20 since we are in a scale of 5 for each
def calculate_dfv_score(df):
    df["DFV Score"] = (
        df["Desirability Score"] * 0.40 +
        df["Feasibility Score"] * 0.30 +
        df["Viability Score"] * 0.30
    ) * 20 
    return df

# Instead of just using a simple DFV Score we are using the Pareto Frontier method to score it differently
# NOTE - This code below is not used as of now
def pareto_frontier(df, cols=["Desirability Score", "Feasibility Score", "Viability Score"]):
    """
    Compute Pareto frontier for a given DataFrame based on the specified columns.
    Keeps rows that are not strictly dominated by any other row.
    """
    data = df[cols].values

    is_efficient = [True] * len(data)

    for i, point in enumerate(data):
        if not is_efficient[i]:
            continue
        for j, other in enumerate(data):
            if i == j or not is_efficient[j]:
                continue
            # "other" dominates "point" if it is >= in all and > in at least one dimension
            if all(other >= point) and any(other > point):
                is_efficient[i] = False
                break

    # Filter and sort by DFV Score descending
    frontier_df = df[is_efficient].copy()
    frontier_df = frontier_df.sort_values(by="DFV Score", ascending=False).reset_index(drop=True)
    return frontier_df


#The function where we loop through all the CSV files and evaluate each problem statement for DFV
#1. Processes each CSV file, evaluates each problem for DFV and creates a File with DFV values,
#2. Then calculates the Top 50% and the Rest 50% of the Problem statements based on the DFV Scores and stores them into
#       relevant folders where each category's Top 50% and Rest 50% are stored
def process_idea_csv_folder(folder_path):
    #Dictionary to hold the processed Dataframes for final processing
    processed_dfs = {}
    pareto_frontiers = {}
    os.makedirs("Top50", exist_ok=True)
    os.makedirs("DFVScores", exist_ok=True)
    os.makedirs("Rest50", exist_ok=True)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        try:
            df = pd.read_csv(file_path)
            print(f"\n***Loaded {file_name} with {df.shape[0]} Problem Statements for DFV processing ***")
            
            #Add Columns for Desirability, Feasibility and Viability Scores to the DF.
            df = add_dfv_columns(df)

            df["Consolidated Problem Statement"] = (
                "Problem Statement: " + df["Problem Statement"].fillna("") +
                ". Who is Affected: " + df["Who is most affected by this problem?"].fillna("") +
                ". Pain Points: " + df["What makes this problem so frustrating or painful?"].fillna("") +
                ". Initial Evidence: " + df["What is your initial evidence?"].fillna("")
                )

            #Now pass each DF to the DFV Processor function
            print(f"Calling the DFV Evaluation for the Problem statements in: {file_name}")
            df = evaluate_dfv(df)

            #Now using the weights for DFV, calculate a final DFV Score
            df = calculate_dfv_score(df)
            processed_dfs[file_name] = df
            only_file_name, ext = os.path.splitext(file_name)
            out_file_name = f"./DFVScores/{only_file_name}-DFVScores.csv"
            df.to_csv(out_file_name)
            print(f"{out_file_name} created with DFV Scores Evaluation")

            #Compute the Top 50%  of the Problem Statements purely based on the DFV Scores
            cutoff = df['DFV Score'].quantile(0.50)  # 50th percentile (top 50%)
            # Filter for rows above or equal to the cutoff
            top_50_df = df[df['DFV Score'] >= cutoff]
            #frontier_df = frontier_df.sort_values(by="DFV Score", ascending=False).reset_index(drop=True)
            top_50_df = top_50_df.sort_values(by="DFV Score", ascending=False).reset_index(drop=True)
            top50_filename = f"./Top50/{only_file_name}-Top50Percent.csv"
            top_50_df.to_csv(top50_filename)
            print(f"{top50_filename} created with the Top 30% of the Problem Statements as per DFV")
            rest_50_df = df[df['DFV Score'] < cutoff]
            rest_50_df = rest_50_df.sort_values(by="DFV Score", ascending=False).reset_index(drop=True)
            rest50_filename = f"./Rest50/{only_file_name}-Rest50Percent.csv"
            rest_50_df.to_csv(rest50_filename)
        except Exception as e:
            print(f"ERROR!!! - Could not process file: {file_name}.  Error: {e}")

    return processed_dfs

#This function is used to print the Average of the DFV Scores individually and the calculated total DFV score for each category
def print_dfv_averages(df, df_name):
    columns = ["Desirability Score", "Feasibility Score", "Viability Score", "DFV Score"]
    stats = {}
    for col in columns:
        stats[col] = {
            "Average": df[col].mean(),
            "Highest": df[col].max(),
            "Lowest": df[col].min()
        }
    
    print(f"\n📊 DFV Stats for {df_name}:")
    for col, values in stats.items():
        print(f"{col}: Avg = {values['Average']:.2f}, Highest = {values['Highest']}, Lowest = {values['Lowest']}")


if __name__ == "__main__":
    dfs = process_idea_csv_folder(problems_csv_folder)

    #for file_name, df in dfs.items():
    #    print_dfv_averages(df, df_name=file_name)