import os
import csv
import pdfplumber
import re
import unicodedata
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

# === CONFIGURATION ===
INPUT_FOLDER = "./PitchPDFs"        # Folder containing all pitch deck PDFs
OUTPUT_FILE = "pitch_decks.csv"
CLEANED_OUTPUT_FILE =  "pitch_decks_cleaned.csv"

llm_client = genai.Client(api_key = os.getenv('GOOGLE_API_KEY'))

# Define mapping of logical slide sections to page indices (0-indexed)
PAGE_MAPPING = {
    "Problem Statement": 1,                # Page 2
    "Problem Evidence": 2,                 # Page 3
    "Market Opportunity Viability": 3,     # Page 4
    "TIPSC": 4,                            # Page 5
    "Competition": 5,                      # Page 6
    "Solution Hypothesis": 6,              # Page 7
    "References": 7                        # Page 8
}

#Extracts the meaningful Problem Statement section from a slide's text. 
# Starts from the first occurrence of 'Core Problem Statement' or 'Problem Statement'
#and keeps everything after it, removing any preceding content like names or departments.
def extract_problem_section(text: str) -> str:
    if not text:
        return ""

    # Normalize line endings and spacing
    text = text.replace('\r', '\n')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Define flexible patterns for start of relevant content
    pattern = re.compile(
        r'(Core\s+Problem\s+Statement[:\-].*|Problem\s+Statement[:\-].*)',
        flags=re.IGNORECASE | re.DOTALL
    )
    # Search for the first occurrence of the pattern
    match = pattern.search(text)
    if match:
        # Keep everything from the start of the matched phrase
        cleaned_text = text[match.start():].strip()
    else:
        # If not found, return the text as-is (fallback)
        cleaned_text = text

    # Optional: remove any residual redundant spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    print(cleaned_text)
    return cleaned_text

#Text extracted from PPTs/Canva into PDF and then into CSV will have some encoding errors and hence result in non-legible
#characters in text and also at times the representation of symbols will be incorrect which will impact the evaluation
#of the Pitch Deck text via LLMs and/or NLP or ZSC approaches and hence we clean it in this function.
def clean_text(text: str) -> str:
    """
    Cleans messy text extracted from PDFs without losing meaningful data.
    """

    if not text:
        return ""

    # Step 1: Normalize Unicode to fix encoding inconsistencies
    text = unicodedata.normalize("NFKC", text)
   
    # Step 2: Replace common mojibake (encoding errors)
    replacements = {
        "‚Äô": "'",      # apostrophe
        "‚Äö": '"',
        "‚Äö√Ñ√∂": '"',
        "‚Äì": "-",      # en dash
        "‚Äî": "—",      # em dash
        "‚Äú": '"',
        "‚Äù": '"',
        "‚Ä¶": "...",
        "‚Çπ": "₹",      # rupee symbol
        "√": "",         # stray math symbols
        "Ã—": "x",
        "Â": "",         # often appears before spaces
        "•": "-",        # bullet replacement
        "‚": ",",
        "„": ",",
        "√¢‚Ç¨‚Ñ¢": "'",  # weird single quote artifact
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # Step 3: Remove non-printable characters (control chars, weird spaces)
    text = ''.join(ch for ch in text if ch.isprintable())

    # Step 4: Replace multiple spaces or line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Removes Slide labels like "Slide 2: etc"
def remove_slide_labels(text):
    if not text:
        return ""
    # Remove variants like "Slide 5:", "Slide 6 -", "Slide 7–", etc.
    cleaned_text = re.sub(r'Slide\s*\d+\s*[:\-–]?', '', text, flags=re.IGNORECASE)
    # Remove any double spaces left behind
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    return cleaned_text

#This is the function that calls the LLM to create a Summary and general evaluation of the overall Pitch Deck flow.
#Using Gemini Flash lite 2.5 for the same
def createProblemSummary(text):
    summary_prompt = f"""
        You are evaluating a undergraduate student submitted problem pitch content for pitch clarity, coherence and compelling narrative.
        Given the following sections from the pitch deck: {text}

        Provide the following:
        1. A concise 150 word summary of the pitch narrative.
        2. A short evaluation (3-5 sentences) of how cohesive and logical the overall pitch deck feels.
        3. Rate it on a Scale of 1 to 10 where 1 is the lowest and 10 is the highest.
        4. Optionally, provide one Key Strength and one Area of improvement for the Slide.
    """
    print(f"Input Text is: {text}\n")
    response = llm_client.models.generate_content(model = 'gemini-2.5-flash-lite', contents=summary_prompt)
    print(f"The Summary and Evaluation from the LLM is: \n: {response.text}\n")
    return response

#This function will create a Problem Summary by first adding up Problem, Evidence, Market Opportunity, Competition and Hypothesis
#It will then clean up some un-wanted words from the Text like Slide numbers if any.
#It will then invoke an LLM to get the summary of the slide
def summarizeProblems():
    df = pd.read_csv(CLEANED_OUTPUT_FILE)
    for i, row in df.iterrows():
        problem_statement = f"Problem Statement: {df['Problem Statement (cleaned)'][i]}"
        problem_evidence = f"\nProblem Evidence: {df['Problem Evidence'][i]}"
        market_opportunity = f"\nMarket Opportunity and Viability: {df['Market Opportunity Viability'][i]}"
        competition = f"\nCompetition: {df['Competition'][i]}"
        solution_hypothesis = f"\nSolution Hypothesis: {df['Solution Hypothesis'][i]}"
        input_slide_text = problem_statement + problem_evidence + market_opportunity + competition + solution_hypothesis
        input_slide_text = remove_slide_labels(input_slide_text)
        #print(input_slide_text)
        createProblemSummary(input_slide_text)
    return

# === MAIN SCRIPT ===
def extract_text_from_pdf(pdf_path):
    """Extracts required pages’ text from the PDF and returns as a dict."""
    data = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for key, page_index in PAGE_MAPPING.items():
                if page_index < total_pages:
                    page = pdf.pages[page_index]
                    text = page.extract_text() or ""
                    #text = clean_text(page.extract_text() or "")
                    # Clean up whitespace and line breaks
                    text = ' '.join(text.split())
                    
                else:
                    text = ""
                cleaned_text = clean_text(text)
                data[key] = cleaned_text
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        for key in PAGE_MAPPING.keys():
            data[key] = ""
    return data


def main():
    # Create output CSV and write header
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ["Team Name", "Problem Statement", "Problem Evidence",
                      "Market Opportunity Viability", "TIPSC", "Competition",
                      "Solution Hypothesis", "References"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through all PDFs in folder
        for filename in os.listdir(INPUT_FOLDER):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(INPUT_FOLDER, filename)
                print(f"📖 Reading: {filename}")

                extracted = extract_text_from_pdf(pdf_path)
                row = {"Team Name": filename.replace(".pdf", "")}
                row.update(extracted)
                writer.writerow(row)

    print(f"\n✅ Extraction complete! Data saved to '{OUTPUT_FILE}'")
    
    
    df = pd.read_csv(OUTPUT_FILE)
    df["Problem Statement (cleaned)"] = df["Problem Statement"].apply(extract_problem_section)
    df.to_csv(CLEANED_OUTPUT_FILE, index=False)
    print(f"\n✅ Cleaned Data saved to '{CLEANED_OUTPUT_FILE}'")

    summarizeProblems()

if __name__ == "__main__":
    main()
