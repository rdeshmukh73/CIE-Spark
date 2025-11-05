# %% [markdown]
# ### Separate prompt evaluation for each criterion (Alternative Implementation)
# This notebook uses independent API calls for each evaluation criterion without maintaining chat context.

# %% [markdown]
# ### Key Implementation Details
# 
# **Architecture:**
# - Each evaluation criterion gets a separate, independent API call
# - No context is maintained between different criteria evaluations
# - Each call creates a fresh chat session
# 
# **Rate Limiting Strategy:**
# 1. Configurable delay between each API call (default: 1s)
# 2. Additional delay between processing different teams (default: 2s)
# 3. Retry logic with backoff for failed requests
# 4. Progress tracking with detailed logging
# 
# **Advantages:**
# - ✅ **Resilient**: If one criterion fails, others continue
# - ✅ **Transparent**: Clear progress tracking and error reporting
# - ✅ **Configurable**: Easy to adjust rate limits based on API quotas
# - ✅ **Scalable**: Can be extended to parallel processing
# 
# **Trade-offs:**
# - ⚠️ May use more tokens than multi-turn approach (no context sharing)
# - ⚠️ Takes longer due to rate limiting delays
# - ⚠️ No semantic connection between criterion evaluations

# %%
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from google.genai import chats
import time
import asyncio
import pandas as pd
from datetime import datetime
load_dotenv()

# %% [markdown]
# ### TIPSC Few-shot examples

# %%
TIPSC_FEW_SHOT_EXAMPLES = f"""
Example 1 - 
Pitch Statement:
"An AI-powered tool that detects early signs of diabetic foot ulcers using smartphone images, helping rural healthcare 
workers intervene before complications arise."

TIPSC Review:
Timely: Rising diabetes cases in rural areas make early detection critical. (Score: 5)
Important: Addresses a major healthcare gap affecting millions. (Score: 5)
Profitable: Strong market through health-tech startups and public health programs. (Score: 4)
Solvable: Feasible with current AI imaging and mobile tech. (Score: 5)
Contextual: Team has medical + AI expertise with NGO partnerships. (Score: 5)

Overall Assessment: Excellent (95%)
Brief Justification: The problem is urgent, large-scale, and solvable with current technology. 
The team’s alignment with healthcare stakeholders strengthens contextual fit and market potential.

Example 2 -
Pitch Statement:
"An app that changes your phone wallpaper every hour to keep you inspired and motivated throughout the day."

TIPSC Review:

Timely: No identifiable need or trend driving this idea. (Score: 2)
Important: Minimal user impact; cosmetic value only. (Score: 2)
Profitable: No clear revenue stream or differentiator. (Score: 1)
Solvable: Very easy to build with existing APIs. (Score: 4)
Contextual: Team lacks direction and product reasoning. (Score: 2)

Overall Assessment: Poor (40%)
Brief Justification: Technically trivial concept with no significant need, value proposition, 
or sustainable market advantage; fails to meet hackathon impact criteria.
"""

# %% [markdown]
# ### Problem Evidence and Validation (Weightage : 30%)

# %%
def prompt_prob_evidence_val(problem_statement_text) :
    return f"""
        You are an expert evaluator for university hackathon pitch decks. Your task is to assess the Problem Evidence & Validation based on the rubric below.

        RUBRIC:
        - Excellent (90-100%): 10+ interviews with diverse stakeholders; multiple direct quotes; clear quantification of time/money impact
        - Good (70-89%): 5-9 interviews; some relevant quotes; basic quantification
        - Fair (50-69%): 3-4 interviews; limited evidence; vague numbers
        - Poor (0-49%): <3 interviews; no direct evidence; purely anecdotal

        The Problem Evidence and Validation content to evaluate is here:  {problem_statement_text}

        INSTRUCTIONS:
        1. Assign ONE category: Excellent, Good, Fair, or Poor
        2. Provide a 2-3 sentence justification citing specific evidence (or lack thereof) from the pitch deck
        3. Note the approximate number of interviews mentioned (if any)

        OUTPUT FORMAT:
        Category: [Excellent/Good/Fair/Poor]
        Justification: [Your short 2-3 sentence reasoning]
        Interview Count: [Number or "Not specified"]
        """


# %% [markdown]
# ### Market Opportunity & Viability (Weightage : 20%)

# %%
def prompt_market_viability(market_opportunity_viability_text) :
   return f"""
      You are an expert evaluator for university hackathon pitch decks. Your task is to assess Market Opportunity & Viability 
      based on the rubric below.

      RUBRIC:
      - Excellent (90-100%): Clear TAM/SAM/SOM with credible sources; strong profitability argument; competitive gap identified
      - Good (70-89%): Basic market sizing; some business potential; mentions competitors
      - Fair (50-69%): Vague market references; unclear business model
      - Poor (0-49%): No market analysis; no commercial viability

      The Market Opportunity and Viability content to evaluate is here: {market_opportunity_viability_text}

      INSTRUCTIONS:
      1. Assign ONE category: Excellent, Good, Fair, or Poor
      2. Provide a 2-3 sentence justification focusing on:
         - Quality of market sizing (TAM/SAM/SOM presence and credibility)
         - Business model clarity
         - Competitive analysis depth
      3. Note if credible sources are cited for market data

      OUTPUT FORMAT:
      Category: [Excellent/Good/Fair/Poor]
      Justification: [Your short 2-3 sentence reasoning]
      Market Data Quality: [Strong/Moderate/Weak/Absent]
      """

# %% [markdown]
# ### TIPSC (Weightage : 15%)

# %%
def prompt_tipsc(tipsc_text) :
   return f"""
      You are an expert evaluator for university hackathon pitch decks. Your task is to assess Problem Significance using 
      the TIPSC framework. 
      TIPSC means the following:
      T = Timely = Is the problem curent and in need of an urgent solution or recently emergent and a solution can wait?
      I = Important = Does the solution or solving this problem matter to a large or key group of customers or market sectors/segments?
      P = Profitable = Will solving this problem yield Revenue or Value or a potential for these exist (even if limited)?
      S = Solvable = Is it possible to create a solution for this problem now given the technology and other required resources?
      C = Contextual = Is the current situation like team, policiefs, company, approach the right fit?

      Here are a few examples of how to evaluate or assess TIPSC: 
      {TIPSC_FEW_SHOT_EXAMPLES}

      RUBRIC:
      - Excellent (90-100%): Compelling urgency + major impact + clear team advantage + realistic solution path
      - Good (70-89%): Some timeliness + moderate impact + reasonable team fit
      - Fair (50-69%): Vague timing + minor impact + generic team fit
      - Poor (0-49%): No urgency + trivial problem + poor team fit

      The TIPSC Content to be evaluated is: {tipsc_text}

      INSTRUCTIONS:
      1. Assign ONE category: Excellent, Good, Fair, or Poor
      2. Provide a 2-3 sentence justification addressing:
         - Timeliness/urgency of the problem
         - Scale and severity of impact
         - Team's relevant advantage or expertise
         - Realism of proposed solution path
      3. Identify the strongest and weakest TIPSC element

      OUTPUT FORMAT:
      Category: [Excellent/Good/Fair/Poor]
      Justification: [Your reasoning]
      Strongest Element: [T/I/P/S/C]
      Weakest Element: [T/I/P/S/C]
   """

# %% [markdown]
# ### Solution Direction & Value Proposition (Weightage : 15%)
# ### FOR RD SIR TO VERIFY

# %%
def prompt_solution(solution_value_prop) :
   return f"""
      You are an expert evaluator for university hackathon pitch decks. Your task is to assess Solution Direction & Value Proposition based on the rubric below.

      RUBRIC:
      - Excellent (90-100%): Clear solution hypothesis directly addressing gaps; strong unique value proposition
      - Good (70-89%): Basic solution direction; addresses some gaps
      - Fair (50-69%): Vague solution idea; weak value proposition
      - Poor (0-49%): No clear solution direction; copies existing solutions

      Solution Hypothesis of the Pitch Deck is here: {solution_value_prop}

      INSTRUCTIONS:
      1. Assign ONE category: Excellent, Good, Fair, or Poor
      2. Provide a 2-3 sentence justification focusing on:
         - Strength of value proposition
         - Real-world impact
      3. Note the most significant presentation strength or weakness

      OUTPUT FORMAT:
      Category: [Excellent/Good/Fair/Poor]
      Justification: [Your reasoning]
      Key Strength/Weakness: [Brief description]

      """

# %% [markdown]
# ### Presentation Comprehension (Weightage : 20%)

# %%
def prompt_pres_comp(presentation_cohesion) :
   return f"""
      You are an expert evaluator for university hackathon pitch decks. Your task is to assess Presentation & Cohesion based on the rubric below.

      RUBRIC:
      - Excellent (90-100%): Compelling narrative; logical flow; professional design; clear communication
      - Good (70-89%): Mostly coherent; decent design; some gaps in logic
      - Fair (50-69%): Disjointed arguments; basic design; confusing flow
      - Poor (0-49%): Incoherent story; poor design; unclear messaging

      Summary of the Pitch Deck is here: {presentation_cohesion}


      INSTRUCTIONS:
      1. Assign ONE category: Excellent, Good, Fair, or Poor
      2. Provide a 2-3 sentence justification focusing on:
         - Narrative coherence and logical flow
         - Clarity of communication
         - Overall professional quality
      3. Note the most significant presentation strength or weakness

      OUTPUT FORMAT:
      Category: [Excellent/Good/Fair/Poor]
      Justification: [Your reasoning]
      Key Strength/Weakness: [Brief description]

"""

# %%
pitch_decks_df = pd.read_csv("EvaluateStudentIdeas/pitch_decks_cleaned.csv")

# %%
# Rename columns to replaces spaces with underscores
for col in pitch_decks_df :
    pitch_decks_df = pitch_decks_df.rename(columns={col : col.replace(' ', '_')})
pitch_decks_df = pitch_decks_df.rename(columns={'Problem_Statement_(cleaned)' : 'Problem_Statement_Cleaned'})

# %%
pitch_decks_df

# %%
client = genai.Client(api_key = os.getenv('GOOGLE_API_KEY'))

# %% [markdown]
# ### Rate Limiting Configuration
# Adjust these parameters based on your API quota and rate limits

# %% [markdown]
# ### LLM-evaluation using separate prompts for each criterion
# **Key differences from multi-turn approach:**
# - Each criterion uses an independent API call
# - No context sharing between evaluations
# - Rate limiting to avoid API throttling
# - Progress tracking for transparency

# %%
grade_cols = ['Team_Name', 'Problem_Evidence', 'Market_Opp_Viability', 'TIPSC', 'Solution_Dir_Val_Prop', 'Pres_Cohesion', 'Final_Score']
grade_df = pd.DataFrame(columns=grade_cols)

token_cols = ['Team_Name', 'Candidate_Tokens', 'Thought_Tokens', 'Input_Tokens', 'Output_Tokens', 'Total_Tokens']
token_df = pd.DataFrame(columns=token_cols)

# %%
# Configuration for rate limiting
DELAY_BETWEEN_CALLS = 1.0  # seconds between API calls
DELAY_BETWEEN_TEAMS = 2.0  # seconds between processing different teams
MAX_RETRIES = 3  # number of retries on API failure
RETRY_DELAY = 5.0  # seconds to wait before retry

# %%
# Calculate expected runtime
num_teams = len(pitch_decks_df)
num_criteria = 5
time_per_criterion = DELAY_BETWEEN_CALLS  # seconds
time_per_team = (num_criteria * time_per_criterion) + DELAY_BETWEEN_TEAMS
estimated_total_time = num_teams * time_per_team

print(f"📊 Evaluation Configuration:")
print(f"   Teams to evaluate: {num_teams}")
print(f"   Criteria per team: {num_criteria}")
print(f"   Delay between API calls: {DELAY_BETWEEN_CALLS}s")
print(f"   Delay between teams: {DELAY_BETWEEN_TEAMS}s")
print(f"\n⏱️  Estimated runtime: {estimated_total_time:.0f} seconds (~{estimated_total_time/60:.1f} minutes)")
print(f"   (Actual time may vary due to API response times and retries)")
print(f"\n💡 Tip: You can adjust rate limiting parameters in the configuration cell above")

# %%
def extract_grade_from_response(res):
    '''Extract the word score (Excellent, Good, etc.) from a single response'''
    try:
        word_score = res.text.split("Category:")[1].split("\n")[0].strip()
        return word_score
    except (IndexError, AttributeError) as e:
        print(f"Error extracting grade: {e}")
        print(f"Response text: {res.text[:200]}")
        return "Fair"  # Default fallback score

def get_token_counts(res):
    '''Extract token counts from a response'''
    prompt_tokens = res.usage_metadata.prompt_token_count if res.usage_metadata.prompt_token_count else 0
    cand_tokens = res.usage_metadata.candidates_token_count if res.usage_metadata.candidates_token_count else 0
    thought_tokens = res.usage_metadata.thoughts_token_count if res.usage_metadata.thoughts_token_count else 0
    
    return prompt_tokens, cand_tokens, thought_tokens

# %%
async def evaluate_criterion_with_retry(client, prompt, criterion_name, max_retries=MAX_RETRIES):
    '''
    Evaluate a single criterion with retry logic for robustness.
    Uses separate API call for each criterion (no context maintenance).
    '''
    for attempt in range(max_retries):
        try:
            # Create a fresh chat session for this criterion
            chat = client.aio.chats.create(model='gemini-2.5-flash-preview-09-2025')
            response = await chat.send_message(prompt)
            
            # Add delay to respect rate limits
            await asyncio.sleep(DELAY_BETWEEN_CALLS)
            
            return response
            
        except Exception as e:
            print(f"  ⚠️  Error evaluating {criterion_name} (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"  ⏳ Retrying in {RETRY_DELAY} seconds...")
                await asyncio.sleep(RETRY_DELAY)
            else:
                print(f"  ❌ Failed after {max_retries} attempts")
                raise
    
    return None

# %%
# Main evaluation loop using separate prompts for each criterion
# Each criterion gets an independent API call (no context sharing)

async def main():
    total_teams = len(pitch_decks_df)
    start_time = datetime.now()

    print(f"🚀 Starting evaluation of {total_teams} teams using separate prompts")
    print(f"⏱️  Start time: {start_time.strftime('%H:%M:%S')}")
    print(f"⚙️  Rate limiting: {DELAY_BETWEEN_CALLS}s between calls, {DELAY_BETWEEN_TEAMS}s between teams\n")

    for teamNum, team in enumerate(pitch_decks_df.itertuples()):
        print(f"{'='*60}")
        print(f"📊 Processing Team {teamNum + 1}/{total_teams}: {team.Team_Name}")
        print(f"{'='*60}")
        
        grade_df.at[teamNum, 'Team_Name'] = team.Team_Name
        token_df.at[teamNum, 'Team_Name'] = team.Team_Name
        
        total_prompt_tokens = 0
        total_cand_tokens = 0
        total_thought_tokens = 0
        
        try:
            # 1. PROBLEM EVIDENCE & VALIDATION
            print("  📝 Evaluating Problem Evidence...")
            ps_raw = "Core Problem Statement: " + team.Problem_Statement.split("Core Problem Statement:", 1)[1].strip() + "\n"
            ps_evidence = team.Problem_Evidence.split("Slide", 1)[1].split(":", 1)[1].strip()
            problem_statement_text = ps_raw + ps_evidence
            
            PE_res = await evaluate_criterion_with_retry(
                client, 
                prompt_prob_evidence_val(problem_statement_text),
                "Problem Evidence"
            )
            PE_word_score = extract_grade_from_response(PE_res)
            grade_df.at[teamNum, 'Problem_Evidence'] = PE_word_score
            p_tokens, c_tokens, t_tokens = get_token_counts(PE_res)
            total_prompt_tokens += p_tokens
            total_cand_tokens += c_tokens
            total_thought_tokens += t_tokens
            print(f"     ✓ Score: {PE_word_score}")
            
            
            # 2. MARKET OPPORTUNITY & VIABILITY
            print("  💼 Evaluating Market Opportunity...")
            market_opportunity_viability_text = team.Market_Opportunity_Viability.split("Slide", 1)[1].split(":", 1)[1].strip()
            
            MOV_res = await evaluate_criterion_with_retry(
                client,
                prompt_market_viability(market_opportunity_viability_text),
                "Market Opportunity"
            )
            MOV_word_score = extract_grade_from_response(MOV_res)
            grade_df.at[teamNum, 'Market_Opp_Viability'] = MOV_word_score
            p_tokens, c_tokens, t_tokens = get_token_counts(MOV_res)
            total_prompt_tokens += p_tokens
            total_cand_tokens += c_tokens
            total_thought_tokens += t_tokens
            print(f"     ✓ Score: {MOV_word_score}")
            
            
            # 3. TIPSC
            print("  🎯 Evaluating TIPSC...")
            tipsc_text = "Timely: " + team.TIPSC.split("Timely", 1)[1].split(":", 1)[1].strip()
            
            TIPSC_res = await evaluate_criterion_with_retry(
                client,
                prompt_tipsc(tipsc_text),
                "TIPSC"
            )
            TIPSC_word_score = extract_grade_from_response(TIPSC_res)
            grade_df.at[teamNum, 'TIPSC'] = TIPSC_word_score
            p_tokens, c_tokens, t_tokens = get_token_counts(TIPSC_res)
            total_prompt_tokens += p_tokens
            total_cand_tokens += c_tokens
            total_thought_tokens += t_tokens
            print(f"     ✓ Score: {TIPSC_word_score}")
            
            
            # 4. SOLUTION DIRECTION & VALUE PROPOSITION
            print("  💡 Evaluating Solution Direction...")
            solution_value_prop = team.Solution_Hypothesis.split("Slide", 1)[1].split(":", 1)[1].strip()
            
            sol_res = await evaluate_criterion_with_retry(
                client,
                prompt_solution(solution_value_prop),
                "Solution Direction"
            )
            sol_word_score = extract_grade_from_response(sol_res)
            grade_df.at[teamNum, 'Solution_Dir_Val_Prop'] = sol_word_score
            p_tokens, c_tokens, t_tokens = get_token_counts(sol_res)
            total_prompt_tokens += p_tokens
            total_cand_tokens += c_tokens
            total_thought_tokens += t_tokens
            print(f"     ✓ Score: {sol_word_score}")
            
            
            # 5. PRESENTATION & COHESION
            print("  🎨 Evaluating Presentation...")
            presentation_cohesion = team.Problem_Statement_Cleaned
            
            cohesion_res = await evaluate_criterion_with_retry(
                client,
                prompt_pres_comp(presentation_cohesion),
                "Presentation"
            )
            cohesion_word_score = extract_grade_from_response(cohesion_res)
            grade_df.at[teamNum, 'Pres_Cohesion'] = cohesion_word_score
            p_tokens, c_tokens, t_tokens = get_token_counts(cohesion_res)
            total_prompt_tokens += p_tokens
            total_cand_tokens += c_tokens
            total_thought_tokens += t_tokens
            print(f"     ✓ Score: {cohesion_word_score}")
            
            
            # Record token statistics
            token_df.at[teamNum, 'Input_Tokens'] = total_prompt_tokens
            token_df.at[teamNum, 'Candidate_Tokens'] = total_cand_tokens
            token_df.at[teamNum, 'Thought_Tokens'] = total_thought_tokens
            token_df.at[teamNum, 'Output_Tokens'] = total_cand_tokens + total_thought_tokens
            token_df.at[teamNum, 'Total_Tokens'] = total_prompt_tokens + total_cand_tokens + total_thought_tokens
            
            print(f"  📈 Total tokens: {token_df.at[teamNum, 'Total_Tokens']}")
            print(f"  ✅ Team {teamNum + 1} completed successfully\n")
            
            # Delay between teams to respect rate limits
            if teamNum < total_teams - 1:
                await asyncio.sleep(DELAY_BETWEEN_TEAMS)
                
        except Exception as e:
            print(f"  ❌ Error processing team {team.Team_Name}: {str(e)}")
            print(f"  Skipping to next team...\n")
            continue

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"✨ Evaluation complete!")
    print(f"⏱️  Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"📊 Teams processed: {len(grade_df)}/{total_teams}")
    print(f"🔢 Total tokens used: {token_df['Total_Tokens'].sum()}")
    print(f"{'='*60}")

# Run the async main function
asyncio.run(main())

# %%
grade_df

# %%
# Display token usage statistics
print("Token Usage Summary:")
print("="*60)
print(f"Total Input Tokens: {token_df['Input_Tokens'].sum()}")
print(f"Total Output Tokens: {token_df['Output_Tokens'].sum()}")
print(f"Total Tokens: {token_df['Total_Tokens'].sum()}")
print(f"\nAverage tokens per team: {token_df['Total_Tokens'].mean():.2f}")
print(f"Max tokens for a single team: {token_df['Total_Tokens'].max()}")
print(f"Min tokens for a single team: {token_df['Total_Tokens'].min()}")
print("="*60)

# %%
token_df

# %%
grade_df = grade_df.drop(grade_df.index[1])
grade_df

# %% [markdown]
# ### Get final scores for each idea using weightages

# %%
# Word Score to Decimal Score mapping
score_map = {'Poor' : 0.25,
             'Fair' : 0.5,
             'Good' : 0.75,
             'Excellent' : 1
            }

# %%
for teamNum, team in enumerate(grade_df.itertuples()) :
    grade_df.at[teamNum, 'Final_Score'] = 0.3 * score_map[team.Problem_Evidence] + \
                                        0.2 * score_map[team.Market_Opp_Viability] + \
                                        0.15 * score_map[team.TIPSC] + \
                                        0.15 * score_map[team.Solution_Dir_Val_Prop] + \
                                        0.2 * score_map[team.Pres_Cohesion]
    

# %%
grade_df

# %% [markdown]
# ### Implementation Comparison: Multi-turn vs Separate Prompts
# 
# **Multi-turn approach (RateRealSolutions.ipynb):**
# - ✅ Maintains context across all 5 criteria
# - ✅ May provide more coherent evaluations
# - ✅ Potentially fewer total tokens (shared context)
# - ❌ Single chat session = single point of failure
# - ❌ If one criterion fails, affects subsequent ones
# 
# **Separate prompts approach (this notebook):**
# - ✅ Independent evaluation for each criterion
# - ✅ More resilient - failures are isolated
# - ✅ Better rate limiting control
# - ✅ Can parallelize if needed (future enhancement)
# - ✅ Explicit retry logic per criterion
# - ❌ No context sharing between criteria
# - ❌ May use more tokens (repeated context)
# 
# **Rate Limiting Strategy:**
# - `{DELAY_BETWEEN_CALLS}s` delay between each API call
# - `{DELAY_BETWEEN_TEAMS}s` delay between processing teams
# - Retry logic with exponential backoff
# - Progress tracking for transparency

# %% [markdown]
# ### Save Results (Optional)

# %%
# Save evaluation results to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
grade_df.to_csv(f'alt_evaluation_results_{timestamp}.csv', index=False)
token_df.to_csv(f'alt_token_usage_{timestamp}.csv', index=False)
print(f"✅ Results saved:")
print(f"   - alt_evaluation_results_{timestamp}.csv")
print(f"   - alt_token_usage_{timestamp}.csv")

# %% [markdown]
# ---
# ### 📚 When to Use Which Approach?
# 
# **Use Multi-turn Chat (RateRealSolutions.ipynb) when:**
# - You have generous API rate limits
# - Context between criteria is important for evaluation consistency
# - You want to minimize token usage
# - You need faster execution time
# - Your dataset is small to medium sized
# 
# **Use Separate Prompts (this notebook) when:**
# - You have strict API rate limits
# - You need maximum reliability and fault tolerance
# - You want independent, unbiased evaluations per criterion
# - You're dealing with large datasets
# - You need detailed progress tracking and error handling
# - You want to experiment with different evaluation strategies per criterion
# 
# **Performance Comparison:**
# 
# | Metric | Multi-turn | Separate Prompts |
# |--------|-----------|------------------|
# | Context Sharing | ✅ Yes | ❌ No |
# | Fault Tolerance | ⚠️ Medium | ✅ High |
# | Rate Limit Control | ⚠️ Limited | ✅ Excellent |
# | Token Efficiency | ✅ Higher | ⚠️ Lower |
# | Execution Speed | ✅ Faster | ⚠️ Slower |
# | Progress Tracking | ⚠️ Basic | ✅ Detailed |
# | Retry Logic | ❌ No | ✅ Yes |
# 
# ---


