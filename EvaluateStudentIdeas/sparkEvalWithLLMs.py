import pandas as pd

problem_statement_text = "" #From slide 1 & 2 - Problem and Evidence
market_opportunity_viability_text = "" #From slide 3 - Quantifying the problem
tipsc_text = "" #From slide 5 - Why this problem is TIPSC slide
solution_value_prop = "" #From slide 6 - Solution hypothesis
presentation_cohesion = "" #Overall Slide deck content OR a comprehensive summary


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
"A wearable hydration tracker that reminds users to drink water based on real-time sweat analysis and weather conditions."

TIPSC Review:
Timely: Wellness tech is growing, though hydration-specific solutions are not urgent. (Score: 4)
Important: Moderate market among fitness and sports users. (Score: 4)
Profitable: Viable as a premium product but niche appeal. (Score: 4)
Solvable: Current sensors and IoT make it achievable. (Score: 5)
Contextual: Team has IoT experience but limited market understanding. (Score: 3)

Overall Assessment: Good (80%)
Brief Justification: A relevant and buildable solution with moderate market potential; success depends on positioning 
and user adoption beyond enthusiasts.

Example 3 - 
Pitch Statement:
"A desktop app to remind remote workers to stretch every 30 minutes and suggest exercises."

TIPSC Review:
Timely: The post-pandemic remote work trend is stabilizing. (Score: 3)
Important: Mildly useful but low perceived urgency. (Score: 3)
Profitable: Free alternatives exist; monetization unclear. (Score: 2)
Solvable: Technically simple; easy to build. (Score: 5)
Contextual: Team has coding skills but lacks health/UX expertise. (Score: 3)

Overall Assessment: Fair (60%)
Brief Justification: Simple, achievable idea with limited novelty and unclear market traction; 
lacks compelling urgency or differentiator.

Example 4 - 
Pitch Statement:
"An app that plays motivational quotes every hour to keep users positive."

TIPSC Review:

Timely: No clear trend or urgency for hourly motivational quotes. (Score: 2)
Important: Trivial problem with low impact. (Score: 2)
Profitable: Difficult to monetize; saturated with free apps. (Score: 1)
Solvable: Technically easy but adds little value. (Score: 4)
Contextual: Team lacks psychological or design expertise. (Score: 2)

Overall Assessment: Poor (40%)
Brief Justification: While easily implementable, the idea solves no pressing problem, 
lacks clear market differentiation, and shows weak contextual relevance.

Example 5 -
Pitch Statement:
"A low-cost smart inhaler system that tracks asthma medication usage, predicts attacks using environmental data, 
and alerts caregivers in real time."

TIPSC Review:

Timely: Asthma rates are increasing due to urban pollution; immediate relevance. (Score: 5)
Important: Critical for patients, families, and healthcare providers. (Score: 5)
Profitable: Strong potential for insurance tie-ins and health partnerships. (Score: 4)
Solvable: Current IoT + predictive AI make this feasible. (Score: 5)
Contextual: Team has biomedical and data analytics background. (Score: 5)

Overall Assessment: Excellent (95%)
Brief Justification: Urgent and impactful healthcare problem with a clear path to implementation and adoption; 
strong interdisciplinary team fit enhances feasibility and trust.

Example 6 - 
Pitch Statement:
"An AI chatbot that suggests eco-friendly alternatives when users shop online — like showing sustainable brands or 
second-hand options."

TIPSC Review:

Timely: Sustainability awareness is increasing but not yet mainstream behavior. (Score: 4)
Important: Appeals to a growing but niche eco-conscious segment. (Score: 4)
Profitable: Monetization possible via affiliate or brand partnerships. (Score: 4)
Solvable: Readily achievable using APIs and recommendation engines. (Score: 5)
Contextual: Team has AI experience but limited marketing background. (Score: 3)

Overall Assessment: Good (80%)
Brief Justification: Strong alignment with sustainability trends and implementable tech; 
moderate commercial potential limited by user behavior change barriers.

Example 7 -
Pitch Statement:
"A mobile app that helps people organize their daily to-do lists using colorful emojis and sound alerts to make productivity fun."

TIPSC Review:

Timely: Productivity apps remain evergreen but oversaturated. (Score: 3)
Important: Low differentiation; helps individuals but no major impact. (Score: 3)
Profitable: Hard to stand out in a crowded, free-app market. (Score: 2)
Solvable: Simple app; easily buildable with existing frameworks. (Score: 5)
Contextual: Team has beginner-level coding skills; limited UX experience. (Score: 3)

Overall Assessment: Fair (60%)
Brief Justification: Technically achievable but lacks novelty, urgency, and clear market pull; 
execution quality will determine limited success.

Example 8 -
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
#Prompt for the Problem Evidence and Validation that carries 30% weightage
PROMPT_PROBLEM_EVIDENCE_VALIDATION = f"""
You are an expert evaluator for university hackathon pitch decks. Your task is to assess the Problem Evidence & Validation based on the rubric below.

RUBRIC:
- Excellent (90-100%): 10+ interviews with diverse stakeholders; multiple direct quotes; clear quantification of time/money impact
- Good (70-89%): 5-9 interviews; some relevant quotes; basic quantification
- Fair (50-69%): 3-4 interviews; limited evidence; vague numbers
- Poor (0-49%): <3 interviews; no direct evidence; purely anecdotal

The Problem Evidenct and Validation content to evaluate is here:  {problem_statement_text}

INSTRUCTIONS:
1. Assign ONE category: Excellent, Good, Fair, or Poor
2. Provide a 2-3 sentence justification citing specific evidence (or lack thereof) from the pitch deck
3. Note the approximate number of interviews mentioned (if any)

OUTPUT FORMAT:
Category: [Excellent/Good/Fair/Poor]
Justification: [Your short 2-3 sentence reasoning]
Interview Count: [Number or "Not specified"]
"""

PROMPT_MARKET_OPPORTUNITY_VIABILITY = f"""
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

PROMPT_TIPSC_EVALUATION = f"""
You are an expert evaluator for university hackathon pitch decks. Your task is to assess Problem Significance using 
the TIPSC framework. 
TIPSC means the following:
T = Timely = Is the problem curent and in need of an urgent solution or recently emergent and a solution can wait?
I = Important = Does the solution or solving this problem matter to a large or key group of customers or market sectors/segments?
P = Profitable = Will solving this problem yield Revenue or Value or a potential for these exist (even if limited)?
S = Solvable = Is it possible to create a solution for this problem now given the technology and other required resources?
C = Contextual = Is the current situation like team, policies, company, approach the right fit?

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

PROMPT_PRESENTATION_COMPREHENSION = f"""
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