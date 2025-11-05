import os, re, json, time, sqlite3, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "gemini-2.5-pro"

llm = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "response_mime_type": "application/json"
    },
)

emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

DB_PATH = "data/embeddings.db"
IDEAS_PATH = "data/conf_SparkPhase3.csv"
OUTPUT_JSON = "output/scored_ideas.json"
OUTPUT_CSV = "output/scored_ideas.csv"

SLEEP_SECONDS = 6 if "flash" in MODEL_NAME else 30

def load_examples():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT category, problem, justification, scores, embedding FROM examples")
    rows = cur.fetchall()
    conn.close()
    data = []
    for r in rows:
        emb = np.frombuffer(r[4], dtype=np.float32)
        data.append({
            "category": r[0],
            "problem": r[1],
            "justification": r[2],
            "scores": json.loads(r[3]),
            "embedding": emb
        })
    return data

examples = load_examples()

def retrieve(text, k=3):
    q = emb_model.encode(text)
    sims = []
    for ex in examples:
        sim = np.dot(q, ex["embedding"]) / (np.linalg.norm(q) * np.linalg.norm(ex["embedding"]))
        sims.append((sim, ex))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in sims[:k]]

def truncate(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n-1] + "…"

def format_ex(ex):
    return (
        "Idea: " + truncate(ex["problem"], 220) + "\n" +
        "Scores: " + json.dumps(ex["scores"], ensure_ascii=False) + "\n" +
        "Justification: " + truncate(ex["justification"], 240) + "\n"
    )

# ---------- JSON parsing ----------
JSON_BLOCK_RE = re.compile(r"(```json|```)\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", re.IGNORECASE)

def extract_json(text):
    if not text or not text.strip():
        return None
    m = JSON_BLOCK_RE.search(text)
    if m:
        text = m.group(2)
    # fallback: take first {..} or [..]
    start_obj = text.find("{")
    start_arr = text.find("[")
    start = min(x for x in [start_obj, start_arr] if x != -1) if (start_obj != -1 or start_arr != -1) else -1
    if start > 0:
        text = text[start:]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------- Scoring ----------
def build_prompt(ref_block, idea):
    # Keep prompt minimal; schema must be explicit.
    return {
        "role": "user",
        "parts": [f"""
You evaluate student innovation ideas using TIPSC.

Rubric:
- Timely: urgency now
- Important: scale & seriousness
- Profitable: realistic revenue/funding path
- Solvable: feasible with current tech/resources
- Contextual: fit for student/campus/society

Be strict. Most ideas score 5–8.

Reference examples:
{ref_block}

Idea:
{idea}

Return ONLY this JSON object:
{{
  "Timely": <int 1-10>,
  "Important": <int 1-10>,
  "Profitable": <int 1-10>,
  "Solvable": <int 1-10>,
  "Contextual": <int 1-10>,
  "average": <float>,
  "rationale": ["<~10 words>", "<~10 words>", "<~10 words>"]
}}
"""]
    }

def gemini_call(msg):
    res = llm.generate_content(msg)
    usage = getattr(res, "usage_metadata", None)
    prompt_toks = getattr(usage, "prompt_token_count", 0) if usage else 0
    resp_toks = getattr(usage, "candidates_token_count", 0) if usage else 0
    return res, res.text or "", prompt_toks, resp_toks

def score(idea):
    refs = retrieve(idea, 3)
    ref_block = "\n".join([format_ex(r) for r in refs])

    msg = build_prompt(ref_block, idea)
    res, txt, p_toks, c_toks = gemini_call(msg)
    parsed = extract_json(txt)

    fallback_used = False
    if parsed is None:
        # Second attempt: explicitly say JSON only
        fix_msg = {
            "role": "user",
            "parts": [f"Output ONLY valid JSON for the previous task. No commentary. Here was your output:\n{txt}"]
        }
        res2, txt2, p2, c2 = gemini_call(fix_msg)
        parsed = extract_json(txt2)
        p_toks += p2
        c_toks += c2

    if parsed is None:
        fallback_used = True
        parsed = {
            "Timely": 5,
            "Important": 5,
            "Profitable": 5,
            "Solvable": 5,
            "Contextual": 5,
            "average": 5.0,
            "rationale": ["fallback", "json parse failed", "model twice failed"]
        }

    return parsed, p_toks, c_toks, fallback_used

def main(limit_first_n=None):
    df = pd.read_csv(IDEAS_PATH)
    if limit_first_n:
        df = df.head(limit_first_n)

    results = []
    total_prompt_tokens = 0
    total_response_tokens = 0
    fallbacks = 0

    n = len(df)
    for idx, row in df.iterrows():
        idea = row["Core Problem Statement"]
        team = row["Team Name"] if "Team Name" in row else idx + 1
        print(f"\nScoring Idea {idx+1}/{n}")

        scores, p_tok, c_tok, fb = score(idea)

        total_prompt_tokens += p_tok
        total_response_tokens += c_tok
        if fb:
            fallbacks += 1
            print(f"Team: {team}  [FALLBACK USED]")
        else:
            print(f"Team: {team}")

        print(f"Prompt tokens: {p_tok}, Response tokens: {c_tok}")

        row_dict = row.to_dict()
        row_dict.update(scores)
        results.append(row_dict)

        # Throttle for rate limits
        time.sleep(SLEEP_SECONDS)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    print("\nBatch scoring complete.")
    print("=== Token Usage Summary ===")
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_response_tokens}")
    print(f"Total Tokens: {total_prompt_tokens + total_response_tokens}")
    print(f"Fallbacks used: {fallbacks}/{n}")

if __name__ == "__main__":
    main(limit_first_n=5)