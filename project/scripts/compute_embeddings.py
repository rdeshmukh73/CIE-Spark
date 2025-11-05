import json, sqlite3, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

conn = sqlite3.connect("data/embeddings.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT,
    problem TEXT,
    justification TEXT,
    scores TEXT,
    embedding BLOB
)
""")

with open("./data/conf_TISPC-Examples_condensed.json", "r") as f:
    examples = json.load(f)

for ex in examples:
    text = ex["problem"] + " " + ex["justification"]
    vec = model.encode(text)
    vec_bytes = vec.astype(np.float32).tobytes()
    
    cur.execute("""
        INSERT INTO examples (category, problem, justification, scores, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (
        ex["category"],
        ex["problem"],
        ex["justification"],
        json.dumps(ex["scores"]),
        vec_bytes
    ))

conn.commit()
conn.close()

print("Success! Embeddings stored.")