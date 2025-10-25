### PESU CIE - Spark - Idea valiation for Desirability, Viability, Feasibility
### Raghavendra Deshmukh, 30-Sep-2025

# This is the 2nd part of the Problem Statement Evaluation for PESU CIE Spark
# This code will consider only the Top50% of the Problem Statements based on the DFV Evaluation and 
# 1. Create a Sub-Category under each Category like EdTech, Fin&Eco, Food&Agri etc
# This is needed because the high level category of problems yield a lot of disparate problems and considering 
# them as-is still gives us about ~370 problems which are one too many for us to consider for CIE Spark
# The Idea is to sub-categorize and identify patterns of problems that are same or similar based on Key words and
# this code does that using BERTopic and KeyBERT and SentenceTransformer
# BERTopic: https://maartengr.github.io/BERTopic/index.html
# KeyBERT: https://maartengr.github.io/KeyBERT/
# Note that these 2 are by the same author https://www.linkedin.com/in/mgrootendorst/
# In due course of time we can explore if these algos are performant or we need to evaluate more
# Alternatives considered and experimented by this author before using BERTopic for identifying Sub-Categories include:
# K-Means, Agglomerative and HDBScan all of which gave some sub-categories but not as well as BERTopic
#Alternatives considered and experimented by this author along with KeyBERT to identify the right name for the sub-category 
#was using a cosine similarity centroid based keyword which yields one keyword while KeyBERT gave ~3 keywords which was helpful


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from sklearn.cluster import KMeans

import pandas as pd
import ast
import numpy as np
import os

#The Base folder where the Top50% of each of the Categories of Problems are placed in CSVs
top_50_folder = "/Users/raghavendradeshmukh/Deshmukh2025/PESU-CIE/Projects/CIE-Spark2025/DFVEvaluation/Top50"
sub_cat_folder = f"{top_50_folder}/SubCategories"
os.makedirs(sub_cat_folder, exist_ok=True)

#Create an Embedding model using Sentence Transformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Initialize the KeyBERT Model to get the right Keyword for a group of Keywords to identify a given set of problem statements
# We can experiment with different embedding models to understand how well they can make KeyBERT work
kw_model = KeyBERT(model=embedding_model)

#The Bag of Words that we create as a vector representation 
vectorizer_model = CountVectorizer(
    stop_words="english",      # remove English stopwords
    max_features=1000,         # limit to top features
    ngram_range=(1, 2),        # include bi-grams for more context
)

### Note: I am setting the Clusters to 7-8 here, but it is a gut feel based on using 5-6 earlier and not seeing ideal results
### Typically we would need not use KMeans as the HDBScan should be possible, but for the Top 50% problem statements, HDBScan
### was unable to categorize them and put all of them as outliers.
### The ideal measure of the clusters can be done using something like: 
# sil = silhouette_score(embeddings, labels) OR ch = calinski_harabasz_score(embeddings, labels) to check the right ones.
cluster_model = KMeans(n_clusters=8, random_state=42)

# Fit BERTopic - This will help us create a Topic Model which will give us the Sub-Categories
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model, 
    hdbscan_model=cluster_model
)

#Cleans up and removes duplicate words so that we ensure no noisy words creep into the Sub-Categories
def clean_topic_label(words, top_n=3):
    cleaned = []
    for w in words:
        if any(w in c or c in w for c in cleaned):
            continue
        cleaned.append(w)
        if len(cleaned) >= top_n:
            break
    return " ".join(cleaned)

# Use KeyBERT to determine the right labels for the words/tokens in the Representation Docs 
def keybert_label(rep_docs, max_ngram=3):
    if not rep_docs:
        return "Misc"
    text = " ".join(rep_docs)
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, max_ngram),
        stop_words="english",
        top_n=1
    )
    return keywords[0][0] if keywords else "Misc"

# Pick the phrase closest to the centroid of all phrase embeddings using the cosine similarity, embeddings and centroid
def centroid_label(rep_phrases):
    if not rep_phrases:
        return "Misc"
    embeddings = embedding_model.encode(rep_phrases)
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embeddings)[0]
    best_idx = np.argmax(sims)
    return rep_phrases[best_idx]

# Use the Bertopic generated Representation Docs and the related problems statements to find the right
# sub category based on the Keybert and centroid based approaches.
def generateSubCategories(topics_info):
    cleaned_labels = []
    keybert_labels = []
    centroid_labels = []

    for idx, row in topics_info.iterrows():
        # Parse Representation
        rep_phrases = row["Representation"]
        if isinstance(rep_phrases, str):
            rep_phrases = ast.literal_eval(rep_phrases)

        # Parse Representative_Docs
        rep_docs = row["Representative_Docs"]
        if isinstance(rep_docs, str):
            rep_docs = ast.literal_eval(rep_docs)

        # Cleaned BERTopic
        cleaned_labels.append(clean_topic_label(rep_phrases))

        # KeyBERT
        keybert_labels.append(keybert_label(rep_docs))

        # Centroid
        centroid_labels.append(centroid_label(rep_phrases))

    # Add new columns
    topics_info["Label_Cleaned"] = cleaned_labels
    topics_info["Label_KeyBERT"] = keybert_labels
    topics_info["Label_Centroid"] = centroid_labels

    return topics_info

#This function reads the Top50 CSVs for each category of Problem Statements and generates Sub-Categories
#Stores them in a relevant CSV file
def createProblemSubCategories(top_50_folder):
    for file_name in os.listdir(top_50_folder):
        if not file_name.lower().endswith(".csv"):
            continue
        
        file_path = os.path.join(top_50_folder, file_name)
        only_file_name, ext = os.path.splitext(file_name)
        print(f"Identifying and Creating Sub-Categories for: {only_file_name}")

        #Read the CSV file
        df = pd.read_csv(file_path)
        # Convert the Problem Statement to list of strings
        problem_statements = df["Problem Statement"].dropna().astype(str).tolist()
        #Now use the BERTopic model which is using KMeans for Clustering and the Bag of Words created by the Vectorizer
        topics, probs = topic_model.fit_transform(problem_statements)

        #Get the Dataframe that has the identified Topics, Representation (raw sub-categories) and the sub-list of Problem Statements
        #that are represented by these raw sub-categories
        sub_cat_df = topic_model.get_topic_info()
        print(f"The Sub-Categories for {only_file_name} are {sub_cat_df}")

        cleaned_sub_cat_df = generateSubCategories(sub_cat_df)
        sub_cat_file = f"{sub_cat_folder}/{only_file_name}-ProblemSubCategories.csv"
        cleaned_sub_cat_df.to_csv(sub_cat_file)
        print(f"Sub-Categories for {only_file_name} created in {sub_cat_file}")
    return


#Start here to create subcategory of problems
if __name__ == "__main__":
    createProblemSubCategories(top_50_folder)