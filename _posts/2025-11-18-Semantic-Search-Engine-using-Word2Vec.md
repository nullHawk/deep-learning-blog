---
title: "Building a Semantic Search Engine for 1M+ arXiv Papers (<10ms Query Time)"
date: 2025-11-18
categories: [Deep Learning, Natural Language Processing]
tags: [deep-learning, neural-networks, nlp]
author: nullHawk
math: true
description:
---


Using a custom-trained Word2Vec model, centroid-based embeddings, and a FAISS ANN index, I built a semantic search engine that retrieves relevant arXiv papers in **<10 ms**, even at million-scale. This post covers the full approach — from training the model to optimizing the data pipeline — along with the challenges and engineering decisions behind it.

**Live Demo:** <https://huggingface.co/spaces/nullHawk/arxive-semantic-search>  
**GitHub Repo:** <https://github.com/nullHawk/word2vec-semantic-search>  
**Notebook Reference:** <https://github.com/nullHawk/word2vec-semantic-search/blob/main/semantic_search_using_word2vec_runpod_latest.ipynb>


---

## **Why Semantic Search?**

Keyword search breaks down quickly with scientific papers:

* Authors use different terminology for the same idea.
* Important concepts might not appear as exact tokens in the abstract.
* Keyword matching ignores context and meaning.

So I wanted a system that **understands queries at a conceptual level**, not just as a string of words.
That’s how this semantic search engine began.

---

## **Approach Overview**

Here is the full pipeline that powers semantic search on 1M+ arXiv abstracts:

### **1. Training Word2Vec on the Entire arXiv Corpus**

Instead of pre-trained embeddings, I trained a **Word2Vec model on the entire dataset**, allowing the vector space to align with scientific terminology.

Steps:

* Preprocessed all abstracts
* Tokenized text using NLTK
* Trained a Word2Vec model (Gensim) on millions of sentences

This produced domain-specific word vectors that capture scientific semantics better than general-purpose models.

---

### **2. Generating Abstract Embeddings via Centroids**

For each arXiv paper:

1. Tokenize the abstract
2. Convert each token into a word vector
3. Compute the **centroid** (average vector)

The centroid acts as a representation of the entire document, capturing its core meaning.

This allowed fast, lightweight embedding computation — crucial for million-scale vector databases.

---

### **3. From Cosine Search to FAISS ANN**

Initially, I used cosine similarity over all vectors:

```python
similarities = cosine_similarity(query_vec, all_vectors)
```

This worked — but took **~90-100 seconds** on a million vectors.

To fix this, I switched to **FAISS Approximate Nearest Neighbor (ANN)** search:

* Built a FAISS index (IVF/Flat depending on experiments)
* Stored only references to the actual DB rows
* Achieved sub-10ms top-k queries

This alone improved search speed from **100 seconds → milliseconds**.

---

### **4. Memory Optimization: Pandas → Dask → DuckDB**

Initially, I loaded the entire dataset using pandas:

* Used ~12GB RAM

Then I moved to **Dask** using Parquet:

* Reduced memory load
* Faster partial reads, but slower random-access lookups because data is split across many Parquet row groups and requires scanning multiple chunks

Finally, the best performer was:

### **DuckDB + Precomputed Hash Maps**

* Loaded the dataset into an embedded DuckDB database
* Built hash lookups for metadata at app launch
* Ultra-fast random row access
* Persistent & container-friendly

This made the backend both stable and zero-configuration for Docker deployments.

---

## **How Search Works Internally**

### **1. Convert query → vector**

Tokenize the user query, fetch word embeddings, compute centroid.

### **2. FAISS returns top-k similar vectors**

```
distances, indices = index.search(query_vec, k)
```

### **3. Lookup paper metadata in DuckDB**

The FAISS index stores only integer IDs pointing to rows in `arxiv.db`.

### **4. Streamlit UI displays results**

Papers are ranked by semantic similarity.

---


## **Performance Summary**

| Stage                    | Time       |
| ------------------------ | ---------- |
| Query embedding          | ~1 ms      |
| FAISS ANN search         | **<10 ms** |
| Metadata lookup (DuckDB) | ~2–3 ms    |
| Total end-to-end         | **~15 ms** |

That’s near-instant semantic search over **1M+ documents**.

---

## **Key Learnings**

* **Domain-specific Word2Vec** beats generic embeddings for scientific papers.
* **Centroid embeddings** are surprisingly effective and extremely efficient.
* **FAISS** turns million-scale search into milliseconds.
* **DuckDB** is perfect for local analytic workloads and random access.
* **Optimizing data loading** matters as much as optimizing models.

---

## **Future Improvements**

Here are several upgrades I plan to add (and you can too):

1. Move to Transformer-Based Embeddings

    Models like:

    - SPECTER / SPECTER2
    - SciBERT
    - E5-large
    - Instructor-L

    offer better semantic representation than Word2Vec.
    The challenge is embedding 1M+ documents efficiently — requiring batching, GPUs, or offline preprocessing.

2.  Use Vector Compression for Faster FAISS Search

    FAISS supports:
    - OPQ (Optimized Product Quantization)
    - PQ (Product Quantization)
    - HNSW graphs

    These reduce memory footprint and speed up million-scale search.

3. Add Hybrid Search (Keyword + Semantic)

    Combining BM25 with semantic search improves:
    - precision
    - interpretability
    - typo tolerance

4. Add Reranking Models

    After retrieving top-200 papers, use a cross-encoder (e.g., ms-marco-MiniLM-L-6-v2) to rerank results with higher accuracy.

5. Add Caching + Query Logs
    To make popular queries instantaneous and enable personalization.

---

## **Final Thoughts**

This project started as a small experiment but turned into a fully optimized semantic search engine for large-scale scientific data. It combines classical NLP (Word2Vec), modern vector search (FAISS), and pragmatic system design (DuckDB + Docker).
