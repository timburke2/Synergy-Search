# Synergy Search Project

Synergy Search is a research-oriented project designed to process collections of documents, compute synergy measures between terms, and build sophisticated indices for information retrieval tasks. It provides tools for term ranking, query optimization, and evaluation using information-theoretic metrics like surprisal and synergy.

---

## Features
- **Document Tokenization**: Tokenizes and processes raw document data into unique terms.
- **Surprisal Calculation**: Measures the information content of terms within the collection.
- **Synergy Ranking**: Computes term synergy to rank terms based on their mutual information.
- **Sparse Synergy Matrix**: Efficiently generates and stores pairwise synergy values using sparse matrices.
- **Query Evaluation**: Precomputes synergy scores for query terms to optimize search results.
- **Search and Evaluation**: Supports information retrieval benchmarking using P@1, P@10, MAP, MRR, and NDCG.

---

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/timburke2/Synergy-Search
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Preprocessing Documents
Ensure that the following files are in the project directory:
- `Answers.json`: The raw answer texts.
- `topics_1.json`, `topics_2.json`: Topic data.
- `qrel_1.tsv`, `qrel_2.tsv`: Ground truth data.

Run the following command to preprocess the data:
```
python synergy_processor.py
```

### 2. Compute Synergy Matrix
To generate the synergy matrix:
```
python synergy_processor.py --compute_matrix
```

### 3. Perform Search
To execute a synergy-based search for topics:
```
python search.py
```

### 4. Evaluate Results
Evaluate search results using predefined qrels:
```
python evaluate.py
```

---

## File Structure
- **`synergy_processor.py`**: Main script for preprocessing and computing synergy metrics.
- **`search.py`**: Implements the search functionality using precomputed indices.
- **`evaluate.py`**: Evaluates search results against ground truth using standard metrics.
- **`Answers.json`**: Raw document answers.
- **`topics_1.json`** Raw topic data.
- **`topics_2.json`** Raw topic data.
- **`qrel_1.tsv`** Ground truth data for topics 1 evaluation
- **`qrel_2.tsv`** Ground truth data for topics 2 evaluation
- **`tokenized_answers.json`**: Processed tokenized answers.
- **`tokenized_topics_1.json`** Processed query data.
- **`tokenized_topics_2.json`** Processed query data.
- **`term_index.json`**: Index containing terms, surprisal, and synergy data.
- **`synergy_matrix.npz`**: Sparse synergy matrix file.
- **`res_binary_1.tsv`** Search output for topics 1
- **`res_binary_2.tsv`** Search output for topics 2
- **`topics_1_eval.png`** Data table for topics 1 evaluation
- **`topics_2_eval.png`** Data table for topics 2 evaluation
  
---

## Key Concepts
- **Surprisal**: Measures the rarity of a term in the document collection.
- **Synergy**: Quantifies how much mutual information two terms provide when used together.
- **Sparse Matrices**: Used for efficient storage and computation of term-term relationships.

---

## Example Workflow
1. **Preprocess Data**:
   ```
   python synergy_processor.py
   ```

2. **Generate Synergy Matrix**:
   ```
   python synergy_processor.py --compute_matrix
   ```

3. **Search Topics**:
   ```
   python search.py
   ```

4. **Evaluate Results**:
   ```
   python evaluate.py
   ```

---

## Dependencies
- Python 3.8+
- Libraries: `numpy`, `scipy`, `tqdm`, `bs4`, `matplotlib`, `ranx`

Install all dependencies using:
```
pip install -r requirements.txt
```

---
