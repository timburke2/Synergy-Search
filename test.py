import math
from tqdm import tqdm
import scipy.sparse as sp
import json
import numpy as np

def from_json(filepath:str):
    """Load a JSON file and return the deserialized object."""
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def save_json(filepath: str, data: dict):
    """Save a dictionary as a JSON file with indentation."""
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


def compute_synergy_general(term1: str, term2: str, term_index, num_docs) -> float:
    """
    Compute synergy between two terms when both sets of docs are known.
    Returns 0 if no overlap.
    """
    actual_matches = len(term_index[term1]['Docs'] & term_index[term2]['Docs'])
    if actual_matches == 0:
        return 0

    p1 = term_index[term1]['Surprisal']
    p2 = term_index[term2]['Surprisal']
    expected_information = p1 + p2
    actual_information = -math.log2(actual_matches/num_docs)
    synergy = actual_information - expected_information

    return synergy

def get_synergy(index):
    # Get the column slice up to index1 (terms < index1)
    col_slice = normal_full_synergy_matrix[:index, index].toarray().flatten()

    # Get the row slice after index1 (terms > index1)
    row_slice = normal_full_synergy_matrix[index, index+1:].toarray().flatten()

    # Combine them with a zero at the position of the term itself
    synergy_slice = np.concatenate([col_slice, [0], row_slice])

    return synergy_slice


synergy_matrix = sp.load_npz("synergy_matrix.npz")
full_synergy_matrix = synergy_matrix + synergy_matrix.T - sp.diags(synergy_matrix.diagonal())

col_norms = np.sqrt(full_synergy_matrix.power(2).sum(axis=0))
col_norms = np.array(col_norms).ravel()

col_norms[col_norms == 0] = 1
D = sp.diags(1.0 / col_norms, offsets=0, shape=(full_synergy_matrix.shape[1], full_synergy_matrix.shape[1]))

normal_full_synergy_matrix = full_synergy_matrix * D

num_docs = len(from_json("Answers.json"))
term_index = from_json("term_index.json")
answers = from_json("tokenized_answers.json")
for term in term_index:
    term_index[term]["Docs"] = term_index[term]["Docs"].keys()

local_term_index = from_json("super_term_index.json")
sorted_terms = sorted(local_term_index.keys(), key=lambda term: local_term_index[term]['Surprisal'])
term_to_index = {term: idx for idx, term in enumerate(sorted_terms)}
index_to_term = {idx: term for idx, term in enumerate(sorted_terms)}

sentance = answers["23"]
sentance = ["the"]

num_terms = len(term_index)

epislon= 1e-10
sentance_vector = np.zeros(num_terms, dtype=float)
for term in sentance:
    index = term_to_index[term]
    term_vector = normal_full_synergy_matrix.getrow(index).toarray().flatten()
    sentance_vector = np.add(sentance_vector, term_vector)

sentance_vector = sentance_vector / np.linalg.norm(sentance_vector)
similarity_vector = sentance_vector * normal_full_synergy_matrix

counter = 0
for index in np.argsort(-similarity_vector):

    if 7 < term_index[index_to_term[index]]["Surprisal"] < 15:
        print(f"Sim: {similarity_vector[index]:.2f} | Term: {index_to_term[index]}\t | Surprisal: {term_index[index_to_term[index]]["Surprisal"]:.2f}")
        counter += 1
    if counter >= 15: break