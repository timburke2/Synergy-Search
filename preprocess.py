import json
import math
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, save_npz

class SynergyProcessor:
    """
    Processes a collection of documents and terms to compute 'synergy' measures
    between terms, build indices, and optionally create a synergy matrix.
    """

    def __init__(self, 
                 answers_filepath="Answers.json", 
                 tokenized_answers_filepath="tokenized_answers.json",
                 term_index_filepath="term_index.json",
                 topics_1_filepath="topics_1.json",
                 topics_2_filepath="topics_2_.json",
                 tokenized_topics_1_filepath="tokenized_topics_1.json",
                 tokenized_topics_2_filepath="tokenized_topics_2.json"):
        """
        Initialize paths and placeholders for data structures.
        """
        self.answers_filepath = answers_filepath
        self.tokenized_answers_filepath = tokenized_answers_filepath
        self.term_index_filepath = term_index_filepath
        self.topics_1_filepath = topics_1_filepath
        self.topics_2_filepath = topics_2_filepath
        self.tokenized_topics_1_filepath = tokenized_topics_1_filepath
        self.tokenized_topics_2_filepath = tokenized_topics_2_filepath

        self.term_index = None
        self.num_docs = None

    def load_data(self):
        """
        Ensure that tokenized answers, term index, and tokenized topics are available.
        Loads them if needed and updates internal state.
        """
        valid_input_files = [
            Path(self.answers_filepath).exists(),
            Path(self.topics_1_filepath).exists(),
            Path(self.topics_2_filepath).exists()
        ]

        if not all(valid_input_files):
            print("Error loading input files. Ensure Answers.json, topics_1.json, and topics_2.json are in the project directory.")
            return

        if not Path(self.tokenized_answers_filepath).exists():
            self.tokenize_answers(self.answers_filepath)

        if not Path(self.term_index_filepath).exists():
            self.build_initial_index(self.tokenized_answers_filepath)

        self.term_index = self.from_json(self.term_index_filepath)

        answers = self.from_json(self.answers_filepath)
        self.num_docs = len(answers)

        # Tokenize topics if needed
        if not Path(self.tokenized_topics_1_filepath).exists():
            self.tokenize_topics(self.topics_1_filepath, self.tokenized_topics_1_filepath)

        if not Path(self.tokenized_topics_2_filepath).exists():
            self.tokenize_topics(self.topics_2_filepath, self.tokenized_topics_2_filepath)
    
    def from_json(self, filepath:str):
        """Load a JSON file and return the deserialized object."""
        with open(filepath, 'r', encoding='utf-8') as infile:
            return json.load(infile)
    
    def save_json(self, filepath: str, data: dict):
        """Save a dictionary as a JSON file with indentation."""
        with open(filepath, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)

    def clean_text(self, text: str) -> list[str]:
        """
        Extract and clean text from HTML, splitting into tokens and filtering out unwanted terms.
        """
        soup = BeautifulSoup(text, "lxml")
        cleaned_text = soup.get_text(separator=" ")
        tokens = cleaned_text.lower().split()

        clean_tokens = []
        for token in tokens:
            stripped_token = re.sub(r"^[\"']+|[\"']+$", "", token)
            # Check if token is alphabetic (or contains apostrophes) and isn't trivial
            if re.match(r"^[a-zA-Z']+$", stripped_token):
                if len(stripped_token) > 1 or stripped_token in ('a', 'i'):
                    clean_tokens.append(stripped_token)
        return clean_tokens

    def tokenize_answers(self, answers_filepath: str) -> None:
        """
        Convert answer texts into sets of unique tokens and save results.
        """
        answers = self.from_json(answers_filepath)
        answer_data = {}
        for answer in tqdm(answers, desc='Tokenizing documents'):
            doc_id = answer['Id']
            text = self.clean_text(answer['Text'])
            unique_terms = set(text)
            answer_data[doc_id] = list(unique_terms)
        
        self.save_json(self.tokenized_answers_filepath, answer_data)

    def build_initial_index(self, tokenized_answers_filepath: str) -> None:
        """
        Build an initial term index that includes Surprisal and placeholders for synergy scores.
        """
        answers = self.from_json(tokenized_answers_filepath)
        collection_size = len(answers)

        term_data = {}
        for doc_id, terms in tqdm(answers.items(), desc='Building initial index'):
            for term in terms:
                if term not in term_data:
                    term_data[term] = {
                        "Surprisal": 0,
                        "Average Synergy": None,
                        "Docs": {}
                    }
                if doc_id not in term_data[term]["Docs"]:
                    term_data[term]["Docs"][doc_id] = None

        # Compute Surprisal for each term
        for term in tqdm(term_data, desc='Calculating Surprisal'):
            term_subset = len(term_data[term]["Docs"])
            term_data[term]["Surprisal"] = -math.log2(term_subset / collection_size)

        self.save_json(self.term_index_filepath, term_data)

    def compute_synergy_fast(self, term1_docs: set, term1_surprisal: float, term2: str) -> float:
        """
        Quickly compute synergy between a known set of docs for term1 and a second term.
        Returns None if no synergy.
        """
        if term2 not in self.term_index:
            return None
        term2_docs = self.term_index[term2]["Docs"]
        term2_surprisal = self.term_index[term2]["Surprisal"]

        # Using dict-key intersection as sets
        actual_matches = len(term1_docs & term2_docs)
        if actual_matches == 0:
            return None

        expected_information = term1_surprisal + term2_surprisal
        actual_information = -math.log2(actual_matches / self.num_docs)
        synergy = actual_information - expected_information
        return synergy

    def compute_synergy_general(self, term1: str, term2: str) -> float:
        """
        Compute synergy between two terms when both sets of docs are known.
        Returns 0 if no overlap.
        """
        actual_matches = len(self.term_index[term1]['Docs'] & self.term_index[term2]['Docs'])
        if actual_matches == 0:
            return 0

        p1 = self.term_index[term1]['Surprisal']
        p2 = self.term_index[term2]['Surprisal']
        expected_information = p1 + p2
        actual_information = -math.log2(actual_matches/self.num_docs)
        synergy = actual_information - expected_information

        return synergy

    def rank_terms_synergy(self, topic_terms: list[str]) -> dict[str, float]:
        """
        Given a set of topic terms, compute a synergy-based ranking.
        """
        num_terms = len(topic_terms)
        synergy_terms = {}
        for term1 in topic_terms:
            if term1 in self.term_index:
                synergy_sum = 0
                for term2 in topic_terms:
                    if term2 in self.term_index:
                        synergy_sum += self.compute_synergy_general(term1, term2)
                synergy_terms[term1] = (synergy_sum * self.term_index[term1]['Surprisal']) / num_terms
        return dict(sorted(synergy_terms.items(), key=lambda item: item[1]))

    def tokenize_topics(self, topics_filepath: str, tokenized_topics_filepath: str) -> None:
        """
        Tokenize topics and rank their terms by synergy, then save to file.
        """
        topics = self.from_json(topics_filepath)

        topic_data = {}
        for topic in tqdm(topics, desc='Tokenizing topics'):
            topic_id = topic['Id']
            combined_text = f"{topic['Title']} {topic['Body']} {' '.join(topic['Tags'])}"
            text = self.clean_text(combined_text)
            unique_terms = set(text)
            ranked_terms = self.rank_terms_synergy(list(unique_terms))
            topic_data[topic_id] = ranked_terms

        self.save_json(tokenized_topics_filepath, topic_data)

    def precompute_synergy_query_terms(self, num_terms: int = 5, surprisal_threshold: int = 7, process_all_terms: bool = False):
        """
        Precompute synergy for query terms.
        If process_all_terms is False, only process top terms from topics that meet the threshold.
        If True, process all terms above the surprisal threshold.
        """
        local_term_index = self.from_json(self.term_index_filepath)
        answers = self.from_json(self.tokenized_answers_filepath)
        topics1 = self.from_json(self.tokenized_topics_1_filepath)
        topics2 = self.from_json(self.tokenized_topics_2_filepath)

        # Convert local docs dicts to sets for quick intersection
        for term in local_term_index:
            local_term_index[term]["Docs"] = set(local_term_index[term]["Docs"].keys())

        # Determine which terms to process
        if process_all_terms:
            terms_to_process = {term for term in self.term_index if self.term_index[term]["Surprisal"] > surprisal_threshold}
        else:
            terms_to_process = set()
            for topic_id in topics1:
                top_terms = list(topics1[topic_id].keys())[:num_terms]
                for term in top_terms:
                    if term in self.term_index and self.term_index[term]["Surprisal"] > surprisal_threshold:
                        terms_to_process.add(term)

            for topic_id in topics2:
                top_terms = list(topics2[topic_id].keys())[:num_terms]
                for term in top_terms:
                    if term in self.term_index and self.term_index[term]["Surprisal"] > surprisal_threshold:
                        terms_to_process.add(term)

        reverse_index = {doc_id: set(terms) for doc_id, terms in answers.items()}

        # Process each term and update synergy values if missing
        for term in tqdm(terms_to_process, desc="Processing terms for queries"):
            term1_docs = local_term_index[term]["Docs"]
            term1_surprisal = local_term_index[term]["Surprisal"]

            total_synergy = 0
            doc_count = 0

            for doc_id in term1_docs:
                # Skip already processed docs
                if self.term_index[term]["Docs"][doc_id] is not None:
                    continue
                if doc_id not in reverse_index:
                    continue

                terms_in_doc = reverse_index[doc_id]
                synergies = []
                for other_term in terms_in_doc:
                    if other_term != term:
                        synergy = self.compute_synergy_fast(term1_docs, term1_surprisal, other_term)
                        if synergy is not None:
                            synergies.append(synergy)

                if synergies:
                    average_doc_synergy = sum(synergies) / len(synergies)
                    self.term_index[term]["Docs"][doc_id] = average_doc_synergy
                    total_synergy += average_doc_synergy
                    doc_count += 1

            if doc_count > 0:
                self.term_index[term]["Average Synergy"] = total_synergy / doc_count

            self.save_json(self.term_index_filepath, self.term_index)

    def compute_synergy_matrix(self):
        """
        Build and save a sparse synergy matrix for all terms, skipping zeros to save space.
        """
        local_term_index = self.from_json(self.term_index_filepath)
        sorted_terms = sorted(local_term_index.keys(), key=lambda term: local_term_index[term]['Surprisal'])

        # Assign indices to terms
        term_to_index = {term: idx for idx, term in enumerate(sorted_terms)}
        total_terms = len(local_term_index)

        synergy_values = {}

        # Convert Docs to sets for quick intersection
        for term in local_term_index:
            local_term_index[term]["Docs"] = set(local_term_index[term]["Docs"].keys())

        # Compute pairwise synergies
        for i, term1 in tqdm(enumerate(sorted_terms), total=len(sorted_terms), desc='Processing terms'):
            idx1 = term_to_index[term1]
            term1_docs = local_term_index[term1]["Docs"]
            term1_surprisal = local_term_index[term1]["Surprisal"]
            for term2 in sorted_terms[i + 1:]:
                idx2 = term_to_index[term2]
                synergy = self.compute_synergy_fast(term1_docs, term1_surprisal, term2)
                if synergy != 0:
                    synergy_values[(idx1, idx2)] = synergy

        # Create sparse matrix from synergy values
        if synergy_values:
            row_indices, col_indices = zip(*synergy_values.keys())
            values = np.array(list(synergy_values.values()), dtype=float)

            if np.any(np.isnan(values)):
                print("NaN values detected in synergy values.")
                values = np.nan_to_num(values)

            sparse_matrix = csr_matrix(
                (values, (row_indices, col_indices)),
                shape=(total_terms, total_terms)
            )

            save_npz('synergy_matrix', sparse_matrix)
        else:
            print("Synergy values failed to compute")
            pass

    def run(self, compute_matrix=False):
        """
        Main entry point to load data, precompute synergy for query terms, and optionally compute the synergy matrix.
        """
        self.load_data()
        self.precompute_synergy_query_terms(process_all_terms=False)

        if compute_matrix:
            self.compute_synergy_matrix()
            self.precompute_synergy_query_terms(process_all_terms=True)

if __name__ == "__main__":
    calculator = SynergyProcessor()
    calculator.run()
