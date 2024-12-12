import json
from tqdm import tqdm

def from_json(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def search(term_index, topics, topic_flag):
    all_results = {}
    with open(f'res_binary_{topic_flag}.tsv', "w", encoding="utf-8") as outfile:
        for topic_id, topic_terms in tqdm(topics.items()):
            query_terms = [term for term, value in list(topic_terms.items())[:5] if term in term_index and value <= -5]
            
            results = {}
            for term in query_terms:
                docs = term_index[term]['Docs'].items()
                for doc_id, synergy in docs:
                    if doc_id in results:
                        results[doc_id] += synergy
                    else:
                        results[doc_id] = synergy
            
            sorted_results = sorted(results.items(), key=lambda x: x[1])[:100]
            all_results[topic_id] = sorted_results

        for topic_id, doc_list in all_results.items():
            for rank, (doc_id, score) in enumerate(doc_list, start=1):
                outfile.write(f"{topic_id}\tq0\t{doc_id}\t{rank}\t{score:.4f}\t{'synergysearch'}\n")


def main():
    term_index = from_json("super_term_index.json")
    topics_1 = from_json("tokenized_topics_1.json")

    search(term_index, topics_1, '1')

if __name__=="__main__":
    main()