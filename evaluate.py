from ranx import Qrels, Run, evaluate
import matplotlib.pyplot as plt
from pathlib import Path

def full_eval(qrel_file, result_binary, topic_flag):
    qrels = Qrels.from_file(qrel_file, kind="trec")
    run = Run.from_file(result_binary, kind="trec")

    results = evaluate(qrels, run, ["precision@1", "precision@10", "ndcg@10", "mrr", "map"], make_comparable=True)
    
    eval_data = [
        ["precision@1", round(float(results["precision@1"]), 5)],
        ["precision@10", round(float(results["precision@10"]), 5)],
        ["ndcg@10", round(float(results["ndcg@10"]), 5)],
        ["mrr", round(float(results["mrr"]),5)],
        ["map", round(float(results["map"]), 5)]
    ]

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=eval_data, loc='center')

    plt.savefig(f'topics_{topic_flag}_eval.png', bbox_inches='tight')
    plt.close()

def main():
    full_eval('qrel_1.tsv', 'res_binary_1.tsv', '1')

    if Path('qrel_2.tsv').exists():
        full_eval('qrel_2.tsv', 'res_binary_2.tsv', '2')

if __name__=="__main__":
    main()