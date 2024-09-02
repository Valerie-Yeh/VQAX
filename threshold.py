import json
from bert_score import score
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nle_data", type=str,
                    help='nle_data/VQA-X/vqaX_test.json or nle_data/aokvqa/aokvqa_val.json')
parser.add_argument("--src_data_path", type=str,
                    help='results/VQA-X-WOA/vqaX_test_xv.json or results/aokvqa-woa/aokvqa_val_xv.json')
parser.add_argument("--dst_data_path", type=str,
                    help='results/VQA-X-WOA/vqaX_test_th_xv.json or results/aokvqa-woa/aokvqa_val_th_xv.json')
parser.add_argument("--threshold", type=str)
args = parser.parse_args()

data_xv = json.load(open(args.src_data_path, 'r'))
xv_ids_list = list(data_xv.keys())

# Delete empty question
for id in xv_ids_list:
    non_empty_indices = [i for i, item in enumerate(data_xv[id]['question']) if item != '']
    data_xv[id]['question'] = [data_xv[id]['question'][i] for i in non_empty_indices]
    data_xv[id]['explanation'] = [data_xv[id]['explanation'][i] for i in non_empty_indices]
    if not data_xv[id]['question']:
        del data_xv[id]

xv_ids_list = list(data_xv.keys())
# Filtering
threshold = float(args.threshold)

def compute_metrics(predictions, references):
    P, R, F1 = score(predictions, references, model_type="microsoft/deberta-xlarge-mnli")
    return F1

ref = json.load(open(args.nle_data, 'r'))

results = {}
for key in tqdm(xv_ids_list):
    ref_q = []
    att_q = []
    for q in data_xv[key]['question']:
        ref_q.append(ref[key]['question'])
        att_q.append(q)

    f1_score = compute_metrics(att_q, ref_q).tolist()
    for j in range(len(f1_score)):
        if f1_score[j] >= threshold and f1_score[j] != 1:
            if key not in results:
                results[key] = {'question': [], 'explanation': []}
            results[key]['question'].append(data_xv[key]['question'][j])
            results[key]['explanation'].append(data_xv[key]['explanation'][j])
            results[key]['answers'] = data_xv[key]['answers']
            results[key]['image_id'] = data_xv[key]['image_id']
            results[key]['image_name'] = data_xv[key]['image_name']

with open(args.dst_data_path, 'w') as file:
    json.dump(results, file, indent=4)