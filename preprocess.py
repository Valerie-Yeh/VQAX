# Preprocess A-OKVQA dataset
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

data = json.load(open(args.input, 'r'))

count = 0
results = {}
for item in data:
    dictionary = {}
    dictionary['question'] = item['question']
    dictionary['answers'] = item['choices'][item['correct_choice_idx']]
    dictionary['image_id'] = item['image_id']
    dictionary['image_name'] = format(item['image_id'], '012d') + '.jpg'
    dictionary['explanation'] = item['rationales']
    results[str(count)] = dictionary
    count += 1

with open(args.output, 'w') as file:
    json.dump(results, file, indent=4)