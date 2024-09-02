import evaluate
from bert_score import score
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nle_data", type=str,
                    help="nle_data/{VQA-X, aokvqa}/{vqaX, aokvqa}_{test, val}.json")
parser.add_argument("--generation", type=str,
                    help="results/VQA-X-WOA/vqaX_test_att_exp.json")
parser.add_argument("--filter", action="store_true",
                    help="whether to compute explanations under the correct answers")
parser.add_argument("--img_attack", action="store_true")
parser.add_argument("--img_dir", type=str,
                    help="aokvqa-img or VQA-X-img")
parser.add_argument("--task", type=str,
                    help="aokvqa or vqaX")
parser.add_argument("--split", type=str,
                    help="val or test")
args = parser.parse_args()

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

ref = json.load(open(args.nle_data, 'r'))
gen = json.load(open(args.generation, 'r'))
gen_k = list(gen.keys())


def compute_metrics(predictions, references):
    bleuscore1 = bleu.compute(predictions=predictions, references=references, max_order=1)['bleu']
    bleuscore2 = bleu.compute(predictions=predictions, references=references, max_order=2)['bleu']
    bleuscore3 = bleu.compute(predictions=predictions, references=references, max_order=3)['bleu']
    bleuscore4 = bleu.compute(predictions=predictions, references=references, max_order=4)['bleu']
    rougescore = rouge.compute(predictions=predictions, references=references)['rougeL']
    meteorscore = meteor.compute(predictions=predictions, references=references)['meteor']
    P, R, F1 = score(predictions, references, model_type="microsoft/deberta-xlarge-mnli")
    print(f"BLEU-1: {bleuscore1*100:.1f}")
    print(f"BLEU-2: {bleuscore2*100:.1f}")
    print(f"BLEU-3: {bleuscore3*100:.1f}")
    print(f"BLEU-4: {bleuscore4*100:.1f}")
    print(f"ROUGEL: {rougescore*100:.1f}")
    print(f"METEOR: {meteorscore*100:.1f}")
    print(f"BERTScore: {F1.mean()*100:.1f}")

if __name__ == '__main__':
    if not args.img_attack:
        if not args.filter:
            ref_exp = []
            gen_exp = []
            for key in gen_k:
                for e in gen[key]['explanation']:
                    index = e.find('because') + 8
                    if index != -1:
                        result = e[index:]
                    else:
                        result = e
                    gen_exp.append(result)
                    ref_exp.append(ref[key]['explanation'])
            if 'vqaX' in args.nle_data:
                print("***** VQA-X Generated Text vs Ground Truth *****")
            else:
                print("***** AOKVQA Generated Text vs Ground Truth *****")
            compute_metrics(gen_exp, ref_exp)

        else:
            if 'aokvqa' in args.nle_data:
                ref_exp = []
                gen_exp = []
                acc = []
                for key in gen_k:
                    for e in gen[key]['explanation']:
                        a_index = e.find('because') - 1
                        ans = e[:a_index]
                        if ans in ref[key]['direct_answers']:
                            acc.append(1)
                            index = e.find('because') + 8
                            if index != -1:
                                result = e[index:]
                            else:
                                result = e
                            gen_exp.append(result)
                            ref_exp.append(ref[key]['explanation'])
                        else:
                            acc.append(0)

                print("***** AOKVQA Filtered Generated Text vs Ground Truth *****")
                compute_metrics(gen_exp, ref_exp)
                print(f"Accuray: {sum(acc)/len(acc)*100:.1f}")
            else:
                ref_exp = []
                gen_exp = []
                acc = []
                for key in gen_k:
                    for e in gen[key]['explanation']:
                        a_index = e.find('because') - 1
                        ans = e[:a_index]
                        if ans == ref[key]['answers'][0]['answer']:
                            acc.append(1)
                            index = e.find('because') + 8
                            if index != -1:
                                result = e[index:]
                            else:
                                result = e
                            gen_exp.append(result)
                            ref_exp.append(ref[key]['explanation'])
                        else:
                            acc.append(0)

                print("***** VQAX Filtered Generated Text vs Ground Truth *****")
                compute_metrics(gen_exp, ref_exp)
                print(f"Accuray: {sum(acc)/len(acc)*100:.1f}")
    else:
        if not args.filter:
            r_exp = []
            ori_exp = []
            for key in gen_k:
                for e in gen[key]['explanation']:
                    index = e.find('because') + 8
                    if index != -1:
                        result = e[index:]
                    else:
                        result = e
                    ori_exp.append(result)
                    r_exp.append(ref[key]['explanation'])

            print("***** Original Image vs Ground Truth *****")
            compute_metrics(ori_exp, r_exp)

            att_img_dict = {}
            att_img_idx_dict = {}
            for day in ["noise0.05", "rain", "snow", "evening", "night"]:
                att_img_dict[day] = json.load(open(f'results/{args.img_dir}/filtered_exp/{args.task}_{args.split}_att_{day}.json', 'r'))
                att_img_idx_dict[day] = list(att_img_dict[day].keys())

            for day in ["noise0.05", "rain", "snow", "evening", "night"]:
                reference = []
                attack = []
                for key in att_img_idx_dict[day]:
                    for e in att_img_dict[day][key]['explanation']:
                        index = e.find('because') + 8
                        if index != -1:
                            result = e[index:]
                        else:
                            result = e
                        attack.append(result)
                            
                        reference.append(ref[key]['explanation'])
                print(f"***** {args.task} {day} Image vs Ground Truth *****")
                compute_metrics(attack, reference)

        else:
            if 'aokvqa' in args.nle_data:
                r_exp = []
                ori_exp = []
                acc = []
                for key in gen_k:
                    for e in gen[key]['explanation']:
                        a_index = e.find('because') - 1
                        ans = e[:a_index]
                        if ans in ref[key]['direct_answers']:
                            acc.append(1)
                            index = e.find('because') + 8
                            if index != -1:
                                result = e[index:]
                            else:
                                result = e
                            ori_exp.append(result)
                            r_exp.append(ref[key]['explanation'])
                        else:
                            acc.append(0)

                print("***** Original Image vs Ground Truth *****")
                compute_metrics(ori_exp, r_exp)
                print(f"Accuray: {sum(acc)/len(acc)*100:.1f}")

                att_img_dict = {}
                att_img_idx_dict = {}
                for day in ["noise0.05", "snow", "evening", "night"]:
                    att_img_dict[day] = json.load(open(f'results/{args.img_dir}/filtered_exp/{args.task}_{args.split}_att_{day}.json', 'r'))
                    att_img_idx_dict[day] = list(att_img_dict[day].keys())

                for day in ["noise0.05", "snow", "evening", "night"]:
                    reference = []
                    attack = []
                    acc = []
                    for key in att_img_idx_dict[day]:
                        for e in att_img_dict[day][key]['explanation']:
                            a_index = e.find('because') - 1
                            ans = e[:a_index]
                            if ans in ref[key]['direct_answers']:
                                acc.append(1)
                                index = e.find('because') + 8
                                if index != -1:
                                    result = e[index:]
                                else:
                                    result = e
                                attack.append(result)
                                reference.append(ref[key]['explanation'])
                            else:
                                acc.append(0)
                    print(f"***** {args.task} {day} Filtered Image vs Ground Truth *****")
                    compute_metrics(attack, reference)
                    print(f"Accuray: {sum(acc)/len(acc)*100:.1f}")

            else:
                r_exp = []
                ori_exp = []
                acc = []
                for key in gen_k:
                    for e in gen[key]['explanation']:
                        a_index = e.find('because') - 1
                        ans = e[:a_index]
                        if ans == ref[key]['answers'][0]['answer']:
                            acc.append(1)
                            index = e.find('because') + 8
                            if index != -1:
                                result = e[index:]
                            else:
                                result = e
                            ori_exp.append(result)
                            r_exp.append(ref[key]['explanation'])
                        else:
                            acc.append(0)

                print("***** Original Image vs Ground Truth *****")
                compute_metrics(ori_exp, r_exp)
                print(f"Accuray: {sum(acc)/len(acc)*100:.1f}")

                att_img_dict = {}
                att_img_idx_dict = {}
                for day in ["noise0.05", "snow", "evening", "night"]:
                    att_img_dict[day] = json.load(open(f'results/{args.img_dir}/filtered_exp/{args.task}_{args.split}_att_{day}.json', 'r'))
                    att_img_idx_dict[day] = list(att_img_dict[day].keys())

                for day in ["noise0.05", "snow", "evening", "night"]:
                    reference = []
                    attack = []
                    acc = []
                    for key in att_img_idx_dict[day]:
                        for e in att_img_dict[day][key]['explanation']:
                            a_index = e.find('because') - 1
                            ans = e[:a_index]
                            if ans == ref[key]['answers'][0]['answer']:
                                acc.append(1)
                                index = e.find('because') + 8
                                if index != -1:
                                    result = e[index:]
                                else:
                                    result = e
                                attack.append(result)
                                reference.append(ref[key]['explanation'])
                            else:
                                acc.append(0)
                    print(f"***** {args.task} {day} Filtered Image vs Ground Truth *****")
                    compute_metrics(attack, reference)
                    print(f"Accuray: {sum(acc)/len(acc)*100:.1f}")