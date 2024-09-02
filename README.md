1. I use pretrained model, which is inside `pretrained_model` folder, on image captioning provided by NLX-GPT to fine tune on VQA-X training/validation dataset, and VQA-X test dataset is used in the following tasks. Therefore, the fine-tuned model is called REVEXPL, which takes image, answer and explanation as input and generates question.
2. `CreateLe.py` creates a set of statements (Le) that are inconsistent with ground truth explanations.
3. `gen_xv_woa.py` generates xv hat for each explanation hat in Le.
4. `threshold.py` eliminates empty string in question list of `vqaX_test_xv.json` or `eaokvqa_val_xv.json`, and filters out xv question's and ground truth's question BERTScore below a threshold.
5. `attack.py` attacks victim model using image and modified question (xv hat) as input.
6. `python attack.py --src_data_path nle_data/VQA-X-WOA/vqaX_test.json --dst_data_path results/vqaX_test_exp.json` inputs original questions and images and outputs answers and explanations as ground truth to compute BERT score.
7. `threshold.py` filters out adversarial samples that are below a specific threshold

For A-OKVQA dataset
1. I use REVEXPL finetuned on VQAX to generate xv.