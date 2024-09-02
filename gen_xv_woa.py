import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from PIL import Image
from accelerate import Accelerator
from models.gpt import GPT2LMHeadModel
from models.clip_vit import ImageEncoder
from utils.data_utils import *
from utils.eval_utils import top_filtering
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--vqax", action="store_true",
                    help="if True then use vqax dataset")
parser.add_argument("--aokvqa", action="store_true",
                    help="if True then use aokvqa dataset")
parser.add_argument("--src_data_path", type=str,
                    help="load le data")
parser.add_argument("--dst_data_path", type=str,
                    help="store xv data")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model_path = 'ckpts/VQAX_woa/revexpl_model_30'
    tokenizer_path = 'ckpts/VQAX_woa/nle_gpt2_tokenizer'

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)   # load model
    return tokenizer, model
    
class VQAXEvalDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # The answer is <answer> becase <explanation> <bos> <question> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())

        for k,v in self.data.items():   
            if len(v['explanation']) > 1:   # some questions have more than one explanation
                # duplicate them for loading. -1 because one explanation is already in ids_list
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)    

        self.index_tracker = {k: len(v['explanation']) - 1 for k,v in self.data.items()}


    def __getitem__(self, i):
        
        question_id = self.ids_list[i]
        sample = self.data[question_id]
        img_name = sample['image_name']

        exp_idx = self.index_tracker[question_id]    # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[question_id] -= 1    # decrease usage
                
        text_e = sample['explanation'][exp_idx]   # explanation

        # tokenization process
        q_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<explanation>'])
        
        token_e = self.tokenizer.tokenize("because " + text_e)
        segment_ids = [e_segment_id] * len(token_e)
        tokens = token_e

        token_q = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the question is ")
        segment_ids += [q_segment_id] * len(token_q)
        tokens += token_q
        

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        folder = 'images/train2014/' if 'train' in img_name else 'images/val2014/'   # test and val are both in val2014
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(question_id)])
        
        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)
    
class AOKVQADataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # The answer is <answer> becase <explanation> <bos> <question> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())

        for k,v in self.data.items():   
            if len(v['explanation']) > 1:   # some questions have more than one explanation
                # duplicate them for loading. -1 because one explanation is already in ids_list
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)    

        self.index_tracker = {k: len(v['explanation']) - 1 for k,v in self.data.items()}


    def __getitem__(self, i):
        
        question_id = self.ids_list[i]
        sample = self.data[question_id]
        img_name = sample['image_name']

        exp_idx = self.index_tracker[question_id]    # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[question_id] -= 1    # decrease usage
                
        text_e = sample['explanation'][exp_idx]   # explanation

        # tokenization process
        q_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<explanation>'])
        
        token_e = self.tokenizer.tokenize("because " + text_e)
        segment_ids = [e_segment_id] * len(token_e)
        tokens = token_e

        token_q = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the question is ")
        segment_ids += [q_segment_id] * len(token_q)
        tokens += token_q
        

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        folder = 'images/val2017/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(question_id)])
        
        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)


def sample_sequences(model, tokenizer, loader):
    
    model.eval()
    predictions = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    max_len = 20
    
    for i,batch in enumerate(loader):
        
        current_output = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids = batch
        img_embeddings = image_encoder(img)
        
        with torch.no_grad():
            
            for step in range(max_len + 1):
                
                if step == max_len:
                    break
                
                outputs = model(input_ids=input_ids, 
                                past_key_values=None, 
                                attention_mask=None, 
                                token_type_ids=segment_ids, 
                                position_ids=None, 
                                encoder_hidden_states=img_embeddings, 
                                encoder_attention_mask=None, 
                                labels=None, 
                                use_cache=False, 
                                return_dict=True)
                
                lm_logits = outputs.logits 
                logits = lm_logits[0, -1, :] / temperature
                logits = top_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
                
                if prev.item() in special_tokens_ids:
                    break
                
                new_segment = special_tokens_ids[2]   # explanation segment
                    
                new_segment = torch.LongTensor([new_segment]).to(device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)
                
        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        predictions.append({"image_id": img_id.item(), "question": decoded_sequences})

        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')
            
    return predictions


img_size = 224
max_seq_len = 45
no_sample = True
top_k =  0
top_p =  0.9
temperature = 1
save_path = 'results/'

le_data = args.src_data_path

tokenizer, model = load_model()
image_encoder = ImageEncoder(device).to(device)


img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

vqax_val_dataset = VQAXEvalDataset(path = le_data,
                                   transform = img_transform,
                                   tokenizer = tokenizer,
                                   max_seq_len = max_seq_len)


vqax_val_loader = torch.utils.data.DataLoader(vqax_val_dataset,
                                              batch_size = 1,
                                              shuffle=False,
                                              pin_memory=True)


aokvqa_val_dataset = AOKVQADataset(path = le_data,
                                   transform = img_transform,
                                   tokenizer = tokenizer,
                                   max_seq_len = max_seq_len)


aokvqa_val_loader = torch.utils.data.DataLoader(aokvqa_val_dataset,
                                                batch_size = 1,
                                                shuffle=False,
                                                pin_memory=True)


if args.vqax:
    print("this is vqax")
    predictions = sample_sequences(model, tokenizer, vqax_val_loader)
elif args.aokvqa:
    print("this is aokvqa")
    predictions = sample_sequences(model, tokenizer, aokvqa_val_loader)

predictions_file = args.dst_data_path

results = {}
for item in predictions:
    k = str(item['image_id'])
    v = item['question']
    if k not in results:
        results[k] = []
    results[k].append(v)

val_data = json.load(open(le_data, 'r'))
ids_list = list(val_data.keys())

for idx in ids_list:
    val_data[idx]['question'] = results[idx]

with open(predictions_file, 'w') as file:
    json.dump(val_data, file, indent=4)
