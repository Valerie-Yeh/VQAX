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

parser = argparse.ArgumentParser()
parser.add_argument("--src_data_path", type=str,
                    help="load xv dataset")
parser.add_argument("--dst_data_path", type=str,
                    help="store results")
parser.add_argument("--text_attack", action="store_true",
                    help="if True then load vqa or aokvqa dataset")
parser.add_argument("--img_background", type=str, default=None,
                    help="load snow/rain/night/evening images")
parser.add_argument("--std", type=str, default=None,
                    help="0.05 or 0.001")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    if "vqaX" in args.src_data_path:
        model_path = 'finetuned_model/VQAX/nle_model'
        tokenizer_path = 'finetuned_model/VQAX/nle_gpt2_tokenizer'
    elif "aokvqa" in args.src_data_path:
        model_path = 'finetuned_model/aokvqa/nle_model'
        tokenizer_path = 'finetuned_model/aokvqa/nle_gpt2_tokenizer'

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)   # load model
    return tokenizer, model
    
class VQAXDataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len, text_attack = True, img_background = None, std = None):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        self.text_attack = text_attack
        self.img_background = img_background
        self.std = std
        print(f"text_attack {self.text_attack}")
        if self.text_attack == True:
            for k,v in self.data.items():   
                if len(v['question']) > 1:   # most of the images have more than one questions
                    # duplicate them for loading. -1 because one explanation is already in ids_list
                    self.ids_list += [str(k)] * (len(v['question']) - 1)    

            self.index_tracker = {k: len(v['question']) - 1 for k,v in self.data.items()}


    def __getitem__(self, i):
        
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        if self.text_attack == True:
            ques_idx = self.index_tracker[quention_id]
            if ques_idx > 0:
                self.index_tracker[quention_id] -= 1    # decrease usage
                
            text_a = proc_ques(sample['question'][ques_idx])   # question

        else:
            text_a = proc_ques(sample['question'])
        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is")
        answer_len = len(answer)
        tokens += answer 

        segment_ids += [a_segment_id] * answer_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        if self.img_background in ["snow", "rain", "night", "evening"]:
            folder = 'images/' + self.img_background + '_val2014/'
        elif self.std:
            folder = 'images/' + f'noise{self.std}' + '_val2014/'
        else:
            folder = 'images/val2014/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])
        
        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)

class AOKVQADataset(Dataset):

    def __init__(self, path, transform, tokenizer, max_seq_len, text_attack = True, img_background = None, std = None):

        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len       # question + <bos> The answer is <answer> becase <explanation> <eos>
        self.data = json.load(open(path, 'r'))
        self.ids_list = list(self.data.keys())
        self.text_attack = text_attack
        self.img_background = img_background
        self.std = std
        print(f"text_attack {self.text_attack}")
        if self.text_attack == True:
            for k,v in self.data.items():   
                if len(v['question']) > 1:   # most of the images have more than one questions
                    # duplicate them for loading. -1 because one explanation is already in ids_list
                    self.ids_list += [str(k)] * (len(v['question']) - 1)    

            self.index_tracker = {k: len(v['question']) - 1 for k,v in self.data.items()}


    def __getitem__(self, i):
        
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        if self.text_attack == True:
            ques_idx = self.index_tracker[quention_id]
            if ques_idx > 0:
                self.index_tracker[quention_id] -= 1    # decrease usage
                
            text_a = proc_ques(sample['question'][ques_idx])   # question

        else:
            text_a = proc_ques(sample['question'])
        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        tokens = self.tokenizer.tokenize(text_a)
        segment_ids = [q_segment_id] * len(tokens)

        answer = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the answer is")
        answer_len = len(answer)
        tokens += answer 

        segment_ids += [a_segment_id] * answer_len

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        if self.img_background in ["snow", "rain", "night", "evening"]:
            folder = 'images/' + self.img_background + '_val2017/'
        elif self.std:
            folder = 'images/' + f'noise{self.std}' + '_val2017/'
        else:
            folder = 'images/val2017/'
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(quention_id)])
        
        return (img, qid, input_ids, segment_ids)

    def __len__(self):
        return len(self.ids_list)

def sample_sequences(model, tokenizer, loader):
    
    model.eval()
    results_full = []
    SPECIAL_TOKENS = ['<|endoftext|>', '<pad>', '<question>', '<answer>', '<explanation>']
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    because_token = tokenizer.convert_tokens_to_ids('Ġbecause')
    max_len = 20
    
    for i,batch in enumerate(loader):
        
        current_output = []
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, img_id, input_ids, segment_ids = batch
        img_embeddings = image_encoder(img)
        always_exp = False
        
        question = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
        question = question[:len(question)-14]

        with torch.no_grad():
            
            for step in range(max_len):
                
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
                
                # take care of when to start the <explanation> token
                if not always_exp:
                    
                    if prev.item() != because_token:
                        new_segment = special_tokens_ids[-2]   # answer segment
                    else:
                        new_segment = special_tokens_ids[-1]   # explanation segment
                        always_exp = True
                else:
                    new_segment = special_tokens_ids[-1]   # explanation segment
                    
                new_segment = torch.LongTensor([new_segment]).to(device)
                current_output.append(prev.item())
                input_ids = torch.cat((input_ids, prev.unsqueeze(0)), dim = 1)
                segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0)), dim = 1)
                
        decoded_sequences = tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        results_full.append({"image_id": img_id.item(), "question": question,  "explanation": decoded_sequences})
        print("\rEvaluation: Finished {}/{}".format(i, len(loader)), end='          ')
            
    return results_full

img_size = 224
max_seq_len = 45
no_sample = True   
top_k =  0
top_p =  0.9
temperature = 1

xv_data_val = args.src_data_path

tokenizer, model = load_model()
image_encoder = ImageEncoder(device).to(device)


img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if "vqaX" in xv_data_val:
    print("this is vqax")
    vqax_val_dataset = VQAXDataset(path = xv_data_val,
                                   transform = img_transform,
                                   tokenizer = tokenizer,
                                   max_seq_len = max_seq_len,
                                   text_attack = args.text_attack,
                                   img_background = args.img_background,
                                   std = args.std)


    vqax_val_loader = torch.utils.data.DataLoader(vqax_val_dataset,
                                                  batch_size = 1,
                                                  shuffle=False,
                                                  pin_memory=True)
    explanations = sample_sequences(model, tokenizer, vqax_val_loader)

elif "aokvqa" in xv_data_val:
    print("this is aokvqa")
    aokvqa_val_dataset = AOKVQADataset(path = xv_data_val,
                                       transform = img_transform,
                                       tokenizer = tokenizer,
                                       max_seq_len = max_seq_len,
                                       text_attack = args.text_attack,
                                       img_background = args.img_background,
                                       std = args.std)


    aokvqa_val_loader = torch.utils.data.DataLoader(aokvqa_val_dataset,
                                                    batch_size = 1,
                                                    shuffle=False,
                                                    pin_memory=True)
    explanations = sample_sequences(model, tokenizer, aokvqa_val_loader)

results = {}
for item in explanations:
    k = item['image_id']
    vq = item['question']
    ve = item['explanation']
    if k not in results:
        results[k] = {'question': [], 'explanation': []}
    results[k]['question'].append(vq)
    results[k]['explanation'].append(ve)

with open(args.dst_data_path, 'w') as file:
    json.dump(results, file, indent=4)