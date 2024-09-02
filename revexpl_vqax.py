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
import matplotlib.pyplot as plt


def change_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad


def load_checkpoint(ckpt_path, epoch):
    
    model_name = 'VQAX/revexpl_model_{}'.format(str(epoch))
    tokenizer_name = 'VQAX/nle_gpt2_tokenizer'
    filename = 'VQAX/ckpt_stats_' + str(epoch) + '.tar'
    
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path + tokenizer_name)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(ckpt_path + model_name).to(device)   # load model with config
    opt = torch.load(ckpt_path + filename)
    optimizer = get_optimizer(model, learning_rate)
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    start_epoch = opt['epoch'] + 1
    scheduler_dic = opt['scheduler']
    del opt
    torch.cuda.empty_cache()

    return tokenizer, model, optimizer, scheduler_dic, start_epoch

def load_pretrained():
    
    model_path = 'pretrained_model/pretrain_model'
    tokenizer_path = 'pretrained_model/pretrain_tokenizer'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)        # load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)   # load model with config
    return tokenizer, model
    

def save_checkpoint(epoch, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path, **kwargs):
    
    model_name = 'VQAX/revexpl_model_{}'.format(str(epoch))
    tokenizer_name = 'VQAX/nle_gpt2_tokenizer'
    filename = 'VQAX/ckpt_stats_' + str(epoch) + '.tar'
    
    if epoch == 1:
        tokenizer.save_pretrained(ckpt_path + tokenizer_name)   # save tokenizer
    if epoch % 5 == 0:
        unwrapped_model.save_pretrained(ckpt_path + model_name, safe_serialization=False, save_function=accelerator.save)
        
        opt = {'epoch': epoch,
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(),
               **kwargs}
        
        accelerator.save(opt, ckpt_path + filename)



class VQAXDataset(Dataset):

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
        text_q = proc_ques(sample['question'])    # question
        answer = proc_ans(sample['answers'])

        exp_idx = self.index_tracker[question_id]    # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[question_id] -= 1    # decrease usage
                
        text_e = sample['explanation'][exp_idx]   # explanation
        
        # tokenization process
        q_segment_id, a_segment_id, e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', 
                                                                                         '<answer>', 
                                                                                         '<explanation>'])
        token_a = self.tokenizer.tokenize("the answer is " + answer)
        answer_len = len(token_a)
        token_e = self.tokenizer.tokenize(" because " + text_e)
        exp_len = len(token_e)
        tokens = token_a + token_e
        labels = [-100] * (answer_len + exp_len) # we don't want to predict the answer and the explanation, set to pad to ignore
        segment_ids = [a_segment_id] * answer_len
        segment_ids += [e_segment_id] * exp_len

        token_q = [self.tokenizer.bos_token] + self.tokenizer.tokenize(" the question is " + text_q) + [self.tokenizer.eos_token]
        q_len = len(token_q)
        tokens += token_q
        labels += [-100] + token_q[1:] # labels will be shifted in the model, so for now set them same as tokens
        segment_ids += [q_segment_id] * q_len
        

        if len(tokens) > self.max_seq_len :
            tokens = tokens[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            segment_ids = segment_ids[:self.max_seq_len]


        assert len(tokens) == len(segment_ids) 
        assert len(tokens) == len(labels)
        
        seq_len = len(tokens)
        padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * padding_len)
        labels = labels + ([-100] * padding_len)
        
        segment_ids += ([q_segment_id] * padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = [self.tokenizer.convert_tokens_to_ids(t) if t!=-100 else t for t in labels]
        labels = torch.tensor(labels, dtype=torch.long)
        
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        folder = 'images/train2014/' if 'train' in img_name else 'images/val2014/' 
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        qid = torch.LongTensor([int(question_id)])
        
        return (img, qid, input_ids, labels, segment_ids)

    def __len__(self):
        return len(self.ids_list)


def get_optimizer(model, learning_rate):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


accelerator = Accelerator()
device = accelerator.device

finetune_pretrained = True   # if True, finetunes from the image captioning model
eval_batch_size = 1
img_size = 224
ckpt_path = 'ckpts/'
nle_data_train_path = 'nle_data/VQA-X/vqaX_train.json'
nle_data_val_path = 'nle_data/VQA-X/vqaX_val.json'
max_seq_len = 45
load_from_epoch = None
no_sample = True   
top_k =  0
top_p =  0.9
batch_size = 32   # per GPU 
num_train_epochs = 30
weight_decay = 0
learning_rate = 2e-5 if not finetune_pretrained else 1e-5
gradient_accumulation_steps = 1   
start_epoch = 0
temperature = 1

image_encoder = ImageEncoder(device).to(device)
change_requires_grad(image_encoder, False)

if load_from_epoch is not None:
    tokenizer, model, optimizer, scheduler_dic, start_epoch = load_checkpoint(ckpt_path, load_from_epoch)
    
else:
    
    if finetune_pretrained:
        tokenizer, model = load_pretrained()
        optimizer = get_optimizer(model, learning_rate)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        orig_num_tokens = len(tokenizer.encoder)
        
        num_new_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>',
                                                       'additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
        
        assert len(tokenizer) == orig_num_tokens + num_new_tokens
        config = AutoConfig.from_pretrained('distilgpt2')
        
        # Add configs
        setattr(config, 'img_size', None)
        setattr(config, 'max_seq_len', None)   
        config.img_size = img_size
        config.max_seq_len = max_seq_len 
        config.add_cross_attention = True
        
        model = GPT2LMHeadModel.from_pretrained('distilgpt2', config = config)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        optimizer = get_optimizer(model, learning_rate)
        
print("Model Setup Ready...")


img_transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = VQAXDataset(path = nle_data_train_path, 
                                 transform = img_transform, 
                                 tokenizer = tokenizer, 
                                 max_seq_len = max_seq_len)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)

val_dataset = VQAXDataset(path = nle_data_val_path,
                              transform = img_transform,
                              tokenizer = tokenizer,
                              max_seq_len = max_seq_len)


val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size = batch_size,
                                         shuffle=False,
                                         pin_memory=True)


model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

t_total = (len(train_loader) // gradient_accumulation_steps) * num_train_epochs
warmup_steps = 0   # 0.10 * t_total
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

if load_from_epoch is not None:
    scheduler.load_state_dict(scheduler_dic)

x = []
t_loss = []
v_loss = []
for epoch in range(start_epoch, num_train_epochs):
    
    model.train()
    accum_loss = 0
    train_loss = 0
    train_count = 0
    
    for step, batch in enumerate(train_loader):
        
        
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, _, input_ids, labels, segment_ids = batch
        
        img_embeddings = image_encoder(img)
        
        outputs = model(input_ids=input_ids, 
                        past_key_values=None, 
                        attention_mask=None, 
                        token_type_ids=segment_ids, 
                        position_ids=None, 
                        encoder_hidden_states=img_embeddings, 
                        encoder_attention_mask=None, 
                        labels=labels, 
                        use_cache=False, 
                        return_dict=True)
        
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        train_loss += loss.item()
        accum_loss += loss.item()
        
        if step % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accelerator.print("\rEpoch {} / {}, Iter {} / {}, Train Loss: {:.3f}".format(epoch, 
                                                                                   num_train_epochs, 
                                                                                   step, len(train_loader), 
                                                                                   accum_loss), 
                              end='          ')
            accum_loss = 0

        train_count += 1
    
    model.eval()
    val_loss = 0
    val_count = 0
    for step, batch in enumerate(val_loader):
        val_batch = tuple(input_tensor.to(device) for input_tensor in batch)
        img, _, input_ids, labels, segment_ids = val_batch
        
        img_embeddings = image_encoder(img)
        with torch.no_grad():
            val_outputs = model(input_ids=input_ids, 
                            past_key_values=None, 
                            attention_mask=None, 
                            token_type_ids=segment_ids, 
                            position_ids=None, 
                            encoder_hidden_states=img_embeddings, 
                            encoder_attention_mask=None, 
                            labels=labels, 
                            use_cache=False, 
                            return_dict=True)
        
        eval_loss = val_outputs.loss.item()
        val_loss += eval_loss
        val_count += 1
        accelerator.print("\rEpoch {} / {}, Iter {} / {}, Validation Loss: {:.3f}".format(epoch, 
                                                                                   num_train_epochs, 
                                                                                   step, len(val_loader), 
                                                                                   eval_loss), 
                              end='          ')
            
    x.append(epoch + 1)
    t_loss.append(train_loss / train_count)
    v_loss.append(val_loss / val_count)
    try:
        train_loss_lines.remove(train_loss_lines[0])
        val_loss_lines.remove(val_loss_lines[0])
    except Exception:
        pass
    train_loss_lines = plt.plot(x, t_loss, 'r', lw=2)
    val_loss_lines = plt.plot(x, v_loss, 'b', lw=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "val_loss"])
    plt.savefig("loss.png")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_checkpoint(epoch+1, unwrapped_model, optimizer, tokenizer, scheduler, ckpt_path)