import os
import math
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW, get_constant_schedule_with_warmup

import utils
from models.modeling_bert import BertModel
from huggingface_datasets import IMDbDataset
from models.bert_sentiment import BERT_sentiment



parser = argparse.ArgumentParser(description='BERT based Model finetuning using text classification dataset')
parser.add_argument('--projects_name', type=str, default="BERTsentiment")
parser.add_argument('--runs_name', type=str, default="")
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bert_model_name', type=str, default="bert-base-uncased")
parser.add_argument('--hf_dataset_name', type=str, default="stanfordnlp/imdb")
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--testdata_split_rate', type=int, default=10)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--is_DataParallel', type=bool, default=True)
args = parser.parse_args()

if args.wandb == True:
    import wandb
    wandb.login(key="00fe025208d55e3e209f0132d63704ebc4c03b13")
    wandb.init(project=args.projects_name,
               name=args.runs_name,
               config=args,
               )
    wandb.alert(title=f"from WandB infomation project:{args.projects_name}", 
                text=f"start run {args.runs_name}"
                )
else:
    wandb = None

# check project name
save_dir = f'./result/run_{args.runs_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print("train [info] : This project name is already running")
    raise RuntimeError("Duplicate project name. Program stopped.")


print("Registered variables in args:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

# 乱数の初期化／固定
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU') 
else: print(f'Use {torch.cuda.device_count()} GPUs')

# TokenizerとBertの定義
tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

bert = BertModel.from_pretrained(args.bert_model_name)
model = BERT_sentiment(bert=bert, output_dim=args.class_num)
summary(model)

if args.is_DataParallel == True:
    model = nn.DataParallel(model)

model = model.to(device)

# 最適化関数の定義
optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-6)
# 損失関数の定義
criterion = nn.CrossEntropyLoss().to(device)


# Dataset, Dataloaderの定義
train_dataset = IMDbDataset(hf_dataset_name = args.hf_dataset_name, 
                            is_train = True,
                            )

test_dataset = IMDbDataset(hf_dataset_name = args.hf_dataset_name, 
                           is_train = False,
                           testdata_split_rate = args.testdata_split_rate,
                           )

train_dataloader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              )

test_dataloader = DataLoader(test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=False,
                             )


# スケジューラの定義
warmup_percent = 0.2
total_steps = math.ceil(args.epochs*len(train_dataset)*1./args.batch_size)
warmup_steps = int(total_steps*warmup_percent)
scheduler = get_constant_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=warmup_steps)


# Training
best_valid_loss = float('inf')

for epoch in range(args.epochs):
    start_time = time.time()
    train_loss, train_acc = utils.train(model=model, 
                                        dataloader=train_dataloader, 
                                        tokenizer=tokenizer,
                                        optimizer=optimizer, 
                                        criterion=criterion,
                                        scheduler=scheduler,
                                        args=args,
                                        wandb=wandb,
                                        )
    
    valid_loss, valid_acc= utils.evaluate(model=model, 
                                         dataloader=test_dataloader, 
                                         tokenizer=tokenizer,
                                         criterion=criterion,
                                         args=args,
                                         wandb=wandb,
                                         )
    
    #  処理時間の計算
    end_time = time.time()
    epoch_time = end_time - start_time

    # 検証データの損失が最もいい場合は，bestとして保存
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        if args.is_DataParallel == True:
            torch.save(model.module.state_dict(), os.path.join(save_dir, f'{arg.runs_name}best_valid_loss.pt'))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, f'{arg.runs_name}best_valid_loss.pt'))
    

    if args.is_DataParallel == True:
        torch.save(model.module.state_dict(), os.path.join(save_dir, f'{arg.runs_name}_{epoch+1}epoch.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, f'{arg.runs_name}_{epoch+1}epoch.pt'))
    

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_time}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    if args.wandb:
        wandb.log({"epoch": epoch+1,
                   "train_loss": train_loss,
                   "train_acc": train_acc,
                   "valid_loss": valid_loss,
                   "valid_acc": valid_acc,
                   })
        
        wandb.alert(title=f"from WandB infomation project:{args.projects_name} run:{args.run}", 
                    text=f"at the end of {epoch+1}epoch\ntrain_loss: {train_loss}\ntrain_acc: {train_acc}\nvalid_loss: {valid_loss}\nvalid_acc: {valid_acc}\n"
                    )

wandb.finish()
