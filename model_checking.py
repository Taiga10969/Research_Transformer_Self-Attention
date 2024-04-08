import torch
from models.modeling_bert import BertModel
from models.bert_sentiment import BERT_sentiment

from torchinfo import summary 

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU') 
else: print(f'Use {torch.cuda.device_count()} GPUs')

OUTPUT_DIM = 2
bert = BertModel.from_pretrained('bert-base-uncased')
model = BERT_sentiment(bert, OUTPUT_DIM).to(device)

print(model)
summary(model)