# モデルの定義
# 事前学習済みモデルの後段に線形関数を追加
import torch.nn as nn
#from .modeling_bert import BertModel

class BERT_sentiment(nn.Module):
    def __init__(self, bert, output_dim):
        super().__init__()
        
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, inputs):
        #text = [batch size, sent len]

        #embedded = [batch size, emb dim]
        embedded = self.bert(input_ids=inputs["input_ids"], 
                             attention_mask=inputs["attention_mask"],
                             )[1]
        
        #output = [batch size, out dim]
        output = self.out(embedded)
        
        return output

if __name__ == '__main__':
    import torch
    from .modeling_bert import BertModel
    # check GPU usage
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count()==0: print('Use 1 GPU') 
    else: print(f'Use {torch.cuda.device_count()} GPUs')

    OUTPUT_DIM = 2
    bert = BertModel.from_pretrained('bert-base-uncased')

    model = BERT_sentiment(bert, OUTPUT_DIM).to(device)