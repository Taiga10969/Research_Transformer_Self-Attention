import os
import umap
import torch
import argparse
import matplotlib.pyplot as plt
from transformers import BertTokenizer

from models.modeling_bert_get_feature import BertModel
from huggingface_datasets import IMDbDataset
from models.sectence_classification import sectence_classification

## check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU') 
else: print(f'Use {torch.cuda.device_count()} GPUs')

save_path = './feature_distribution_umap'

parser = argparse.ArgumentParser(description='feature_distribution umap')
parser.add_argument('--dataset_index', type=int, default=3)
parser.add_argument('--umap_seed', type=int, default=2)
parser.add_argument('--metric', type=str, default="euclidean", choices=["euclidean", "manhattan", "cosine"])
parser.add_argument('--trained_pth', type=str, default='./result/run_training_imdb_001/training_imdb_001best_valid_loss.pt')
parser.add_argument('--save_path', type=str, default='./feature_distribution_umap')
parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased')
parser.add_argument('--hf_dataset_name', type=str, default="stanfordnlp/imdb")
args = parser.parse_args()

# make directory
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
save_dir = os.path.join(args.save_path, f"dataset{args.dataset_index}_umap_seed{args.umap_seed}_metric_{args.metric}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(os.path.join(save_dir,"pdf")):
    os.makedirs(os.path.join(save_dir,"pdf"))
if not os.path.exists(os.path.join(save_dir,"png")):
    os.makedirs(os.path.join(save_dir,"png"))

## 学習済みのText Classificationモデルの定義／読み込み ===============================
##モデルの定義
OUTPUT_DIM = 2
bert = BertModel.from_pretrained(args.bert_model_name)
tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
model = sectence_classification(bert, OUTPUT_DIM).to(device)

##学習済みモデルの読み込み
msg = model.load_state_dict(torch.load(args.trained_pth, map_location=torch.device(device)))
print(f"model.load_state_dict info : {msg}")

## データセットの定義 ==============================================================
testdata_split_rate=100
test_dataset = IMDbDataset(hf_dataset_name = args.hf_dataset_name, 
                           is_train = False,
                           testdata_split_rate = testdata_split_rate,
                           )
imdb_class = ["negative", "positive"]

## 使用データの読み込み =============================================================
text, label = test_dataset[args.dataset_index]
print(f"text (len : {len(text)}) : {text}")

## 推論 ===========================================================================
BATCH=0
input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
tokens = tokenizer.convert_ids_to_tokens(input_ids[BATCH])

## BERTモデルに入力し、隠れ状態と出力を取得
model = model.eval()
with torch.no_grad():
    outputs = model.research_forward(input_ids)
    prediction_class = torch.argmax(outputs["logit"][BATCH])
state = True if prediction_class == label else False
print(f"prediction_class : {imdb_class[prediction_class]} ({state})")

## ================================================================================
## UMAPによる次元圧縮による2次元特徴量分布の可視化 =======================================
## 可視化する特徴量を選択／定義
## 今回は，入力から1層目までの特徴量から特徴量の分布の変化を確認する
## 可視化する特徴量の位置は以下の通りである．

## 1. embed_output : Input Embedding後
vec_1 = outputs["embed_output"][BATCH].cpu()
## 2. Add_PE
vec_2 = outputs["hidden_states"][0][BATCH].cpu()
## 3. attn_head_concat : Self-Attention の Head　Concat後 Valueの加重和
vec_3 = [outputs["attn_head_concat"][i][BATCH].cpu() for i in range(len(outputs["attn_head_concat"]))]
## 4. attn_linear : Self-Attention の出力（Spik conection前）
vec_4 = [outputs["attn_linear"][i][BATCH].cpu() for i in range(len(outputs["attn_linear"]))]
## 5. attn_addskip : の出力（Spik conection後）
vec_5 = [outputs["attn_addskip"][i][BATCH].cpu() for i in range(len(outputs["attn_addskip"]))]
## 6. ffn_out : Feed Forward Netwarkの出力（Spik conection前）
vec_6 = [outputs["ffn_out"][i][BATCH].cpu() for i in range(len(outputs["ffn_out"]))]
## 7. Feed-Forward
vec_7 = [outputs["hidden_states"][i+1][BATCH].cpu() for i in range(len(outputs["hidden_states"])-1)]

# [vector_name, label, layer]
vecs = [[vec_1, 'Input Embedding' , 'layer: -'],
        [vec_2, 'add PE', 'layer: -'], 
         ]
for index in range(12):
    vecs.append([vec_3[index], 'Self-Attention (Head Concat)',      f'layer: {index+1}'] )
    vecs.append([vec_4[index], 'Self-Attention (Linear output)',       f'layer: {index+1}'] )
    vecs.append([vec_5[index], 'Self-Attention (add skip conection)',  f'layer: {index+1}'] )
    vecs.append([vec_6[index], 'Feed-Forward (Linear output)',               f'layer: {index+1}'] )
    vecs.append([vec_7[index], 'Feed-Forward (add skip conection)',               f'layer: {index+1}'] )


# 色の指定 layer_name:[color,sub_color]
# sub_colorは，重ねていく際に色を薄くしたい時に使用（予定）
color_sets = {'Input Embedding':['darkred', 'lightcoral'], 
             'add PE':['fuchsia', 'plum'], 
             'Self-Attention (Head Concat)':['gold', 'khaki'], 
             'Self-Attention (Linear output)':['orange', 'navajowhite'],
             'Self-Attention (add skip conection)':['green', 'lightgreen'], 
             'Feed-Forward (Linear output)':['deepskyblue', 'skyblue'], 
             'Feed-Forward (add skip conection)':['blueviolet', 'thistle'], 
}

## UMAPで可視化する際に用いるmapperを上記で定義したベクトル全てを使って作成する．==============
## (可視化する特徴空間の基準:選択した全特徴量)
combined_vector = [item[0] for item in vecs]
combined_vector = torch.cat(combined_vector, dim=0)


## mapper関数の作成 =================================================================
mapper = umap.UMAP(random_state=args.umap_seed, 
                   metric=args.metric, 
                   )
_ = mapper.fit_transform(combined_vector)

## 作成したmapper関数を使用して，各特徴量を次元圧縮して，可視化する．=========================
plot_size=8
## 1枚づつ上に載せていく形で表示する方法 ==================================================
legend_handles = []
for i, vec in enumerate(vecs):
    map_vec = mapper.transform(vec[0])
    scatter = plt.scatter(map_vec[:, 0], map_vec[:, 1], c=color_sets[vec[1]][0], s=plot_size, label=vec[1])
    if i < 7:
        legend_handles.append(scatter)

    plt.legend(handles=legend_handles, loc='upper right')
    plt.title(f"{vec[2]}, {vec[1]}", fontsize=15, loc='left')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.xlim(-10,25)
    plt.ylim(-15,30)
    plt.savefig(os.path.join(save_dir, "png", f"feature_umap_all_base_{i}.png"))
    plt.savefig(os.path.join(save_dir, "pdf", f"feature_umap_all_base_{i}.pdf"))
plt.close()

## ===================================================================================
#legend_handles = []
##for i, vec in enumerate(vecs):
#for num in range(len(vecs)):
#    for i, vec in enumerate(vecs[:num+1]):
#        map_vec = mapper.transform(vec[0])
#        print(f'i:{i}, num:{num}')
#        scatter = plt.scatter(map_vec[:, 0], map_vec[:, 1], c=color_sets[vec[1]][0], s=plot_size, label=vec[1])
#        if scatter not in legend_handles:
#            legend_handles.append(scatter)
#
#    plt.legend(handles=legend_handles, loc='upper right')
#    plt.title(f"{vec[2]}, {vec[1]}", fontsize=20, loc='left')
#    plt.xlabel('UMAP Dimension 1')
#    plt.ylabel('UMAP Dimension 2')
#    #　初回に描画した際のxlim, ylimを保存
#    plt.xlim(-10,25)
#    plt.ylim(-13,23)
#    plt.savefig(os.path.join(save_path, f"feature_umap_all_base_{i}.png"))
#    plt.savefig(os.path.join(save_path, f"feature_umap_all_base_{i}.pdf"))
#    plt.close()


'''
## UMAPで可視化する際に用いるmapperを1.のベクトルだけを使って作成する．==============
## (可視化する特徴空間の基準:vec_1)
## mapper関数の作成 =================================================================
mapper_2 = umap.UMAP(random_state=random_seed, 
                     metric=metric,
                     )
_ = mapper_2.fit_transform(vec_1)

## 作成したmapper関数を使用して，各特徴量を次元圧縮して，可視化する．=========================
mpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plot_size=8

legend_handles = []
for i, vec in enumerate(vecs):
    map_vec = mapper_2.transform(vec)
    scatter = plt.scatter(map_vec[:, 0], map_vec[:, 1], c=mpl_colors[i], s=plot_size, label=vec_names[i])
    legend_handles.append(scatter)

    plt.legend(handles=legend_handles, loc='best')
    #plt.title('UMAP Visualization with Labels')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.xlim(-20, 20)
    plt.ylim(-40, 20)
    plt.savefig(os.path.join(save_path, f"feature_umap_vec_1_base_{i}.png"))
    plt.savefig(os.path.join(save_path, f"feature_umap_vec_1_base_{i}.pdf"))
plt.close()
'''
