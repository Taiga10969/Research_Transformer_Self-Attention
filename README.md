# Research of the degree of feature mixing of Transformer's Self-Attention

Transformer内に組み込まれているSelf-Attentionは，異なるLinear層で線形変換されたベクトル (Query : $Q$, Key : $K$, Value : $V$) を用いて，$Q$, $K$の内積から計算されるAttention Weightで重み付けされたValueベクトルを組み合わせて（加重和した）ベクトルを出力する．
つまり，あるトークンの特徴ベクトル$x(i)$はSelf-Attention機構を経て，全トークンの特徴ベクトルをAttention Weightを重みとした加重和によって生成される．
そのため，Self-Attention機構の処理後の$x(i)$は他のトークンの特徴を取り入れた新たな特徴ベクトルになる．しかし，Self-Attention機構の処理後のトークンの特徴ベクトルが，自身を含む他のトークンの特徴がどの程度混ぜ合わさった特徴になるかは不明確である．
そこで，本研究では，Skip Connection，Feed Forward Network機構も含めて，特徴ベクトルの変化を確認し，Self-Attention機構による特徴ベクトルの混合度合いを調査する．

## 動作環境
使用Dockerイメージ：[taiga10969/basic_image:cuda12.1.0-ubuntu22.04-python3.10](https://hub.docker.com/layers/taiga10969/basic_image/cuda12.1.0-ubuntu22.04-python3.10/images/sha256-076a9005a1daafe2910eda4354921bd852f8611fa70d040313a4504e880f981e?context=repo)<br>
```
python3 -m pip install --upgrade pip
cd Research_transformer_Self-Attention
pip -r requirements.txt
```

   
## 1.sentence classificationモデルの構築
BERTモデルにLinear層を追加したクラス分類モデルを構築し，学習を行う．
### sentence_classificationモデル
エンコーダとしてtransformersで提供されているBERTを使用（defaultで指定している"bert-base-uncased"を想定）する．<br>
このBERTモデルをsectence_classificationモデルの引数```bert```に渡すことで，外部(🤗transformers)で定義されたモデルをエンコーダとして使用する．<br>
sectence_classificationでは，bertモデルの出力に対して，全結合層を追加してクラス分類を行う．<br>
学習時には，bertモデルを🤗transformersから直接読み込む．<br>
特徴量分布の可視化を行う際には，各特徴量を出力可能にプログラムを改変したファイル/models/modeling_bert_get_feature.pyからモデルを読み込む．<br>
-  学習時
```
from transformers import BertModel
from models.sectence_classification import sectence_classification
bert = BertModel.from_pretrained(bert_model_name)
model = sectence_classification(bert=bert, output_dim=class_num)
```
-  特徴量分布解析時
```
from models.modeling_bert_get_feature import BertModel
from models.sectence_classification import sectence_classification
bert = BertModel.from_pretrained(bert_model_name)
model = sectence_classification(bert=bert, output_dim=class_num)
```

### Training
[IMDBデータセット](https://huggingface.co/datasets/stanfordnlp/imdb)を使用して，sentence_classificationモデルを2クラス (positive or negative) 分類タスクを解くモデルとして構築し，学習を行う．
1. ```chmod +x train.sh```<br>
2. ```bash train.sh```<br>


## 2.特徴量分布の可視化
あるテストデータを入力した際のBERTモデル内の各ポイントでの特徴量を取得し，UMAPで次元削減し2次元のマップとして可視化を行う．<br>
これにより，各ポイントでの特徴量の分布を可視化し，Transformer Encoder内部での特徴空間の変化を確認する．<br>
特徴量の可視化を行う箇所については，以下の通りである．
- Input Embedding後
- PE後 ⚫︎
- Multi-Head Attention (Self-Attention) のHead　Concat後
- Multi-Head Attention (Self-Attention) の出力（Spik conection前）
- Multi-Head Attention (Self-Attention) の出力（Spik conection後）
- Feed Forward Netwarkの出力（Spik conection前）
- Feed Forward Netwarkの出力（Spik conection後） ⚫︎ ※Encoder blockの出力<br>
※ ⚫︎ : defaultの状態で```hidden_states```で獲得可能な特徴ベクトル<br>
の計5箇所で，Input EmbeddingとPE以外はTransformer Encoderの積層数分(12layer)あるため，可視化する特徴量は合計38となる．
```
python3 feature_distribution_umap.py --[option]
```
**--[options]**<br>
```dataset_index```: モデルに推論させるデータを指定．IMDB datasetのテストデータのindexを指定．(default=3)<br>
```umap_seed```: umapで次元削減する際のumapのrandom_stateを指定．(default=2)<br>
```metric```: umapで次元削減する際のumapのmetricを指定．(default="euclidean")["euclidean", "manhattan", "cosine"]etc.<br>
```trained_pth```: 学習済みパラメータのファイルパスを指定．(default=デフォルトのままtrain.shを実行してモデルを学習した際に，最もvalid lossが低かった時のパラメータを読み込みしています．)<br>
```save_path```: 特徴量の分布の可視化結果の保存先ディレクトリの指定．(default="./feature_distribution_umap")<br>
```bert_model_name```: 読み込みを行うbertのモデル名．🤗Huggingface (default='bert-base-uncased')<br>
```hf_dataset_name```: 読み込みを行うデータセット名．🤗Huggingface Datasets (default='stanfordnlp/imdb')<br>




