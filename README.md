# Research of the degree of feature mixing of Transformer's Self-Attention

- Transformer内に組み込まれているSelf-Attentionは，異なるLinear層で線形変換されたベクトル (Query : $Q$, Key : $K$, Value : $V$) を用いて，$Q$, $K$の内積から計算されるAttention Weightで重み付けされたValueベクトルを組み合わせて（加重和した）ベクトルを出力する．
  つまり，あるトークンの特徴ベクトル$x(i)$はSelf-Attention機構を経て，全トークンの特徴ベクトルをAttention Weightを重みとした加重和によって生成する．
  そのため，Self-Attention機構の処理後の$x(i)$は他のトークンの特徴を取り入れた新たな特徴ベクトルになる．しかし，Self-Attention機構の処理後のトークンの特徴ベクトルが，自身を含む他のトークンの特徴がどの程度混ぜ合わさった特徴になるかは不明確である．
  そこで，本研究では，Skip Connection，Feed Forward Network機構も含めて，特徴ベクトルの変化を確認し，Self-Attention機構による特徴ベクトルの混合度合いを調査する．




- 作業メモ

使用Dockerイメージ：[taiga10969/basic_image:cuda12.1.0-ubuntu22.04-python3.10](https://hub.docker.com/layers/taiga10969/basic_image/cuda12.1.0-ubuntu22.04-python3.10/images/sha256-076a9005a1daafe2910eda4354921bd852f8611fa70d040313a4504e880f981e?context=repo)<br>
1. ```python3 -m pip install --upgrade pip```<br>



### Training
IMDBデータセットを使用して，BERTモデルにLinear層を追加した2クラス (positive or negative) に分類するモデルを構築し，学習を行なった．
1. ```pip install -r requirements.txt```<br>
2. ```chmod +x train.sh```<br>
3. ```bash train.sh```<br>






