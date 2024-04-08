from datasets import load_dataset
from torch.utils.data import Dataset


class IMDbDataset(Dataset):
    def __init__(self, hf_dataset_name="stanfordnlp/imdb", is_train=True, testdata_split_rate=10):
        split = "train" if is_train else f"test[:{str(testdata_split_rate)}%]"
        self.imdb_dataset = load_dataset(hf_dataset_name, split=split)
        print(f"IMDbDataset __init__ end load dataset. >> hf_dataset_name:{hf_dataset_name}, is_train:{is_train}, data_len:{len(self.imdb_dataset)}")

    def __len__(self):
        return len(self.imdb_dataset)

    def __getitem__(self, idx):
        # テキストとラベルを返す
        text = self.imdb_dataset[idx]["text"]
        label = self.imdb_dataset[idx]["label"]
        return text, label
    

if __name__ == '__main__':

    from tqdm import tqdm
    from collections import Counter
    
    train_dataset = IMDbDataset(hf_dataset_name = "stanfordnlp/imdb", 
                                is_train = True)
    
    print("len(train_dataset) : ", len(train_dataset))
    
    i = 0
    text, label = train_dataset[i]
    print(f"train_dataset[{i}] text : {text}")
    print(f"train_dataset[{i}] label : {label}")


    # データセットの長さ
    dataset_length = len(train_dataset)
    
    # ラベルの出現回数を格納するCounterオブジェクトを作成
    label_counter = Counter()
    
    # tqdmを使って進捗を表示しながらデータセット内の全てのデータに対してラベルを調査
    for i in tqdm(range(dataset_length), desc="Processing data", unit="sample"):
        _, label = train_dataset[i]
        label_counter[label] += 1
    
    # ラベルの分布を表示
    print("Label distribution:")
    for label, count in label_counter.items():
        print(f"Label {label}: {count} occurrences")

