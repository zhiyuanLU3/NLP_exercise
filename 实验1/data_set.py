from torch.utils.data import Dataset

file_path = 'E:\\Study\\NLP\\output_strings_new.txt'

# 自定义Dataset类
class MyDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()  # 调用父类的构造函数
        self.data = []  # 存储数据的列表

        # 读取文本文件并处理数据
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
                # 假设每行包含一个句子，你可能需要进一步的数据清洗
        text = text.split()  # 假设用空格分词
        self.data = text

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        # 实现该方法以返回数据集中指定索引位置的样本
        sample = self.data[idx]
        return sample 