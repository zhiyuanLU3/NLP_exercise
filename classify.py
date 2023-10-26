import pandas as pd
import torch
import numpy as np
import CBOW_model

model_path = 'cbow_model.pth'
cbow_model = CBOW_model.CBOW(15968, 16000)
cbow_model.load_state_dict(torch.load(model_path))
cbow_model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cbow_model.to(device)
df = pd.read_csv("output_train_new.txt", sep='\t')
sentences1 = df['#1 String'].tolist()
sentences2 = df['#2 String'].tolist()
labels = df['Quality'].tolist()



def generate_sentence_vector(sentence):
    # 分词句子
    words = sentence.split()
    word = set(words)
    word_to_ix = {word: ix for ix, word in enumerate(word)}
    ix_to_word = {ix: word for ix, word in enumerate(word)}
    # 为每个单词获取词向量
    word_vectors = [cbow_model.get_word_vector(word,word_to_ix) for word in words]
    # 计算平均句子向量
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector

X = []
for sent1, sent2 in zip(sentences1, sentences2):
    vector1 = generate_sentence_vector(sent1)
    vector2 = generate_sentence_vector(sent2)
    feature_vector = np.concatenate((vector1, vector2), axis=None)
    X.append(feature_vector)
X = np.array(X)


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_report}')
import joblib

joblib.dump(svm_classifier, 'svm_classifier.pkl')

# 加载 SVM 分类器
# loaded_classifier = joblib.load('svm_classifier.pkl')

# # 使用加载的分类器进行预测
# y_pred_loaded = loaded_classifier.predict(X_test)

# # 评估加载的分类器性能
# accuracy_loaded = accuracy_score(y_test, y_pred_loaded)

# print(f'Accuracy (Loaded Classifier): {accuracy_loaded}')
