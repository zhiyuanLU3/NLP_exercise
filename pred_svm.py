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
df = pd.read_csv("output_test_new.txt", sep='\t')
sentences1 = df['#1 String'].tolist()
sentences2 = df['#2 String'].tolist()
labels = df['Quality'].tolist()

def generate_sentence_vector(sentence):
    words = sentence.split()
    word = set(words)
    word_to_ix = {word: ix for ix, word in enumerate(word)}
    word_vectors = [cbow_model.get_word_vector(word,word_to_ix) for word in words]
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector

X = []
for sent1, sent2 in zip(sentences1, sentences2):
    vector1 = generate_sentence_vector(sent1)
    vector2 = generate_sentence_vector(sent2)
    feature_vector = np.concatenate((vector1, vector2), axis=None)
    X.append(feature_vector)
X = np.array(X)

from sklearn.metrics import accuracy_score

import joblib
loaded_classifier = joblib.load('svm_classifier.pkl')
y_pred_loaded = loaded_classifier.predict(X)
accuracy_loaded = accuracy_score(labels, y_pred_loaded)

print(f'Accuracy (Loaded Classifier): {accuracy_loaded}')
