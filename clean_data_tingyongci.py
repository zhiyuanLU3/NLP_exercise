import re

def remove_special_characters(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

from nltk.stem import WordNetLemmatizer
import nltk
lemmatizer = WordNetLemmatizer()       
nltk.download('wordnet')
def lemmatized_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

with open('output_test.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

cleaned_sentences = []
for line in lines:
    line_ = line.strip().split('\t')
    string1, string2, label = line_[0], line_[1], line_[2]

    string1_n = string1.lower() 
    string2_n = string2.lower() 
    string1_nn = remove_special_characters(string1_n)
    string2_nn = remove_special_characters(string2_n)
    cleaned_sentences.append([string1_nn,string2_nn,label])
    # cleaned_sentence = remove_special_characters(cleaned_sentence)
    # cleaned_sentence = remove_stopwords(cleaned_sentence)
    # cleaned_sentence = lemmatized_text(cleaned_sentence)
    # cleaned_sentences.append(cleaned_sentence)

with open('output_test_new.txt', 'w', encoding='utf-8') as output_file:
    for row in cleaned_sentences:
        output_file.write('\t'.join(row) + '\n')
