import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import SnowballStemmer
import spacy
from pathlib import Path



def stemming_tokens(token_list):
    stemmer = SnowballStemmer('spanish')
    aux = []
    for i in range(len(token_list)):
        aux.append(stemmer.stem(token_list[i]))
    return aux

def lemmatizing(text):
    doc = nlp(text)
    lemmas = [tok.lemma_.lower() for tok in doc if tok.pos_ != 'PRON']
    return lemmas


def freq_table(df, col_name='tokens'):
    items = []
    labels = []
    column = df.iloc[:, 0]
    label = df.iloc[:, 1]
    for clm, tg in zip(column, label):
        for item in clm:
            items.append(item)
            labels.append(tg)
    df_freq = pd.DataFrame({col_name: items, 'label': labels})
    return df_freq

def worclouds(df):
    data_neg = df_clean.where(df_clean['label']=='N').dropna()['text']
    data_pos = df_clean.where(df_clean['label']=='P').dropna()['text']
    data_none = df_clean.where(df_clean['label']=='NONE').dropna()['text']
    data_neu = df_clean.where(df_clean['label']=='NEU').dropna()['text']

    plt.figure(figsize = (20,20))
    wc_N = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                collocations=False).generate(" ".join(data_neg))
    plt.title('Negative')
    plt.imshow(wc_N)
    plt.show()
    plt.figure(figsize = (20,20))
    wc_P = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                collocations=False).generate(" ".join(data_pos))
    plt.title('Positive')
    plt.imshow(wc_P)    
    plt.show()
    plt.figure(figsize = (20,20))
    wc_None = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                collocations=False).generate(" ".join(data_none))
    plt.title('None')
    plt.imshow(wc_None)    
    plt.show()
    plt.figure(figsize = (20,20))
    wc_Neu = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                collocations=False).generate(" ".join(data_neu))
    plt.title('NEU')
    plt.imshow(wc_Neu)
    plt.show()


if __name__ == "__main__":
    nlp = spacy.load('es_core_news_md')
    train = False
    if train:
        df_clean = pd.read_csv('../datasets/df_clean.csv')
    else:
        df_clean = pd.read_csv('../datasets/test_clean.csv')


    tokenizer = RegexpTokenizer(r'\w+')
    df_clean['tokens'] = df_clean['text'].apply(tokenizer.tokenize)
    df_clean['stems'] = df_clean['tokens'].apply(lambda x: stemming_tokens(x))
    df_clean['lemmas'] = df_clean['text'].apply(lambda x: lemmatizing(x))

    wordclouds = False
    if wordclouds:
        worclouds(df_clean)

    freq_tables = True
    if freq_tables:
        df_lemas_freq = freq_table(df_clean[['lemmas', 'label']], 'lemmas')
        df_lemas_freq['lemmas'].value_counts()[0:10].plot(kind='bar', title='lemmas')
        plt.show()
        df_tokens_freq = freq_table(df_clean[['tokens', 'label']], 'tokens')
        df_tokens_freq['tokens'].value_counts()[0:10].plot(kind='bar', title='tokens')
        plt.show()
        df_stems_freq = freq_table(df_clean[['stems', 'label']], 'stems')
        df_stems_freq['stems'].value_counts()[0:10].plot(kind='bar', title='stems')
        plt.show()

    if train:
        filepath = Path('../datasets/df_tokenized.csv')
        df_clean.to_csv(filepath)
    else:
        filepath = Path('../datasets/test_tokenized.csv')
        df_clean.to_csv(filepath)
