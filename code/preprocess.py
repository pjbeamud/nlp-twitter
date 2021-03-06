import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import string
import emoji
from pathlib import Path

def distribution(df):
    axis = df[['id','label']].groupby('label').count().plot(kind='bar', title='Dataset distribution', legend=False)
    plt.ylabel('tweets')
    plt.show()

def cleaning_stopwords(text, stopwords):
    return " ".join([word for word in str(text).split() if word not in stopwords])

def removing_stopwords(df):
    df['text'] = df['text'].str.lower()
    with open('../datasets/false_stops_es.json','r') as file:
        data = file.read()
    false_stops_es = json.loads(data)
    stopwords = set(false_stops_es)
    df['text'] = df['text'].apply(lambda text: cleaning_stopwords(text, stopwords))
    return df

def cleaning_emojis(df):
    df['text'] = df['text'].apply(lambda text: emoji.demojize(text, language='es'))
    df['text'] = df['text'].apply(lambda text: text.replace('_',' '))
    return df


def cleaning_punctuation(text):
    punctuation_list = string.punctuation
    translator = str.maketrans('', '', punctuation_list)
    return text.translate(translator)


def repeating_characters(text):
    return re.sub(r'(.)1+',r'1', text)

def removing_urls(text):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',text)

def cleaning_numbers(text):
    return re.sub('[0-9]+', '', text)

def processing(df):
    removing_stopwords(df)
    cleaning_emojis(df)
    df['text']= df['text'].apply(lambda x: cleaning_punctuation(x))
    df['text'] = df['text'].apply(lambda x: repeating_characters(x))
    df['text'] = df['text'].apply(lambda x: removing_urls(x))
    df['text'] = df['text'].apply(lambda x: cleaning_numbers(x))
    return df



if __name__ == "__main__":
    names_columns = ['id','label','text']
    train = pd.read_csv('../datasets/training.txt', sep='\t', names = names_columns)
    test = pd.read_csv('../datasets/test.txt', sep='\t', names = names_columns)
    development = pd.read_csv('../datasets/development.txt', sep='\t', names = names_columns)
    full = train.append(development)

    train = True
    if train:
        distribution(full)
        df_clean = processing(full)

        filepath = Path('../datasets/df_clean.csv')
        df_clean.to_csv(filepath)
    else:
        test_clean = processing(test)

        filepath = Path('../datasets/test_clean.csv')
        test_clean.to_csv(filepath)
