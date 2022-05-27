from unittest import result
from django.forms import FilePathField
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pathlib import Path

def modelSVC(X_train, y_train, X_test):
    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    vectoriser.fit(X_train)
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)
    svc = LinearSVC().fit(X_train, y_train)
    test_label = svc.predict(X_test)
    return test_label

def working_dataset(df):
    X_tokens = [' '.join(token) for token in list(df['tokens'])]
    X_lemmas = [' '.join(token) for token in list(df['lemmas'])]
    X_stems = [' '.join(token) for token in list(df['stems'])]

    d = {'X_tokens': X_tokens, 'X_lemmas': X_lemmas, 'X_stems':X_stems, 'label': df['label']}
    df = pd.DataFrame(data=d)
    return df

if __name__ == "__main__":
    train = pd.read_csv('../datasets/df_tokenized.csv',converters={'tokens': literal_eval, 'lemmas': literal_eval, 'stems':literal_eval})
    test = pd.read_csv('../datasets/test_tokenized.csv',converters={'tokens': literal_eval, 'lemmas': literal_eval, 'stems':literal_eval})

    test_id = test['id']
    train, test = working_dataset(train), working_dataset(test)

    test_label = modelSVC(train['X_tokens'], train['label'], test['X_tokens'])

    d = {'id': test_id, 'label': test_label}
    results = pd.DataFrame(data=d)

    filepath = Path('../datasets/results.txt')
    results.to_csv(path_or_buf=filepath, sep="\t", index=False, header=False)






