import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def working_dataset(df):
    X_tokens = [' '.join(token) for token in list(df['tokens'])]
    X_lemmas = [' '.join(token) for token in list(df['lemmas'])]
    X_stems = [' '.join(token) for token in list(df['stems'])]

    d = {'X_tokens': X_tokens, 'X_lemmas': X_lemmas, 'X_stems':X_stems, 'label': df['label']}
    df = pd.DataFrame(data=d)
    return df

def resultsLogReg(X_train, X_test, y_train, y_test):
    vectoriser.fit(X_train)
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)
    logReg = LogisticRegression(C=4).fit(X_train, y_train)
    y_pred = logReg.predict(X_test)
    print("--------------------")
    print("Classification Report for Logistic Regression")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("--------------------")
    print("Accuracy for Logistic Regression:", accuracy)
    return y_pred, accuracy

def resultsSVC(X_train, X_test, y_train, y_test):
    vectoriser.fit(X_train)
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)
    svc = LinearSVC().fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print("--------------------")
    print("Classification Report for SVC")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("--------------------")
    print("Accuracy for SVC:", accuracy)
    return y_pred, accuracy

def resultsRF(X_train, X_test, y_train, y_test):
    vectoriser.fit(X_train)
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)
    clf = RandomForestClassifier(random_state=0)
    rf = clf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("--------------------")
    print("Classification Report for RF")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("--------------------")
    print("Accuracy for RF:", accuracy)
    return y_pred, accuracy

if __name__ == "__main__":
    df_clean = pd.read_csv('../datasets/df_tokenized.csv',converters={'tokens': literal_eval, 'lemmas': literal_eval, 'stems':literal_eval})

    working_df = working_dataset(df_clean)
    X_train_tokens, X_test_tokens, y_train_tokens, y_test_tokens = train_test_split(working_df['X_tokens'], working_df['label'],test_size = 0.2, random_state =0)
    X_train_lemmas, X_test_lemmas, y_train_lemmas, y_test_lemmas = train_test_split(working_df['X_lemmas'], working_df['label'],test_size = 0.2, random_state =0)
    X_train_stems, X_test_stems, y_train_stems, y_test_stems = train_test_split(working_df['X_stems'], working_df['label'],test_size = 0.2, random_state =0)

    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    print("Results for tokens")
    y_pred_tokens_log, accuracy_tokens_log = resultsLogReg(X_train_tokens, X_test_tokens, y_train_tokens, y_test_tokens)
    y_pred_tokens_SVC, accuracy_tokens_SVC = resultsSVC(X_train_tokens, X_test_tokens, y_train_tokens, y_test_tokens)
    y_pred_tokens_rf, accuracy_tokens_rf = resultsRF(X_train_tokens, X_test_tokens, y_train_tokens, y_test_tokens)
    print("Results for stems")
    y_pred_stems_log, accuracy_stems_log = resultsLogReg(X_train_stems, X_test_stems, y_train_stems, y_test_stems)
    y_pred_stems_SVC, accuracy_stems_SVC = resultsSVC(X_train_stems, X_test_stems, y_train_stems, y_test_stems)
    y_pred_stems_rf, accuracy_stems_rf = resultsRF(X_train_stems, X_test_stems, y_train_stems, y_test_stems)
    print("Results for lemmas")
    y_pred_lemmas_log, accuracy_lemmas_log = resultsLogReg(X_train_lemmas, X_test_lemmas, y_train_lemmas, y_test_lemmas)
    y_pred_lemmas_SVC, accuracy_lemmas_SVC = resultsSVC(X_train_lemmas, X_test_lemmas, y_train_lemmas, y_test_lemmas)
    y_pred_lemmas_rf, accuracy_lemmas_rf = resultsRF(X_train_lemmas, X_test_lemmas, y_train_lemmas, y_test_lemmas)
