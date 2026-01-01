# Synthetic dataset generator + simple trainer (RandomForest)
import random
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from .features import features_from_python, features_from_cpp

def synth_python_example():
    n_funcs = random.randint(0,5)
    lines = []
    for i in range(n_funcs):
        lines.append(f"def f{i}():\\n    return {i}\\n")
    if random.random() < 0.25:
        lines.append("assert 1==1\\n")
    for _ in range(random.randint(0,3)):
        lines.append("for i in range(3):\\n    pass\\n")
    code = "\\n".join(lines)
    label = 1 if (n_funcs >= 1 or 'assert' in code) else 0
    return code, label

def synth_cpp_example():
    n_funcs = random.randint(0,5)
    lines = ['#include <bits/stdc++.h>'] if random.random() < 0.6 else []
    for i in range(n_funcs):
        lines.append(f"int f{i}(){{ return {i}; }}")
    if random.random() < 0.2:
        lines.append('assert(1);')
    code = "\\n".join(lines)
    label = 1 if n_funcs >= 1 else 0
    return code, label

def generate_dataset(n=800):
    rows = []
    for _ in range(n):
        if random.random() < 0.6:
            code, label = synth_python_example()
            feats = features_from_python(code)
        else:
            code, label = synth_cpp_example()
            feats = features_from_cpp(code)
        feats['label'] = label
        rows.append(feats)
    df = pd.DataFrame(rows).fillna(0)
    return df

def train_and_save(path='model.joblib'):
    df = generate_dataset(1000)
    X = df.drop(columns=['label']).select_dtypes(include=['number'])
    y = df['label']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump({'model': clf, 'columns': X.columns.tolist()}, path)
    print("Saved model to", path)

if __name__ == '__main__':
    train_and_save()
