import typer, joblib
from pathlib import Path
from .features import detect_language, features_from_python, features_from_cpp

app = typer.Typer()

@app.command()
def analyze(path: Path, model: Path = Path('src/autograder/model.joblib')):
    if not path.exists():
        print('file not found', path); raise SystemExit(1)
    code = path.read_text(encoding='utf-8', errors='ignore')
    lang = detect_language(path.name)
    if lang == 'python':
        feats = features_from_python(code)
    elif lang == 'cpp':
        feats = features_from_cpp(code)
    else:
        print('unsupported file extension'); raise SystemExit(1)
    if not model.exists():
        print('model not found. run: python -m src.autograder.train'); raise SystemExit(1)
    data = joblib.load(model)
    clf = data['model']; cols = data['columns']
    X = [feats.get(c,0) for c in cols]
    pred = clf.predict([X])[0]
    probs = clf.predict_proba([X])[0].tolist() if hasattr(clf,'predict_proba') else None
    print('Features:', feats)
    print('Prediction:', 'PASS' if pred==1 else 'FAIL', 'probabilities=', probs)

if __name__ == '__main__':
    app()
