from flask import Flask, request, render_template, jsonify
from .features import detect_language, features_from_python, features_from_cpp
import joblib, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.joblib'
print("Looking for model at:", MODEL_PATH)

app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'))

def load_model():
    if MODEL_PATH.exists():
        data = joblib.load(MODEL_PATH)
        return data['model'], data['columns']
    return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    f = request.files.get('file')
    if not f:
        return jsonify({'error':'no file uploaded'}), 400
    fname = f.filename
    code = f.read().decode('utf-8', errors='ignore')
    lang = detect_language(fname)
    if lang == 'python':
        feats = features_from_python(code)
    elif lang == 'cpp':
        feats = features_from_cpp(code)
    else:
        return jsonify({'error':'unsupported file type'}), 400
    model, cols = load_model()
    if model is None:
        return jsonify({'error':'model not trained. Run train.py to create model.joblib'}), 500
    X = [feats.get(c,0) for c in cols]
    pred = model.predict([X])[0]
    probs = model.predict_proba([X])[0].tolist() if hasattr(model, 'predict_proba') else None
    return render_template('result.html', filename=fname, feats=feats, pred=pred, prob=probs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
