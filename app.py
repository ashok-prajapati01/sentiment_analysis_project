import os
import joblib
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load vectorizer and sentiment model
MODEL_PATH = "senti_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = None
vectorizer = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        print("Warning: senti_model.pkl or vectorizer.pkl missing in working directory.")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sentiment Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.06);
            --glass-border: rgba(255, 255, 255, 0.12);
            --glass-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            --accent-cyan: #38bdf8;
            --accent-purple: #c084fc;
            --accent-pink: #f472b6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            min-height: 100vh;
            background: var(--bg-gradient);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            overflow-x: hidden;
            position: relative;
        }

        /* Ambient Animated Blobs */
        .blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.5;
            animation: float 10s infinite alternate ease-in-out;
            z-index: 0;
        }

        .blob-1 {
            width: 320px;
            height: 320px;
            background: #6366f1;
            top: 10%;
            left: 15%;
        }

        .blob-2 {
            width: 380px;
            height: 380px;
            background: #ec4899;
            bottom: 10%;
            right: 15%;
            animation-delay: -5s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(30px, -40px) scale(1.1); }
        }

        /* Glassmorphism Container */
        .glass-card {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 650px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 28px;
            padding: 40px;
            box-shadow: var(--glass-shadow);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple), var(--accent-pink));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .input-group {
            margin-bottom: 25px;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 16px;
            color: #fff;
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.25);
            background: rgba(15, 23, 42, 0.8);
        }

        textarea::placeholder {
            color: #64748b;
        }

        .btn-analyze {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 16px;
            background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
            background-size: 200% 200%;
            color: #fff;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.4s ease;
            box-shadow: 0 10px 25px rgba(168, 85, 247, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        .btn-analyze:hover {
            background-position: right center;
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(168, 85, 247, 0.45);
        }

        .btn-analyze:active {
            transform: translateY(0);
        }

        /* Result Section */
        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            display: none;
            text-align: center;
            animation: slideUp 0.5s ease-out forwards;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .sentiment-badge {
            display: inline-block;
            font-size: 1.6rem;
            font-weight: 700;
            padding: 8px 24px;
            border-radius: 50px;
            margin-bottom: 8px;
            text-transform: capitalize;
        }

        .positive {
            color: #4ade80;
            background: rgba(74, 222, 128, 0.15);
            border: 1px solid rgba(74, 222, 128, 0.3);
            box-shadow: 0 0 20px rgba(74, 222, 128, 0.2);
        }

        .negative {
            color: #f87171;
            background: rgba(248, 113, 113, 0.15);
            border: 1px solid rgba(248, 113, 113, 0.3);
            box-shadow: 0 0 20px rgba(248, 113, 113, 0.2);
        }

        .neutral {
            color: #fbbf24;
            background: rgba(251, 191, 36, 0.15);
            border: 1px solid rgba(251, 191, 36, 0.3);
            box-shadow: 0 0 20px rgba(251, 191, 36, 0.2);
        }

        .confidence {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .spinner {
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>

    <div class="glass-card">
        <div class="header">
            <h1>Sentiment AI Engine</h1>
            <p>Paste text below to extract tone, emotion, and sentiment</p>
        </div>

        <div class="input-group">
            <textarea id="inputText" placeholder="Type or paste review, tweet, or sentence..."></textarea>
        </div>

        <button class="btn-analyze" id="analyzeBtn" onclick="analyzeSentiment()">
            <span id="btnText">Analyze Sentiment</span>
            <div class="spinner" id="btnSpinner"></div>
        </button>

        <div class="result-box" id="resultBox">
            <div class="result-title">Predicted Sentiment</div>
            <div class="sentiment-badge" id="sentimentBadge">-</div>
            <div class="confidence" id="confidenceText"></div>
        </div>
    </div>

    <script>
        async function analyzeSentiment() {
            const text = document.getElementById('inputText').value.trim();
            const resultBox = document.getElementById('resultBox');
            const sentimentBadge = document.getElementById('sentimentBadge');
            const confidenceText = document.getElementById('confidenceText');
            const btnSpinner = document.getElementById('btnSpinner');
            const btnText = document.getElementById('btnText');

            if (!text) {
                alert("Please enter text before analyzing.");
                return;
            }

            // Show Loading State
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';
            resultBox.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                if (response.ok) {
                    const rawSentiment = String(data.sentiment).toLowerCase();
                    sentimentBadge.textContent = data.sentiment;
                    sentimentBadge.className = 'sentiment-badge';

                    if (rawSentiment.includes('pos') || rawSentiment === '1') {
                        sentimentBadge.classList.add('positive');
                    } else if (rawSentiment.includes('neg') || rawSentiment === '0' || rawSentiment === '-1') {
                        sentimentBadge.classList.add('negative');
                    } else {
                        sentimentBadge.classList.add('neutral');
                    }

                    if (data.confidence !== null && data.confidence !== undefined) {
                        confidenceText.textContent = `Confidence Score: ${data.confidence}%`;
                    } else {
                        confidenceText.textContent = '';
                    }

                    resultBox.style.display = 'block';
                } else {
                    alert(data.error || "An error occurred during prediction.");
                }
            } catch (err) {
                alert("Unable to connect to prediction server.");
            } finally {
                btnText.style.display = 'inline';
                btnSpinner.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded on server.'}), 500

    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No input text provided.'}), 400

    try:
        # Transform text via loaded TfidfVectorizer
        vec_input = vectorizer.transform([text])
        prediction = model.predict(vec_input)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vec_input)[0]
            confidence = round(float(max(probabilities)) * 100, 2)

        return jsonify({
            'sentiment': str(prediction),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
