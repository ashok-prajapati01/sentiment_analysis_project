
import os
import pickle
import joblib
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model and vectorizer with fallback between joblib and pickle
def load_artifact(filename):
    if os.path.exists(filename):
        try:
            return joblib.load(filename)
        except Exception:
            with open(filename, 'rb') as f:
                return pickle.load(f)
    else:
        print(f"Warning: {filename} not found.")
        return None

model = load_artifact('model.pkl')
vectorizer = load_artifact('vectorizer.pkl')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f0c20 0%, #151030 50%, #090514 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            --accent-cyan: #00f2fe;
            --accent-purple: #9d4edd;
            --accent-pink: #ff2a85;
            --text-main: #f8f9fa;
            --text-sub: #a0a5b5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: var(--text-main);
            overflow-x: hidden;
            position: relative;
        }

        /* Ambient Glowing Orbs */
        body::before, body::after {
            content: '';
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            z-index: 0;
            animation: pulse 8s ease-in-out infinite alternate;
        }
        body::before {
            width: 350px;
            height: 350px;
            background: rgba(157, 78, 221, 0.35);
            top: 10%;
            left: 15%;
        }
        body::after {
            width: 400px;
            height: 400px;
            background: rgba(0, 242, 254, 0.25);
            bottom: 10%;
            right: 15%;
        }

        @keyframes pulse {
            0% { transform: scale(1) translate(0, 0); }
            100% { transform: scale(1.15) translate(20px, -20px); }
        }

        .container {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 650px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: var(--glass-shadow);
            animation: slideUp 0.8s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        header {
            text-align: center;
            margin-bottom: 32px;
        }

        .badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 50px;
            background: rgba(0, 242, 254, 0.1);
            border: 1px solid rgba(0, 242, 254, 0.3);
            color: var(--accent-cyan);
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #a0a5b5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        p.subtitle {
            color: var(--text-sub);
            font-size: 0.95rem;
        }

        .input-group {
            position: relative;
            margin-bottom: 24px;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 18px;
            color: var(--text-main);
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.25);
            background: rgba(0, 0, 0, 0.35);
        }

        textarea::placeholder {
            color: rgba(255, 255, 255, 0.3);
        }

        .btn-submit {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-cyan) 100%);
            color: #fff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(157, 78, 221, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 242, 254, 0.5);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Card */
        .result-box {
            margin-top: 28px;
            padding: 24px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--glass-border);
            display: none;
            opacity: 0;
            transition: all 0.4s ease;
            text-align: center;
        }

        .result-box.show {
            display: block;
            opacity: 1;
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        .sentiment-title {
            font-size: 0.85rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .sentiment-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .sentiment-positive {
            color: #00ffaa;
            text-shadow: 0 0 12px rgba(0, 255, 170, 0.4);
        }

        .sentiment-negative {
            color: var(--accent-pink);
            text-shadow: 0 0 12px rgba(255, 42, 133, 0.4);
        }

        .sentiment-neutral {
            color: #ffb703;
            text-shadow: 0 0 12px rgba(255, 183, 3, 0.4);
        }

        .confidence-bar-bg {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }

        .confidence-bar-fill {
            height: 100%;
            width: 0%;
            border-radius: 10px;
            transition: width 0.8s ease-out;
        }

        .loader {
            width: 22px;
            height: 22px;
            border: 3px solid rgba(255,255,255,0.3);
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

    <div class="container">
        <header>
            <span class="badge">NLP Powered</span>
            <h1>Sentiment Analysis</h1>
            <p class="subtitle">Enter any review or phrase to analyze its underlying emotion.</p>
        </header>

        <form id="analyzerForm">
            <div class="input-group">
                <textarea id="inputText" placeholder="Type or paste text here (e.g., 'This product exceeded all my expectations!')" required></textarea>
            </div>
            <button type="submit" class="btn-submit" id="submitBtn">
                <span id="btnText">Analyze Sentiment</span>
                <div class="loader" id="btnLoader"></div>
            </button>
        </form>

        <div class="result-box" id="resultBox">
            <div class="sentiment-title">Predicted Sentiment</div>
            <div class="sentiment-value" id="sentimentResult">-</div>
            <div id="confidenceContainer" style="display: none;">
                <span style="font-size: 0.85rem; color: var(--text-sub);" id="confidenceText">Score: 0%</span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" id="confidenceFill"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('analyzerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = document.getElementById('inputText').value.trim();
            if (!text) return;

            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnLoader = document.getElementById('btnLoader');
            const resultBox = document.getElementById('resultBox');
            const sentimentResult = document.getElementById('sentimentResult');
            const confidenceContainer = document.getElementById('confidenceContainer');
            const confidenceText = document.getElementById('confidenceText');
            const confidenceFill = document.getElementById('confidenceFill');

            // UI Loading state
            btnText.innerText = "Analyzing...";
            btnLoader.style.display = "block";
            submitBtn.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();

                if (data.error) {
                    alert(data.error);
                } else {
                    const sentiment = data.sentiment.toString().toLowerCase();
                    sentimentResult.innerText = data.sentiment;
                    sentimentResult.className = 'sentiment-value';

                    let fillColor = '#00f2fe';
                    if (sentiment.includes('pos') || sentiment === '1') {
                        sentimentResult.classList.add('sentiment-positive');
                        fillColor = '#00ffaa';
                    } else if (sentiment.includes('neg') || sentiment === '0' || sentiment === '-1') {
                        sentimentResult.classList.add('sentiment-negative');
                        fillColor = '#ff2a85';
                    } else {
                        sentimentResult.classList.add('sentiment-neutral');
                        fillColor = '#ffb703';
                    }

                    if (data.probability !== null && data.probability !== undefined) {
                        const probPercent = Math.round(data.probability * 100);
                        confidenceContainer.style.display = 'block';
                        confidenceText.innerText = `Confidence Score: ${probPercent}%`;
                        confidenceFill.style.width = `${probPercent}%`;
                        confidenceFill.style.background = fillColor;
                    } else {
                        confidenceContainer.style.display = 'none';
                    }

                    resultBox.classList.add('show');
                }
            } catch (err) {
                alert("Failed to analyze text. Please check server logs.");
            } finally {
                btnText.innerText = "Analyze Sentiment";
                btnLoader.style.display = "none";
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model or Vectorizer file missing on server.'}), 500

    data = request.get_json()
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({'error': 'Empty input text provided.'}), 400

    # Vectorize input text
    vectorized_text = vectorizer.transform([user_text])
    
    # Generate prediction
    prediction = model.predict(vectorized_text)[0]

    # Calculate probability if model supports predict_proba
    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized_text)[0]
        probability = float(max(probs))

    # Format sentiment output string
    sentiment_label = str(prediction)
    if str(prediction) == '1':
        sentiment_label = "Positive"
    elif str(prediction) == '0':
        sentiment_label = "Negative"

    return jsonify({
        'sentiment': sentiment_label,
        'probability': probability
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
