import os
import joblib
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    model = None
    vectorizer = None
    print(f"Error loading model or vectorizer: {e}")

# Single-file HTML/CSS/JS with Glassmorphism UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sentiment Analysis</title>

    <!-- Google Fonts & FontAwesome -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --glass-shadow: 0 25px 45px rgba(0, 0, 0, 0.4);
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            min-height: 100vh;
            background: var(--bg-gradient);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            overflow-x: hidden;
            color: var(--text-main);
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Ambient Glow Spheres */
        .ambient-sphere {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            z-index: 0;
            opacity: 0.6;
            animation: float 8s ease-in-out infinite alternate;
        }

        .sphere-1 {
            width: 300px;
            height: 300px;
            background: #a855f7;
            top: 10%;
            left: 15%;
        }

        .sphere-2 {
            width: 350px;
            height: 350px;
            background: #06b6d4;
            bottom: 10%;
            right: 15%;
            animation-delay: -4s;
        }

        @keyframes float {
            0% { transform: translateY(0px) scale(1); }
            100% { transform: translateY(-30px) scale(1.08); }
        }

        /* Glassmorphism Container */
        .glass-card {
            position: relative;
            z-index: 10;
            width: 100%;
            max-width: 650px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: var(--glass-shadow);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5), 0 0 30px rgba(168, 85, 247, 0.15);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Input Area */
        .input-group {
            margin-bottom: 25px;
            position: relative;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 16px;
            color: #ffffff;
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.25);
            background: rgba(15, 23, 42, 0.8);
        }

        textarea::placeholder {
            color: #64748b;
        }

        /* Action Buttons */
        .button-group {
            display: flex;
            gap: 12px;
        }

        .btn {
            flex: 1;
            padding: 14px 24px;
            border: none;
            border-radius: 14px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--accent-purple) 0%, var(--accent-cyan) 100%);
            color: #ffffff;
            box-shadow: 0 8px 20px rgba(168, 85, 247, 0.3);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(6, 182, 212, 0.4);
            filter: brightness(1.1);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .btn-clear {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-muted);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-clear:hover {
            background: rgba(255, 255, 255, 0.15);
            color: #ffffff;
        }

        /* Result Display Container */
        .result-container {
            margin-top: 25px;
            opacity: 0;
            max-height: 0;
            overflow: hidden;
            transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .result-container.show {
            opacity: 1;
            max-height: 200px;
        }

        .result-badge {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(10px);
        }

        .sentiment-info {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .sentiment-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .positive .sentiment-icon {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }

        .negative .sentiment-icon {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .neutral .sentiment-icon {
            background: rgba(234, 179, 8, 0.2);
            color: #facc15;
            border: 1px solid rgba(234, 179, 8, 0.3);
        }

        .sentiment-text h3 {
            font-size: 1.1rem;
            font-weight: 600;
        }

        .sentiment-text p {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #ffffff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 480px) {
            .glass-card { padding: 25px; }
            .button-group { flex-direction: column; }
        }
    </style>
</head>
<body>

    <div class="ambient-sphere sphere-1"></div>
    <div class="ambient-sphere sphere-2"></div>

    <div class="glass-card">
        <div class="header">
            <h1><i class="fa-solid fa-brain-circuit" style="margin-right: 8px; color: var(--accent-cyan);"></i>Sentiment AI</h1>
            <p>Analyze the emotional tone of any input text instantly</p>
        </div>

        <form id="sentimentForm">
            <div class="input-group">
                <textarea id="userInput" placeholder="Type or paste your text here..." required></textarea>
            </div>

            <div class="button-group">
                <button type="button" class="btn btn-clear" id="clearBtn">
                    <i class="fa-solid fa-rotate-left"></i> Clear
                </button>
                <button type="submit" class="btn btn-submit" id="submitBtn">
                    <span id="btnText"><i class="fa-solid fa-wand-magic-sparkles"></i> Analyze Sentiment</span>
                    <div class="spinner" id="btnSpinner"></div>
                </button>
            </div>
        </form>

        <div class="result-container" id="resultContainer">
            <div class="result-badge" id="resultBadge">
                <div class="sentiment-info">
                    <div class="sentiment-icon" id="sentimentIcon">
                        <i class="fa-solid fa-face-smile"></i>
                    </div>
                    <div class="sentiment-text">
                        <h3 id="sentimentTitle">Positive Sentiment</h3>
                        <p id="sentimentDesc">The model predicts positive emotion in this text.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('sentimentForm');
        const userInput = document.getElementById('userInput');
        const clearBtn = document.getElementById('clearBtn');
        const submitBtn = document.getElementById('submitBtn');
        const btnText = document.getElementById('btnText');
        const btnSpinner = document.getElementById('btnSpinner');
        const resultContainer = document.getElementById('resultContainer');
        const resultBadge = document.getElementById('resultBadge');
        const sentimentIcon = document.getElementById('sentimentIcon');
        const sentimentTitle = document.getElementById('sentimentTitle');
        const sentimentDesc = document.getElementById('sentimentDesc');

        clearBtn.addEventListener('click', () => {
            userInput.value = '';
            resultContainer.classList.remove('show');
            userInput.focus();
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;

            // UI Loading state
            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';
            submitBtn.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                if (response.ok) {
                    displayResult(data.prediction);
                } else {
                    alert(data.error || 'Prediction failed');
                }
            } catch (err) {
                alert('An error occurred. Make sure backend is running.');
            } finally {
                btnText.style.display = 'inline-flex';
                btnSpinner.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        function displayResult(pred) {
            const raw = String(pred).toLowerCase();
            resultBadge.className = 'result-badge';

            if (raw.includes('pos') || raw === '1' || raw === 'positive') {
                resultBadge.classList.add('positive');
                sentimentIcon.innerHTML = '<i class="fa-solid fa-face-smile"></i>';
                sentimentTitle.textContent = 'Positive Sentiment';
                sentimentDesc.textContent = 'High level of positive polarity detected.';
            } else if (raw.includes('neg') || raw === '0' || raw === '-1' || raw === 'negative') {
                resultBadge.classList.add('negative');
                sentimentIcon.innerHTML = '<i class="fa-solid fa-face-frown"></i>';
                sentimentTitle.textContent = 'Negative Sentiment';
                sentimentDesc.textContent = 'Negative tone or critical polarity detected.';
            } else {
                resultBadge.classList.add('neutral');
                sentimentIcon.innerHTML = '<i class="fa-solid fa-face-meh"></i>';
                sentimentTitle.textContent = 'Neutral Sentiment';
                sentimentDesc.textContent = 'Balanced or neutral statement format.';
            }

            resultContainer.classList.add('show');
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
    if not model or not vectorizer:
        return jsonify({"error": "Model files (.pkl) are missing or failed to load."}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Vectorize input text and predict
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]

    # Convert numeric outputs if necessary
    return jsonify({"prediction": str(prediction)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
