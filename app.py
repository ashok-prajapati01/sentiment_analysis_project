import os
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

model = None
vectorizer = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")

# Single-file HTML template with Glassmorphism, CSS Animations, and UI
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
            --bg-gradient: linear-gradient(135deg, #0d0e15 0%, #1a1c2e 50%, #25123e 100%);
            --glass-bg: rgba(255, 255, 255, 0.06);
            --glass-border: rgba(255, 255, 255, 0.12);
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            --accent-purple: #a855f7;
            --accent-blue: #3b82f6;
            --accent-pink: #ec4899;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
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
            justify-content: center;
            align-items: center;
            padding: 20px;
            overflow-x: hidden;
            color: var(--text-main);
        }

        /* Background Animated Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(90px);
            z-index: 0;
            animation: float 10s ease-in-out infinite alternate;
        }

        .orb-1 {
            width: 320px;
            height: 320px;
            background: rgba(168, 85, 247, 0.3);
            top: 10%;
            left: 15%;
        }

        .orb-2 {
            width: 380px;
            height: 380px;
            background: rgba(59, 130, 246, 0.25);
            bottom: 10%;
            right: 15%;
            animation-delay: -5s;
        }

        @keyframes float {
            0% { transform: translateY(0px) scale(1); }
            100% { transform: translateY(-30px) scale(1.08); }
        }

        /* Main Glass Card */
        .glass-card {
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
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px) scale(0.97); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-sub);
            font-size: 0.95rem;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 16px;
            color: var(--text-main);
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
            background: rgba(15, 23, 42, 0.8);
        }

        textarea::placeholder {
            color: #64748b;
        }

        .btn-submit {
            width: 100%;
            margin-top: 20px;
            padding: 16px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(168, 85, 247, 0.35);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(168, 85, 247, 0.5);
            filter: brightness(1.1);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Section */
        .result-box {
            margin-top: 28px;
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--glass-border);
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-sub);
            margin-bottom: 8px;
        }

        .result-badge {
            display: inline-block;
            font-size: 1.4rem;
            font-weight: 700;
            padding: 8px 24px;
            border-radius: 50px;
            margin-top: 5px;
            text-transform: capitalize;
        }

        .badge-positive {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.3);
            box-shadow: 0 0 15px rgba(74, 222, 128, 0.2);
        }

        .badge-negative {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.3);
            box-shadow: 0 0 15px rgba(248, 113, 113, 0.2);
        }

        .badge-neutral {
            background: rgba(234, 179, 8, 0.15);
            color: #facc15;
            border: 1px solid rgba(250, 204, 21, 0.3);
            box-shadow: 0 0 15px rgba(250, 204, 21, 0.2);
        }

        .error-msg {
            color: #f87171;
            font-size: 0.9rem;
            margin-top: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="glass-card">
        <div class="header">
            <h1>Sentiment Analysis</h1>
            <p>Enter your text below to detect tone and emotional sentiment</p>
        </div>

        <form method="POST" action="/predict">
            <textarea name="text" placeholder="Type or paste your text here..." required>{{ user_text if user_text else '' }}</textarea>
            <button type="submit" class="btn-submit">Analyze Sentiment</button>
        </form>

        {% if result %}
        <div class="result-box">
            <div class="result-title">Detected Sentiment</div>
            {% if 'pos' in result|lower or '1' in result|string %}
                <div class="result-badge badge-positive">Positive 😊</div>
            {% elif 'neg' in result|lower or '0' in result|string %}
                <div class="result-badge badge-negative">Negative 😞</div>
            {% else %}
                <div class="result-badge badge-neutral">{{ result }}</div>
            {% endif %}
        </div>
        {% endif %}

        {% if error %}
        <div class="error-msg">{{ error }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form.get("text", "")
    
    if not user_text.strip():
        return render_template_string(HTML_TEMPLATE, error="Please enter valid text.")

    if model is None or vectorizer is None:
        return render_template_string(
            HTML_TEMPLATE, 
            error="Model files could not be loaded. Ensure model.pkl and vectorizer.pkl exist.",
            user_text=user_text
        )

    try:
        # Transform input text using vectorizer
        transformed_input = vectorizer.transform([user_text])
        # Predict using sentiment model
        prediction = model.predict(transformed_input)[0]
        
        return render_template_string(HTML_TEMPLATE, result=str(prediction), user_text=user_text)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"Prediction Error: {str(e)}", user_text=user_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
