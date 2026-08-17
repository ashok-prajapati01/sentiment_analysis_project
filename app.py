from flask import Flask, request, jsonify, render_template_string
import pickle
import os
import warnings

# ------------------------------------------------------------
# Sentiment Analysis Flask App
# Uses the uploaded model.pkl + vectorizer.pkl directly.
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# The uploaded model/vectorizer were trained with scikit-learn 1.6.1.
warnings.filterwarnings("ignore", category=UserWarning)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


def predict_sentiment(text):
    """Transform raw text with the original TF-IDF vectorizer, then predict."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    text = text.strip()
    if not text:
        raise ValueError("Please enter some text.")

    # IMPORTANT: use the exact vectorizer that was supplied with the model.
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))

    label = str(prediction).strip().lower()

    # Supports the labels in the supplied model: negative / positive.
    if label in {"positive", "pos", "1", "true"}:
        normalized = "positive"
    elif label in {"negative", "neg", "0", "false"}:
        normalized = "negative"
    else:
        normalized = label

    return {
        "sentiment": normalized,
        "confidence": round(confidence * 100, 2) if confidence is not None else None
    }


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SentimentAI • Text Sentiment Analyzer</title>
<meta name="description" content="AI-powered sentiment analysis using TF-IDF and a trained machine-learning model.">
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
        --bg: #070b17;
        --card: rgba(17, 24, 46, .62);
        --card-border: rgba(255,255,255,.12);
        --text: #f7f8ff;
        --muted: #a7afc7;
        --primary: #8b5cf6;
        --secondary: #06b6d4;
        --positive: #22c55e;
        --negative: #ef4444;
    }
    html { scroll-behavior: smooth; }
    body {
        min-height: 100vh;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
        background:
            radial-gradient(circle at 10% 10%, rgba(139,92,246,.25), transparent 30%),
            radial-gradient(circle at 90% 15%, rgba(6,182,212,.20), transparent 28%),
            radial-gradient(circle at 50% 100%, rgba(59,130,246,.16), transparent 35%),
            var(--bg);
        overflow-x: hidden;
    }
    body::before {
        content: "";
        position: fixed; inset: 0; pointer-events: none; z-index: -1;
        background-image: linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: linear-gradient(to bottom, black, transparent 90%);
    }
    .orb {
        position: fixed; border-radius: 50%; filter: blur(4px); opacity: .35;
        pointer-events: none; z-index: -1; animation: float 9s ease-in-out infinite;
    }
    .orb.one { width: 170px; height:170px; background:#8b5cf6; top:18%; left:-70px; }
    .orb.two { width: 130px; height:130px; background:#06b6d4; right:-40px; top:48%; animation-delay:-3s; }
    @keyframes float { 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-28px) scale(1.06)} }

    .wrap { width:min(1080px, 92%); margin:auto; padding:28px 0 55px; }
    nav { display:flex; justify-content:space-between; align-items:center; margin-bottom:58px; }
    .brand { display:flex; align-items:center; gap:11px; font-weight:800; letter-spacing:.2px; }
    .logo {
        width:42px; height:42px; border-radius:13px; display:grid; place-items:center;
        background:linear-gradient(135deg, #8b5cf6, #06b6d4);
        box-shadow:0 10px 35px rgba(139,92,246,.35); font-size:21px;
    }
    .badge { border:1px solid var(--card-border); background:rgba(255,255,255,.05);
        padding:8px 12px; border-radius:999px; color:var(--muted); font-size:12px; backdrop-filter:blur(12px); }

    .hero { text-align:center; margin-bottom:34px; animation:rise .8s ease both; }
    @keyframes rise { from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:translateY(0)} }
    .eyebrow { color:#b9a6ff; text-transform:uppercase; letter-spacing:3px; font-size:11px; font-weight:800; margin-bottom:14px; }
    h1 { font-size:clamp(40px, 7vw, 72px); line-height:.98; letter-spacing:-3px; }
    .gradient { background:linear-gradient(90deg,#fff,#c4b5fd,#67e8f9); -webkit-background-clip:text; color:transparent; }
    .subtitle { max-width:650px; margin:18px auto 0; color:var(--muted); line-height:1.7; font-size:16px; }

    .glass {
        background:linear-gradient(145deg, rgba(255,255,255,.085), rgba(255,255,255,.035));
        border:1px solid var(--card-border); box-shadow:0 25px 70px rgba(0,0,0,.30), inset 0 1px rgba(255,255,255,.07);
        backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px);
    }
    .panel { border-radius:28px; padding:25px; animation:rise .8s .12s ease both; }
    .labelrow { display:flex; justify-content:space-between; margin-bottom:11px; color:#dce1f5; font-weight:700; font-size:14px; }
    textarea {
        width:100%; min-height:190px; resize:vertical; border-radius:19px; padding:20px;
        color:var(--text); background:rgba(3,7,18,.52); border:1px solid rgba(255,255,255,.09);
        outline:none; font:inherit; line-height:1.65; transition:.25s;
    }
    textarea:focus { border-color:rgba(139,92,246,.75); box-shadow:0 0 0 4px rgba(139,92,246,.12); }
    textarea::placeholder { color:#66708c; }
    .actions { display:flex; gap:12px; margin-top:15px; flex-wrap:wrap; }
    button {
        border:0; cursor:pointer; font:inherit; font-weight:800; border-radius:14px; padding:13px 20px;
        transition:.2s; color:#fff;
    }
    .primary { flex:1; min-width:180px; background:linear-gradient(135deg,#8b5cf6,#06b6d4);
        box-shadow:0 12px 30px rgba(99,102,241,.25); }
    .secondary { background:rgba(255,255,255,.07); border:1px solid var(--card-border); color:#d9def1; }
    button:hover { transform:translateY(-2px); filter:brightness(1.08); }
    button:active { transform:translateY(0); }

    .result { margin-top:18px; display:none; border-radius:22px; padding:20px; animation:pop .45s ease; }
    @keyframes pop { from{opacity:0;transform:scale(.98)} to{opacity:1;transform:scale(1)} }
    .result.positive { border-color:rgba(34,197,94,.35); background:rgba(34,197,94,.07); }
    .result.negative { border-color:rgba(239,68,68,.35); background:rgba(239,68,68,.07); }
    .resulthead { display:flex; align-items:center; gap:14px; }
    .emoji { width:52px; height:52px; border-radius:16px; display:grid; place-items:center; font-size:25px; background:rgba(255,255,255,.08); }
    .result h2 { text-transform:capitalize; font-size:25px; }
    .result small { color:var(--muted); }
    .meter { height:8px; border-radius:20px; background:rgba(255,255,255,.08); margin-top:18px; overflow:hidden; }
    .fill { height:100%; width:0; border-radius:20px; background:linear-gradient(90deg,#8b5cf6,#06b6d4); transition:width 1s cubic-bezier(.2,.8,.2,1); }

    .features { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:18px; }
    .feature { padding:19px; border-radius:20px; }
    .feature .icon { font-size:22px; margin-bottom:10px; }
    .feature h3 { font-size:14px; margin-bottom:5px; }
    .feature p { color:var(--muted); font-size:12px; line-height:1.55; }

    .guide { margin-top:22px; padding:24px; border-radius:24px; }
    .guide h2 { margin-bottom:12px; font-size:18px; }
    .steps { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
    .step { border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:15px; background:rgba(0,0,0,.12); }
    .num { color:#a78bfa; font-weight:900; font-size:12px; }
    .step p { color:var(--muted); font-size:12px; line-height:1.55; margin-top:6px; }
    footer { text-align:center; color:#68728d; font-size:12px; margin-top:28px; }

    .loading { display:none; width:16px; height:16px; border:2px solid rgba(255,255,255,.35); border-top-color:#fff; border-radius:50%; animation:spin .7s linear infinite; }
    @keyframes spin { to{transform:rotate(360deg)} }

    @media(max-width:760px) {
        nav { margin-bottom:38px; }
        .badge { display:none; }
        .panel { padding:17px; border-radius:22px; }
        .features,.steps { grid-template-columns:1fr; }
        h1 { letter-spacing:-2px; }
    }
</style>
</head>
<body>
<div class="orb one"></div><div class="orb two"></div>
<div class="wrap">
    <nav>
        <div class="brand"><div class="logo">✦</div><span>SentimentAI</span></div>
        <div class="badge">TF-IDF • Multinomial Naive Bayes</div>
    </nav>

    <section class="hero">
        <div class="eyebrow">Intelligent Text Analysis</div>
        <h1>Understand the <span class="gradient">feeling</span><br>behind every word.</h1>
        <p class="subtitle">Enter a sentence, review or message and let your trained machine-learning model classify it as positive or negative.</p>
    </section>

    <main class="glass panel">
        <div class="labelrow"><span>Your text</span><span id="counter">0 / 2000</span></div>
        <textarea id="text" maxlength="2000" placeholder="Try: “The product is fantastic and I absolutely love the experience!”"></textarea>
        <div class="actions">
            <button class="secondary" id="clearBtn">Clear</button>
            <button class="primary" id="analyzeBtn"><span id="btnText">Analyze Sentiment</span><span class="loading" id="loader"></span></button>
        </div>

        <div id="result" class="glass result">
            <div class="resulthead">
                <div class="emoji" id="emoji">✨</div>
                <div><small>Prediction</small><h2 id="sentiment">—</h2></div>
            </div>
            <div class="meter"><div class="fill" id="fill"></div></div>
            <div style="margin-top:9px;color:#a7afc7;font-size:12px" id="confidence">Confidence: —</div>
        </div>
    </main>

    <section class="features">
        <div class="glass feature"><div class="icon">⚡</div><h3>Instant inference</h3><p>Your text is transformed by the original TF-IDF vectorizer before prediction.</p></div>
        <div class="glass feature"><div class="icon">🧠</div><h3>Your trained model</h3><p>This app loads the supplied Multinomial Naive Bayes model rather than retraining it.</p></div>
        <div class="glass feature"><div class="icon">🔒</div><h3>Simple & lightweight</h3><p>Flask serves one responsive interface and a JSON prediction endpoint.</p></div>
    </section>

    <section class="glass guide">
        <h2>🚀 Render deployment guide</h2>
        <div class="steps">
            <div class="step"><div class="num">01 — FILES</div><p>Keep <b>app.py</b>, <b>model.pkl</b>, <b>vectorizer.pkl</b> and <b>requirements.txt</b> in the same GitHub repository.</p></div>
            <div class="step"><div class="num">02 — REPO</div><p>Push the four files to GitHub. The model filename must be exactly <b>model.pkl</b>.</p></div>
            <div class="step"><div class="num">03 — RENDER</div><p>Create a new Render Web Service, select your repository, and use Python as the runtime.</p></div>
            <div class="step"><div class="num">04 — COMMAND</div><p>Build: <b>pip install -r requirements.txt</b><br>Start: <b>gunicorn app:app</b></p></div>
        </div>
    </section>

    <footer>Built with Flask + scikit-learn • Original TF-IDF vectorizer preserved</footer>
</div>

<script>
const text = document.getElementById("text");
const counter = document.getElementById("counter");
const result = document.getElementById("result");
const analyzeBtn = document.getElementById("analyzeBtn");
const loader = document.getElementById("loader");
const btnText = document.getElementById("btnText");

text.addEventListener("input", () => counter.textContent = `${text.value.length} / 2000`);

document.getElementById("clearBtn").addEventListener("click", () => {
    text.value = "";
    counter.textContent = "0 / 2000";
    result.style.display = "none";
    text.focus();
});

async function analyze() {
    const value = text.value.trim();
    if (!value) {
        text.focus();
        text.style.borderColor = "rgba(239,68,68,.7)";
        setTimeout(() => text.style.borderColor = "", 900);
        return;
    }

    analyzeBtn.disabled = true;
    btnText.style.display = "none";
    loader.style.display = "inline-block";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text: value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed.");

        result.className = `glass result ${data.sentiment}`;
        result.style.display = "block";
        document.getElementById("sentiment").textContent = data.sentiment;
        document.getElementById("emoji").textContent = data.sentiment === "positive" ? "😊" : "☹️";

        const confidence = data.confidence;
        document.getElementById("confidence").textContent =
            confidence !== null ? `Confidence: ${confidence}%` : "Confidence unavailable";

        requestAnimationFrame(() => {
            document.getElementById("fill").style.width = `${confidence ?? 0}%`;
        });
    } catch (err) {
        result.className = "glass result negative";
        result.style.display = "block";
        document.getElementById("emoji").textContent = "⚠️";
        document.getElementById("sentiment").textContent = "Error";
        document.getElementById("confidence").textContent = err.message;
        document.getElementById("fill").style.width = "0%";
    } finally {
        analyzeBtn.disabled = false;
        btnText.style.display = "inline";
        loader.style.display = "none";
    }
}

analyzeBtn.addEventListener("click", analyze);
text.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
});
</script>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HTML)


@app.post("/predict")
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        result = predict_sentiment(text)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Unable to process this text."}), 500


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Local development only. Render uses gunicorn via the start command.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
