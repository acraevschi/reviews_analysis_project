from flask import Flask, request, jsonify
from toxicity_analysis import analyze_toxicity

LINK = "https://www.youtube.com/watch?v=GY7HTeTWleY"
NUM_COMMENTS = 100

app = Flask(__name__)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    youtube_link = data.get("youtube_link")

    if not youtube_link:
        return jsonify({"error": "YouTube link is required"}), 400

    result = analyze_toxicity(youtube_link)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
