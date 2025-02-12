from flask import Flask, request, jsonify
from draft_models.toxicity_analysis import analyze_toxicity

app = Flask(__name__)


@app.route("/analyze", methods=["GET"])
def analyze():
    data = request.get_json()
    youtube_link = data.get("youtube_link")
    num_comments = data.get("num_comments")
    if not youtube_link:
        return jsonify({"error": "YouTube link is required"}), 400
    if not num_comments:
        print("'num_comments' is not provided, using default value of 100")

    result = analyze_toxicity(youtube_link, num_comments)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
