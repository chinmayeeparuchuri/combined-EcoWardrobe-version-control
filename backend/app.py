import google.generativeai as genai
from flask import Flask, request, jsonify
from backend.utils.user_input_parser import parse_user_input
from backend.utils.model import (
    load_sustainability_model,
    predict_sustainability_score,
    explain_sustainability_score
)
from backend.utils.ethical_model import (
    load_ethical_model,
    predict_ethics_score,
    explain_ethics_score
)
from utils.gpt_utils import ( 
    parse_input_with_ollama, 
    generate_gpt_explanation,
    generate_summary
)
from backend.utils.gemini_api_helper import ask_gemini_with_database
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
from pathlib import Path

from dotenv import dotenv_values

os.environ["GEMINI_API_KEY"] = "AIzaSyCJxb6111vvR4vsPTXrwaIAD2xnVz3xjgQ"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("Gemini key loaded:", GEMINI_API_KEY)


# Temporary check
print("Gemini key loaded:", GEMINI_API_KEY)

app = Flask(__name__)

# Load models once at startup
sustainability_model, sustainability_features = load_sustainability_model()
ethical_model = load_ethical_model()

@app.route("/gpt/parse", methods=["POST"])
def parse_input_route():  # 🔧 renamed to avoid shadowing
    data = request.get_json()
    user_input = data.get("text")
    if not user_input:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    result = parse_input_with_ollama(user_input)
    return jsonify(result)

@app.route("/gpt/explain", methods=["POST"])
def gpt_explain():
    data = request.json
    fabric = data.get("fabric", "")
    color = data.get("color", "")
    dye = data.get("dye", "")
    manufacturer = data.get("manufacturer", "")
    sustainability_score = data.get("sustainability_score", 0)
    ethical_score = data.get("ethical_score", 0)
    sustainability_explanation = data.get("sustainability_explanation", "")
    ethical_explanation = data.get("ethical_explanation", "")

    explanation = generate_gpt_explanation(
        fabric=fabric,
        color=color,
        dye=dye,
        manufacturer=manufacturer,
        sustainability_score=sustainability_score,
        ethical_score=ethical_score,
        sustainability_explanation=sustainability_explanation,
        ethical_explanation=ethical_explanation
    )
    return jsonify({"gpt_explanation": explanation})

@app.route("/sustainability_score", methods=["POST"])
def sustainability_score():
    fabric = request.json.get("fabric", "")
    color = request.json.get("color", "")
    dye = request.json.get("dye", None)
    result = predict_sustainability_score(
        fabric=fabric,
        color=color,
        dye=dye,
        model=sustainability_model
    )
    return jsonify(result)

@app.route("/ethical_score", methods=["POST"])
def ethical_score():
    manufacturer = request.json.get("manufacturer", "")
    result = predict_ethics_score(manufacturer, ethical_model)
    return jsonify(result)

@app.route("/explanation", methods=["POST"])
def explanation():
    fabric = request.json.get("fabric", "")
    color = request.json.get("color", "")
    dye = request.json.get("dye", None)
    manufacturer = request.json.get("manufacturer", "")

    sustain_exp = explain_sustainability_score(
        fabric=fabric,
        color=color,
        dye=dye,
        model=sustainability_model
    )
    ethical_exp = explain_ethics_score(manufacturer, ethical_model)

    return jsonify({
        "sustainability_explanation": sustain_exp,
        "ethical_explanation": ethical_exp
    })

@app.route("/full_report", methods=["POST"])
def full_report():
    data = request.get_json()
    text = data.get("text", "")
    parsed = parse_user_input(text)

    fabric = parsed.get("fabric")
    color = parsed.get("color")
    manufacturer = parsed.get("manufacturer")
    input_dye = parsed.get("dye")  # raw user dye if provided

    blend = isinstance(fabric, dict)

    # Sustainability
    sustainability = predict_sustainability_score(
        fabric=fabric,
        color=color,
        dye=input_dye,
        model=sustainability_model
    )
    explanation = explain_sustainability_score(
        fabric=fabric,
        color=color,
        dye=input_dye,
        model=sustainability_model
    )

    # Ethical
    ethical = predict_ethics_score(
        manufacturer_name=manufacturer,
        model=ethical_model
    )
    ethical_score = ethical.get("score")
    ethical_source = ethical.get("source", "manufacturer match" if ethical_score is not None else "fallback")

    # 🔁 ✅ Generate the GPT summary
    full_report_data = {
        "parsed_input": {
            "fabric": fabric,
            "color": color,
            "manufacturer": manufacturer,
            "input_dye": input_dye,
            "blend": blend
        },
        "sustainability_score": sustainability.get("score"),
        "sustainability_source": sustainability.get("source"),
        "sustainability_dye_used": sustainability.get("dye_used"),
        "explanation": {
            "source": sustainability.get("source"),
            "dye_used": sustainability.get("dye_used"),
            "explanation": explanation
        },
        "ethical_score": ethical_score,
        "ethical_source": ethical_source
    }

    from utils.gpt_utils import generate_summary  # make sure this is at the top of the file too
    gpt_summary = generate_summary(full_report_data)

    # ✅ Include the summary in the final response
    full_report_data["gpt_summary"] = gpt_summary

    return jsonify(full_report_data)

# Helper function to extract manufacturer names from the query
def extract_manufacturer_name(query):
    """
    A helper function to extract the manufacturer name from the user query.
    This can be improved using NLP techniques or regex if needed.
    """
    if "manufacturer" in query.lower():
        words = query.split(" ")
        for word in words:
            if word.istitle():  # Placeholder logic for capitalized words (manufacturer names)
                return word
    return None

@app.route("/ask_gpt", methods=["POST"])
def ask_gpt():
    try:
        user_query = request.json.get("query", "")
        print(f"Received query: {user_query}")  # Debug print

        if not user_query:
            return jsonify({"error": "Query is empty"}), 400

        manufacturer_name = extract_manufacturer_name(user_query)

        # Either way, use ask_gemini_with_database
        response_text = ask_gemini_with_database(user_query)

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": f"Gemini API error: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def home():
    return "EcoWardrobe API is running 🚀"

if __name__ == "__main__":
    app.run(debug=True, port=5050)
