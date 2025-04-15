import requests
import json
import re
import subprocess

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"  # Make sure it's pulled: `ollama pull mistral`

def extract_json(text):
    try:
        json_objects = re.findall(r"\{.*?\}", text, re.DOTALL)
        for obj in json_objects:
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"[JSON EXTRACT ERROR] {e}")
    return None

def parse_input_with_ollama(user_input):
    print(f"[CALLING OLLAMA] {user_input}")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Extract fabric, color, dye (if any), and manufacturer from the clothing description as a JSON object with keys: fabric, color, dye, manufacturer. Return *only* the JSON object."
            },
            {"role": "user", "content": user_input}
        ],
        "stream": True
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        raw_response = ""

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        raw_response += data["message"]["content"]
                except Exception as e:
                    print(f"[STREAM PARSE ERROR] {e}")

        print("[OLLAMA RAW RESPONSE]", raw_response)
        parsed = extract_json(raw_response)

        if parsed:
            return {"parsed": parsed}
        else:
            raise ValueError("Could not extract JSON from response")

    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return {"parsed": "Sorry, I couldn't process your request at the moment."}


def generate_gpt_explanation(fabric, color, dye, manufacturer,
                             sustainability_score, ethical_score,
                             sustainability_explanation, ethical_explanation):
    prompt = (
        f"Fabric: {fabric}, Color: {color}, Dye: {dye}, Manufacturer: {manufacturer}\n"
        f"Sustainability Score: {sustainability_score}\n"
        f"Sustainability Explanation: {sustainability_explanation}\n"
        f"Ethical Score: {ethical_score}\n"
        f"Ethical Explanation: {ethical_explanation}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You're an expert in sustainable fashion. Based on the product details, "
                    "sustainability and ethical scores, and explanations, write a short, helpful natural language summary "
                    "for the user. Be concise, human-friendly, and insightful."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False  # 👈 make sure this is false or omitted
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        text = response.text
        print("[OLLAMA RAW TEXT]", text)

        # Try to extract just the assistant message from the response
        parsed = json.loads(text)
        explanation = parsed.get("message", {}).get("content", "").strip()
        return explanation if explanation else "Sorry, I couldn't generate an explanation at the moment."

    except Exception as e:
        print(f"[GPT EXPLANATION ERROR] {e}")
        return "Sorry, I couldn't generate an explanation at the moment."
    

def generate_summary(report):
    prompt = f"""
You are an expert sustainability analyst. Given the following product analysis, summarize the sustainability and ethical scores in natural language. Mention the fabric, dye, and manufacturer contributions, and explain why the scores are high or low. Give clear, helpful suggestions if needed.

Report:
{json.dumps(report, indent=2)}

Write the summary in 2–3 short paragraphs.
"""

    try:
        result = subprocess.run(
            ['ollama', 'run', 'mistral', prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return f"GPT error: {result.stderr.strip()}"
        return result.stdout.strip()
    
    except Exception as e:
        return f"Exception while calling GPT: {str(e)}"
    