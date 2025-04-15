import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="EcoWardrobe", page_icon="🌿")

st.title("🌿 EcoWardrobe")

# --- Input Section ---
st.header("👕 Describe Your Clothing")
st.write("Enter a clothing description to analyze its sustainability and ethical impact.")

user_input = st.text_area(
    "Clothing Description",
    placeholder="e.g., red organic cotton shirt from Andaman Apparel Center"
)

submitted = st.button("Analyze")

if submitted:
    if user_input.strip() == "":
        st.warning("Please enter some description.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:5050/full_report",
                    json={"text": user_input}
                )
                if response.status_code == 200:
                    st.session_state["full_report"] = response.json()
                    st.success("✅ Analysis complete!")
                else:
                    st.error("❌ Backend error. Please ensure the API is running.")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")

# --- Results Section ---
if "full_report" in st.session_state:
    report = st.session_state["full_report"]

    parsed = report["parsed_input"]
    sustainability_score = report["sustainability_score"]
    ethical_score = report["ethical_score"]
    gpt_summary = report.get("gpt_summary", "No summary generated.")
    explanation = report.get("explanation", {})

    st.header("📊 Analysis Results")

    # Parsed Input
    st.subheader("🔍 Parsed Input")
    st.json(parsed)

    # Score Gauges
    def gauge(title, value, max_value=10):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={'axis': {'range': [0, max_value]},
                   'bar': {'color': "green" if value > 7 else "orange" if value > 4 else "red"}}
        ))
        st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        if sustainability_score is not None:
            gauge("Sustainability Score", round(sustainability_score, 2))
        else:
            st.error("⚠️ Could not compute sustainability score.")
    with col2:
        if ethical_score is not None:
            gauge("Ethical Score", round(ethical_score, 2))
        else:
            st.error("⚠️ Could not compute ethical score.")

    # Explanations
    st.subheader("📘 AI-Powered Explanation")
    st.write(gpt_summary)

    # SHAP Feature Contributions
    st.subheader("🔬 SHAP Feature Contributions")
    shap_data = explanation.get("explanation", [])
    if isinstance(shap_data, dict): 
        shap_data = shap_data.get("explanation", [])
    if shap_data and isinstance(shap_data[0], dict):
        # If backend returned structured dicts
        shap_fig = go.Figure()
        for item in shap_data:
            shap_fig.add_trace(go.Bar(
                x=[item["contribution"]],
                y=[item["feature"]],
                orientation="h",
                name=item["feature"]
            ))
        shap_fig.update_layout(title="Sustainability SHAP Values", showlegend=False)
        st.plotly_chart(shap_fig)
    elif shap_data and isinstance(shap_data[0], str):
        # Fallback if backend returned strings like "Biodegradability_y: +0.31"
        shap_fig = go.Figure()
        for item in shap_data:
            try:
                feature, value = item.split(":")
                shap_fig.add_trace(go.Bar(
                    x=[float(value.strip())],
                    y=[feature.strip()],
                    orientation="h",
                    name=feature.strip()
                ))
            except:
                continue
        shap_fig.update_layout(title="Sustainability SHAP Values", showlegend=False)
        st.plotly_chart(shap_fig)
    else:
        st.info("No SHAP explanation available.")
        
# --- Gemini Q&A Section ---
st.header("🤖 Ask EcoWardrobe AI (Gemini)")
user_question = st.text_input("Enter your question here", placeholder="e.g., What are the benefits of sustainable fabrics?")

ask_question_button = st.button("Ask Gemini")

# Handle the Q&A request
if ask_question_button:
    if user_question.strip() == "":
        st.warning("Please enter a question to ask.")
    else:
        with st.spinner("Getting response from Gemini..."):
            try:
                # Send the user question to the Flask backend for Gemini response
                gpt_response = requests.post(
                    "http://127.0.0.1:5050/ask_gpt",
                    json={"query": user_question}
                )
                if gpt_response.status_code == 200:
                    gpt_answer = gpt_response.json().get("response", "")
                    st.success("✅ Gemini says:")
                    st.write(gpt_answer)
                else:
                    st.error("❌ AI query failed. Please check the backend.")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")
