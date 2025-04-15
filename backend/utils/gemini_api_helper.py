import google.generativeai as genai
import pandas as pd
from backend.utils.preprocess import get_highest_scoring_manufacturers

# Configure Gemini API
genai.configure(api_key="AIzaSyCJxb6111vvR4vsPTXrwaIAD2xnVz3xjgQ")  # Replace with actual API key
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Load and preprocess datasets
df_manufacturers = pd.read_csv("data/processed/clean_manufacturers.csv")
df_dyes = pd.read_csv("data/processed/clean_dyes.csv")

# Calculate a 0–5 ethical score using binary columns for manufacturers
def calculate_score(row):
    return (
        int(row["Is Fair Wages"]) +                # Fair Wages
        int(row["Is Worker Safe"]) +                # Worker Safety
        int(row["No Child Labor"]) +                # No Child Labor
        int(row["Is Worker Satisfied"]) +           # Worker Satisfaction
        int(row["Certifications Received"])        # Certifications
    )

df_manufacturers["Ethical Score"] = df_manufacturers.apply(calculate_score, axis=1)

# Helper function to get the top ethical manufacturers
def get_top_manufacturers():
    top = get_highest_scoring_manufacturers()
    response = "🏆 **Top Ethical Manufacturers:**\n"
    for name, score in top:
        response += f"- {name} — Score: {score}/5\n"
    return response

# Handle manufacturer-specific queries
# Helper function to handle the manufacturer query with corrected score logic
def handle_manufacturer_query(query):
    query_lower = query.lower()

    for manufacturer in df_manufacturers["Manufacturer Name"]:
        if manufacturer.lower() in query_lower:
            row = df_manufacturers[df_manufacturers["Manufacturer Name"].str.lower() == manufacturer.lower()].iloc[0]
            
            # Calculate the score based on the criteria
            score = 0
            reasons = []

            # Check Fair Wages
            if row['Is Fair Wages']:
                score += 1
                fair_wages = "✅"
            else:
                fair_wages = f"❌ ({row['Reason for Unfair Wages']})"
                reasons.append(row['Reason for Unfair Wages'])

            # Check Worker Safety
            if row['Is Worker Safe']:
                score += 1
                worker_safety = "✅"
            else:
                worker_safety = f"❌ ({row['Reason for Lack of Worker Safety']})"
                reasons.append(row['Reason for Lack of Worker Safety'])

            # Check No Child Labor
            if row['No Child Labor']:
                score += 1
                no_child_labor = "✅"
            else:
                no_child_labor = f"❌ ({row['Reason for Child Labor']})"
                reasons.append(row['Reason for Child Labor'])

            # Check Worker Satisfaction
            if row['Is Worker Satisfied']:
                score += 1
                worker_satisfaction = "✅"
            else:
                worker_satisfaction = f"❌ ({row['Reason for Low Worker Satisfaction']})"
                reasons.append(row['Reason for Low Worker Satisfaction'])

            # Check Certifications
            if row['Certifications Received']:
                score += 1
                certified = "✅"
            else:
                certified = f"❌ ({row['Reason for No Certifications']})"
                reasons.append(row['Reason for No Certifications'])

            # Cap score at 5
            score = min(score, 5)

            # Build the summary
            summary = (
                f"🔍 **{manufacturer} Ethical Summary:**\n"
                f"- Fair Wages: {fair_wages}\n"
                f"- Worker Safety: {worker_safety}\n"
                f"- No Child Labor: {no_child_labor}\n"
                f"- Worker Satisfaction: {worker_satisfaction}\n"
                f"- Certified: {certified}\n"
                f"➡️ **Ethical Score:** {score}/5"
            )

            # Add the reasons if the score is less than 5
            if score < 5:
                summary += f"\n**Reasons for lower score:**\n"
                for reason in reasons:
                    summary += f"- {reason}\n"

            return summary
    return None

# Handle dye-specific queries
def handle_dye_query(query):
    query_lower = query.lower()

    for dye in df_dyes["Dye Name"]:
        if dye.lower() in query_lower:
            row = df_dyes[df_dyes["Dye Name"].str.lower() == dye.lower()].iloc[0]
            summary = (
                f"🔍 **{dye} Dye Summary:**\n"
                f"- Dye Type: {row['Dye Type']}\n"
                f"- Water Consumption: {row['Water Consumption']} liters\n"
                f"- Energy Consumption: {row['Energy Consumption']} kWh\n"
                f"- Toxicity Level: {row['Toxicity Level']}\n"
                f"- Biodegradability: {row['Biodegradability']}\n"
                f"- Eco-Toxicity: {row['Eco-Toxicity']}\n"
                f"➡️ **Sustainability Score:** {row['Dye Sustainability Score (0–10)']}/10"
            )
            return summary
    return None

# Handle comparison queries
def handle_comparison_query(query):
    query_lower = query.lower()

    # Normalize query split
    if "vs" in query_lower:
        parts = query_lower.split("vs")
    elif "compare" in query_lower:
        parts = query_lower.replace("compare", "").split("and")
    elif "better than" in query_lower:
        parts = query_lower.split("better than")
    elif "more sustainable than" in query_lower:
        parts = query_lower.split("more sustainable than")
    else:
        parts = []

    if len(parts) == 2:
        left = parts[0].strip().title()
        right = parts[1].strip().title()

        # Ensure both manufacturers exist in the dataset
        if left in df_manufacturers["Manufacturer Name"].values and right in df_manufacturers["Manufacturer Name"].values:
            return compare_manufacturers(left, right)  # Compare both manufacturers
        else:
            return "⚠ One or both manufacturers not found."
    return None


# Comparison logic for manufacturers
def compare_manufacturers(manufacturer_1, manufacturer_2):
    """
    Compares the ethical scores of two manufacturers and returns a summary.
    """
    # Find the rows for both manufacturers
    manufacturer_1_row = df_manufacturers[df_manufacturers["Manufacturer Name"].str.lower() == manufacturer_1.lower()]
    manufacturer_2_row = df_manufacturers[df_manufacturers["Manufacturer Name"].str.lower() == manufacturer_2.lower()]

    if manufacturer_1_row.empty or manufacturer_2_row.empty:
        return "⚠️ One or both manufacturers not found."

    # Get the ethical scores for both
    score_1 = manufacturer_1_row["Ethical Score"].values[0]
    score_2 = manufacturer_2_row["Ethical Score"].values[0]

    # Get the reasons for ❌
    reasons_1 = []
    if not manufacturer_1_row["Is Fair Wages"].values[0]:
        reasons_1.append(manufacturer_1_row["Reason for Unfair Wages"].values[0])
    if not manufacturer_1_row["Is Worker Safe"].values[0]:
        reasons_1.append(manufacturer_1_row["Reason for Lack of Worker Safety"].values[0])
    if not manufacturer_1_row["No Child Labor"].values[0]:
        reasons_1.append(manufacturer_1_row["Reason for Child Labor"].values[0])
    if not manufacturer_1_row["Is Worker Satisfied"].values[0]:
        reasons_1.append(manufacturer_1_row["Reason for Low Worker Satisfaction"].values[0])
    if not manufacturer_1_row["Certifications Received"].values[0]:
        reasons_1.append(manufacturer_1_row["Reason for No Certifications"].values[0])

    reasons_2 = []
    if not manufacturer_2_row["Is Fair Wages"].values[0]:
        reasons_2.append(manufacturer_2_row["Reason for Unfair Wages"].values[0])
    if not manufacturer_2_row["Is Worker Safe"].values[0]:
        reasons_2.append(manufacturer_2_row["Reason for Lack of Worker Safety"].values[0])
    if not manufacturer_2_row["No Child Labor"].values[0]:
        reasons_2.append(manufacturer_2_row["Reason for Child Labor"].values[0])
    if not manufacturer_2_row["Is Worker Satisfied"].values[0]:
        reasons_2.append(manufacturer_2_row["Reason for Low Worker Satisfaction"].values[0])
    if not manufacturer_2_row["Certifications Received"].values[0]:
        reasons_2.append(manufacturer_2_row["Reason for No Certifications"].values[0])

    # Create a comparison summary
    comparison = f"**Comparison: {manufacturer_1} vs {manufacturer_2}**\n"
    comparison += f"- {manufacturer_1} Ethical Score: {score_1}/5\n"
    comparison += f"- {manufacturer_2} Ethical Score: {score_2}/5\n"

    # Add reasons for lower scores
    if reasons_1:
        comparison += f"\n**Reasons for {manufacturer_1}'s lower score:**\n"
        for reason in reasons_1:
            comparison += f"- {reason}\n"
    
    if reasons_2:
        comparison += f"\n**Reasons for {manufacturer_2}'s lower score:**\n"
        for reason in reasons_2:
            comparison += f"- {reason}\n"

    if score_1 > score_2:
        comparison += f"🏆 {manufacturer_1} is more ethical."
    elif score_1 < score_2:
        comparison += f"🏆 {manufacturer_2} is more ethical."
    else:
        comparison += "It's a tie! Both manufacturers have the same ethical score."

    return comparison


# Function to compare two dyes based on sustainability
def compare_dyes(dye_1, dye_2):
    """
    Compares the sustainability scores of two dyes and returns a summary.
    """
    # Find the rows for both dyes
    dye_1_row = df_dyes[df_dyes["Dye Name"].str.lower() == dye_1.lower()]
    dye_2_row = df_dyes[df_dyes["Dye Name"].str.lower() == dye_2.lower()]

    if dye_1_row.empty or dye_2_row.empty:
        return "⚠️ One or both dyes not found."

    # Get the sustainability scores and other info for both
    sustainability_score_1 = dye_1_row["Dye Sustainability Score (0–10)"].values[0]
    sustainability_score_2 = dye_2_row["Dye Sustainability Score (0–10)"].values[0]

    # Get reasons for sustainability scores
    reasons_1 = []
    reasons_2 = []

    # Check for factors for dye 1
    if dye_1_row["Water Consumption"].values[0] > 1000:  # Hypothetical threshold
        reasons_1.append("High water consumption.")
    if dye_1_row["Energy Consumption"].values[0] > 500:  # Hypothetical threshold
        reasons_1.append("High energy consumption.")
    if dye_1_row["Toxicity Level"].values[0] > 7:  # Hypothetical threshold
        reasons_1.append("High toxicity level.")
    if dye_1_row["Biodegradability"].values[0] < 4:  # Hypothetical threshold
        reasons_1.append("Low biodegradability.")

    # Check for factors for dye 2
    if dye_2_row["Water Consumption"].values[0] > 1000:
        reasons_2.append("High water consumption.")
    if dye_2_row["Energy Consumption"].values[0] > 500:
        reasons_2.append("High energy consumption.")
    if dye_2_row["Toxicity Level"].values[0] > 7:
        reasons_2.append("High toxicity level.")
    if dye_2_row["Biodegradability"].values[0] < 4:
        reasons_2.append("Low biodegradability.")

    # Create a comparison summary for the dyes
    comparison = f"**Comparison: {dye_1} vs {dye_2}**\n"
    comparison += f"- {dye_1} Sustainability Score: {sustainability_score_1}/10\n"
    comparison += f"- {dye_2} Sustainability Score: {sustainability_score_2}/10\n"

    # Add reasons for sustainability scores
    if reasons_1:
        comparison += f"\n**Reasons for {dye_1}'s lower sustainability score:**\n"
        for reason in reasons_1:
            comparison += f"- {reason}\n"
    
    if reasons_2:
        comparison += f"\n**Reasons for {dye_2}'s lower sustainability score:**\n"
        for reason in reasons_2:
            comparison += f"- {reason}\n"

    if sustainability_score_1 > sustainability_score_2:
        comparison += f"🏆 {dye_1} is more sustainable."
    elif sustainability_score_1 < sustainability_score_2:
        comparison += f"🏆 {dye_2} is more sustainable."
    else:
        comparison += "It's a tie! Both dyes have the same sustainability score."

    return comparison

# Handle dye comparison queries (e.g., "compare indigo dye and direct blue dye")
def handle_dye_comparison_query(query):
    query_lower = query.lower()

    # Check if the query is asking for a comparison between two dyes
    if "compare" in query_lower or "vs" in query_lower:
        # Split query into two parts (e.g., "compare indigo and direct blue")
        parts = query_lower.split("vs") if "vs" in query_lower else query_lower.replace("compare", "").split("and")

        if len(parts) == 2:
            dye_1 = parts[0].strip()
            dye_2 = parts[1].strip()

            # Ensure both dyes exist in the dataset
            if dye_1 in df_dyes["Dye Name"].values and dye_2 in df_dyes["Dye Name"].values:
                return compare_dyes(dye_1, dye_2)
            else:
                return "⚠ One or both dyes not found."
    return None

# Modify the main function to handle dye comparison queries
def ask_gemini(prompt: str) -> str:
    query_lower = prompt.lower()

    # Handle dye comparison queries (e.g., "compare indigo dye vs direct blue dye")
    dye_comparison_result = handle_dye_comparison_query(prompt)
    if dye_comparison_result:
        return dye_comparison_result

    # Handle other queries as before
    manufacturer_summary = handle_manufacturer_query(prompt)
    if manufacturer_summary:
        return manufacturer_summary

    # Handle comparison queries (e.g., "Is Nike better than Adidas?")
    comparison_result = handle_comparison_query(prompt)
    if comparison_result:
        return comparison_result

    # Fallback to Gemini if no match found in dataset
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini error: {str(e)}"

# New function that combines database queries and Gemini fallback
def ask_gemini_with_database(query: str):
    query_lower = query.lower()

    # If user is asking for highest scoring manufacturer
    if "highest ethical score" in query_lower or "best ethical score" in query_lower:
        return get_top_manufacturers()

    # Otherwise, try to handle it with known database queries (manufacturer, dye, comparison, etc.)
    manufacturer_summary = handle_manufacturer_query(query)
    if manufacturer_summary:
        return manufacturer_summary

    # Handle comparison queries (e.g., "compare Nicobar Island Apparel Crafts and Guntur Textile Mills")
    comparison_result = handle_comparison_query(query)
    if comparison_result:
        return comparison_result

    # Fallback to general Gemini response
    return ask_gemini(query)