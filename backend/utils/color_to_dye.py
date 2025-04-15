import pandas as pd

# Load dyes from training data
df = pd.read_csv(
    "/Users/chinmayee/Documents/ecowardrobe-project/data/processed/fabric_dye_training_data.csv",
    low_memory=False
)
df.drop(columns=["Natural Dyes vs Synthetic Blending Sensitivity"], inplace=True, errors="ignore")

valid_dyes = set(df["Dye Name"].dropna().str.strip().str.lower().unique())

# Raw mapping
raw_color_to_dye = {
    "blue": ["Reactive Blue 19", "Indigo", "Direct Blue 86", "Acid Blue 25", "Disperse Blue 3"],
    "red": ["Reactive Red 120", "Natural Madder", "Direct Red 81", "Acid Red 27", "Disperse Red 60"],
    "yellow": ["Reactive Yellow 145", "Turmeric", "Saffron", "Direct Yellow 86"],
    "green": ["Reactive Green 19", "Acid Green 25", "Weld"],
    "black": ["Reactive Black 5", "Logwood"],
    "orange": ["Reactive Orange 16", "Henna", "Onion Skin", "Disperse Orange 3"],
    "brown": ["Coffee Dye", "Black Tea", "Cutch", "Walnut Hull", "Eucalyptus Bark", "Pomegranate Peel", "Sumac"],
    "purple": ["Reactive Magenta", "Cochineal", "Mordant Blue 7", "Disperse Violet 17"],
    "pink": ["Brazilwood", "Avocado Pit Dye"],
    "grey": ["Onion Skin", "Oak Gall"],
    "beige": ["Pomegranate Peel", "Onion Skin", "Sumac", "Eucalyptus Bark"], 
}

# Filter only valid dyes
color_to_dye = {
    color: [dye for dye in dyes if dye.lower() in valid_dyes]
    for color, dyes in raw_color_to_dye.items()
}

def get_dye_from_color(input_str: str) -> list[str]:
    """Returns valid dyes for a given color name or a specific dye name directly if it's known."""
    if not input_str:
        return []

    input_str = input_str.strip().lower()

    # If input is a known dye, return it directly
    if input_str in valid_dyes:
        return [input_str]

    # Otherwise treat input as a color
    return color_to_dye.get(input_str, [])

