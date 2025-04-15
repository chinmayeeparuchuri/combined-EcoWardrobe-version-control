import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path

# Load fabric and dye references
fabric_df = pd.read_csv(Path("data/processed/clean_fabrics.csv"))
dye_df = pd.read_csv(Path("data/processed/clean_dyes.csv"))

# Get unique fabric and dye names
fabric_names = fabric_df["Fabric Name"].dropna().unique().tolist()
dye_names = dye_df["Dye Name"].dropna().unique().tolist()

def build_mapping_table(inputs, reference_list):
    mapping_results = []
    for user_input in inputs:
        match, score, _ = process.extractOne(user_input, reference_list, scorer=fuzz.WRatio)
        mapping_results.append({"User Input": user_input, "Matched Name": match, "Score": score})
    return pd.DataFrame(mapping_results)

def create_fabric_mapping(user_inputs):
    df = build_mapping_table(user_inputs, fabric_names)
    df.to_csv("data/processed/fabric_name_map.csv", index=False)
    print("✅ Saved fabric_name_map.csv")

def create_dye_mapping(user_inputs):
    df = build_mapping_table(user_inputs, dye_names)
    df.to_csv("data/processed/dye_name_map.csv", index=False)
    print("✅ Saved dye_name_map.csv")

if __name__ == "__main__":
    fabric_inputs = [
        "cottn", "cotin", "ctn", "cotton fabric", "normal cotton", "standard cotton",
        "organic cottn", "organik cotton", "org cotton", "eco cotton", "sustainable cotton",
        "hempp", "hamp", "hem fabric", "natural hemp",
        "linen fabric", "lenin", "linnen", "flax fabric",
        "silkk", "silk cloth", "soft silk", "organic silk",
        "polyester", "poly", "polyster", "synthetic fabric",
        "polycotton", "cotton blend", "cotton polyester mix",
        "recycled cotton", "reused cotton", "rcycled cotton",
        "bamboo fabric", "bambo", "bmbu", "natural bamboo",
        "wool", "sheep wool", "organic wool", "org wool",
        "alpaca wool", "camel hair", "cashmir", "kasmir",
        "vegan leather", "pinatex leather", "cactus leather", "mushroom leather"
    ]

    dye_inputs = [
        "red dye", "reactiv red", "reddy", "dark red", "light red",
        "reactiv blu", "blue dye", "blu dye", "sky blue",
        "indigoo", "indigo dye", "deep blue",
        "acid red", "acidic red", "acid blue", "acidic blue", "acid grn", "acid green",
        "disperse red", "disprse blue",
        "natural dye", "plant-based dye", "madder dye", "turmeric dye", "yellow dye", "yellow colorant",
        "henna dye",
        "black dye", "black color", "reactive blk",
        "purple dye", "violet dye", "dark purple",
        "pink dye", "magenta", "maroon dye", "navy blue",
        "synthetic dye", "azo dye"
    ]

    create_fabric_mapping(fabric_inputs)
    create_dye_mapping(dye_inputs)

