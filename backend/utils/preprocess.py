import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

LEVEL_MAPPING = {'low': 0, 'medium': 1, 'med': 1, 'high': 2}
BOOL_MAPPING = {'yes': True, 'no': False, 'true': True, 'false': False}

def encode_column(df, column, mapping):
    if column not in df.columns:
        print(f"⚠️ Warning: Column '{column}' not found. Available columns: {list(df.columns)}")
        return df

    col_series = df[column].astype(str).str.lower().str.strip()
    mapped = col_series.map(mapping)

    unmapped = col_series[~col_series.isin(mapping.keys())]
    if not unmapped.empty:
        print(f"⚠️ Unmapped values in '{column}': {unmapped.unique()} — replacing with NaN")

    df[column] = pd.to_numeric(mapped, errors='coerce')
    return df

def normalize_columns(df, columns):
    scaler = MinMaxScaler()
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = scaler.fit_transform(df[[col]])
        else:
            print(f"⚠️ Column '{col}' not found or not numeric — skipping normalization.")
    return df

def calculate_dye_sustainability_score(df):
    df = df.copy()

    cols_to_normalize = ['Water Consumption', 'Energy Consumption', 'Toxicity Level',
                         'Biodegradability', 'Eco-Toxicity']
    df = normalize_columns(df, cols_to_normalize)

    for col in cols_to_normalize:
        if col not in df.columns:
            df[col] = 0

    df['Base Score'] = (
        0.25 * (1 - df['Water Consumption']) +
        0.25 * (1 - df['Energy Consumption']) +
        0.20 * (1 - df['Toxicity Level']) +
        0.15 * df['Biodegradability'] +
        0.15 * (1 - df['Eco-Toxicity'])
    )

    df['Penalty'] = 0
    for keyword, penalty in [('azo', 0.2), ('heavy metal', 0.2), ('carcinogen', 0.3)]:
        df['Penalty'] += df['Common Dye Type'].str.lower().str.contains(keyword, na=False).astype(int) * penalty

    df['Dye Sustainability Score (0–10)'] = ((df['Base Score'] - df['Penalty']).clip(0, 1)) * 10
    df.drop(columns=['Base Score', 'Penalty'], inplace=True)

    return df

def clean_dye_data(path='data/raw/base_dye_data.csv', save_path='data/processed/clean_dyes.csv'):
    df = pd.read_csv(path)

    rename_map = {
        "Water Consumption (L/kg)": "Water Consumption",
        "Energy Consumption (MJ/kg)": "Energy Consumption"
    }
    df.rename(columns=rename_map, inplace=True)

    fix_map = {
        '0': 'low', '0.0': 'low', '0.5': 'medium', '1': 'high', '1.0': 'high',
        'low': 'low', 'medium': 'medium', 'med': 'medium', 'high': 'high',
        'low/medium': 'medium'
    }

    def normalize_fix_level(val):
        try:
            val = str(val).strip().lower()
            if val in fix_map:
                return fix_map[val]
            float_val = float(val)
            if float_val == 0.0:
                return 'low'
            elif float_val == 0.5:
                return 'medium'
            elif float_val == 1.0:
                return 'high'
        except:
            pass
        return val

    for col in ['Toxicity Level', 'Eco-Toxicity']:
        if col in df.columns:
            df[col] = df[col].apply(normalize_fix_level)

    for col in ['Toxicity Level', 'Biodegradability', 'Eco-Toxicity']:
        df = encode_column(df, col, LEVEL_MAPPING)

    df['Common Dye Type'] = df['Common Dye Type'].astype(str).str.lower().str.strip()

    df.drop_duplicates(inplace=True)
    df.dropna(how='all', inplace=True)

    df = calculate_dye_sustainability_score(df)
    df.to_csv(save_path, index=False)
    return df

def clean_fabric_data(paths=['data/raw/enriched_fabric_data.csv', 'data/raw/blended_fabrics.csv'],
                      save_path='data/processed/clean_fabrics.csv'):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        rename_map = {
            'Common Dye Type': 'Dye Type',
            'Water Usage (L/Kg)': 'Water Usage',
            'Water_Usage_L_Kg': 'Water Usage',
            'CO₂ Emissions (Kg/Kg)': 'CO2 Emissions',
            'CO2_Emissions_Kg_Kg': 'CO2 Emissions',
            'Recyclability Score (1–10)': 'Recyclability Score',
            'Recyclability_Score_1_10': 'Recyclability Score',
            'Biodegradability (Years)': 'Biodegradability',
            'Biodegradability_Years': 'Biodegradability',
            'Recyclability Level': 'Recyclability',
            'CO₂ Level': 'CO2 Level',
            'CO2 Level': 'CO2 Level',
            'Water Usage Level': 'Water Usage Level',
        }
        df.rename(columns=rename_map, inplace=True)

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    for col in ['Recyclability', 'Water Usage Level', 'CO2 Level']:
        df = encode_column(df, col, LEVEL_MAPPING)

    df = normalize_columns(df, ['Water Usage', 'CO2 Emissions'])

    df['Dye Type'] = df['Dye Type'].astype(str).str.lower().str.strip()

    df.drop_duplicates(inplace=True)
    df.dropna(how='all', inplace=True)

    df.to_csv(save_path, index=False)
    return df

def clean_manufacturers(path='data/raw/Manufacturers.xlsx', save_path='data/processed/clean_manufacturers.csv'):
    df = pd.read_excel(path)

    for col in ['Is Fair Wages', 'Is Worker Safe', 'No Child Labor', 'Is Worker Satisfied']:
        df = encode_column(df, col, BOOL_MAPPING)

    df.drop_duplicates(inplace=True)
    df.dropna(how='all', inplace=True)

    df.to_csv(save_path, index=False)
    return df

def encode_additional_columns(df):
    column_mappings = {
        'Microplastic Release': BOOL_MAPPING,
        'Heavy Metals': {'yes': 1, 'no': 0, 'possibly': 0.5},
        'Carcinogenic': {'yes': 1, 'no': 0, 'possibly': 0.5},
        'Toxicity Level': LEVEL_MAPPING,
        'Eco-Toxicity': LEVEL_MAPPING,
        'Sustainability Certifications': lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1
    }

    for col, mapping in column_mappings.items():
        if callable(mapping):
            df[col] = df[col].apply(mapping) if col in df.columns else df.get(col)
        else:
            df = encode_column(df, col, mapping)

    return df

def merge_fabric_dye(fabric_df, dye_df, save_path='data/processed/fabric_dye_training_data.csv'):
    fabric_df = fabric_df.copy()
    dye_df = dye_df.copy()

    fabric_df['Dye Type'] = fabric_df['Dye Type'].astype(str).str.lower().str.strip()
    dye_df['Common Dye Type'] = dye_df['Common Dye Type'].astype(str).str.lower().str.strip()

    fabric_df['Dye Type Split'] = fabric_df['Dye Type'].str.replace(r'[\s]*[/,][\s]*', '|', regex=True)
    fabric_df = fabric_df.assign(Dye_Type_Single=fabric_df['Dye Type Split'].str.split('|')).explode('Dye_Type_Single')
    fabric_df['Dye_Type_Single'] = fabric_df['Dye_Type_Single'].str.strip()

    merged = pd.merge(fabric_df, dye_df, how='left', left_on='Dye_Type_Single', right_on='Common Dye Type')

    unmatched = merged[merged['Dye Sustainability Score (0–10)'].isnull()]
    if not unmatched.empty:
        print(f"\n⚠️ {len(unmatched)} dye values unmatched:")
        print(unmatched['Dye_Type_Single'].dropna().unique())

    merged['Dye Sustainability Score (0–10)'] = merged['Dye Sustainability Score (0–10)'].fillna(0)

    if 'Sustainability Score (1–10)' in merged.columns:
        merged['Final Sustainability Score (0–10)'] = (
            0.6 * merged['Sustainability Score (1–10)'].fillna(0) +
            0.4 * merged['Dye Sustainability Score (0–10)']
        ).clip(0, 10)
    else:
        print("⚠️ Fabric sustainability score missing — using only dye score.")
        merged['Final Sustainability Score (0–10)'] = merged['Dye Sustainability Score (0–10)']

    merged = encode_additional_columns(merged)

    merged.to_csv(save_path, index=False)
    return merged

def get_highest_scoring_manufacturers(csv_path="data/processed/clean_manufacturers.csv", top_n=5):
    df = pd.read_csv(csv_path)

    # Your 5-point score logic (binary flags)
    df["Ethical Score"] = (
        df["Is Fair Wages"] +
        df["Is Worker Safe"] +
        df["Is No Child Labor"] +
        df["Is Worker Satisfied"] +
        df["Is Certified"]
    )

    # Sort and get top N
    top_df = df.sort_values(by="Ethical Score", ascending=False).head(top_n)
    return list(zip(top_df["Manufacturer Name"], top_df["Ethical Score"]))

def clean_all():
    print("🧼 Cleaning dye data...")
    dyes = clean_dye_data()

    print("🧵 Cleaning fabric data (including blends)...")
    fabrics = clean_fabric_data()

    print("🏭 Cleaning manufacturer data...")
    manufacturers = clean_manufacturers()

    print("🔗 Merging fabric + dye by Dye Type...")
    merged = merge_fabric_dye(fabrics, dyes)

    print("✅ All data cleaned and saved in data/processed/")
    return dyes, fabrics, manufacturers, merged