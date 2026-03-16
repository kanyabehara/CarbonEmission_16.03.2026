import pandas as pd
import re
from word2number import w2n
import numpy as np

file = "S&P_Global_S1_Transition_Risk_Data_Assignment_2025-06-03.xlsx"

# -----------------------------
# Load tables
# -----------------------------
t1, t2, t3, t4 = pd.read_excel(
    file,
    sheet_name=["Table 1","Table 2","Table 3","Table 4"]
).values()

# -----------------------------
# Normalization function
# -----------------------------
def normalize(x):
    if pd.isna(x):
        return x

    s = str(x).lower().strip()
    s = s.replace("percent","%").replace(" %","%")

    # handle cases like twenty-twenty
    if "-" in s and any(c.isalpha() for c in s):
        try:
            return int("".join(str(w2n.word_to_num(p)) for p in s.split("-")))
        except:
            pass

    # handle cases like 202five
    m = re.match(r"(\d+)([a-z]+)", s)
    if m:
        try:
            return int(str(m.group(1)) + str(w2n.word_to_num(m.group(2))))
        except:
            pass

    # handle cases like forty%
    try:
        n = w2n.word_to_num(s.replace("%",""))
        return n/100 if "%" in s else n
    except:
        pass
    # final fallback: numeric conversion
    try:
        return float(s)
    except:
        return x

# Duplicate UID resolution

t1_clean = (
    t1.assign(n=t1.notna().sum(axis=1))
      .sort_values("n",ascending=False)
      .drop_duplicates("UID")
      .drop(columns="n")
)


# Build Master sheet from Table 2
master = t2.copy()

# Bring in company-level data from T1: sector, country, year, scopes, production
master = master.merge(
    t1_clean[["UID","GICS Sector","Country","Year",
              "Scope 1 (tonnes)","Scope 2 (tonnes)", "Unit of Production","Production Volume"]],
    on="UID",how="left"
)

# Derive Baseline Emissions from Table 1 based on Type of Target
master["Baseline_Emission"] = np.select(
    [
        master["Type of Target"] == "Scope 1",
        master["Type of Target"] == "Scope 2",
        master["Type of Target"].isin(["Direct","Scope 2 and other"]),
        master["Type of Target"] == "Scope 3"
    ],
    [
        master["Scope 1 (tonnes)"],
        master["Scope 2 (tonnes)"],
        master["Scope 1 (tonnes)"].fillna(0) + master["Scope 2 (tonnes)"].fillna(0),
        np.nan
    ],
    default=np.nan
)
master = master.drop(columns=["Scope 1 (tonnes)", "Scope 2 (tonnes)"])
# AGGREGATE TABLE 3 (ABATEMENT TECHNOLOGY DATA)
# T3 has multiple technologies per Sector+Region.
# Aggregation logic:
#   Capacity → SUM: company can deploy all technologies simultaneously
#   Cost     → MEAN: blended average cost across the technology mix
#   Initial Investment → SUM: total capital required to deploy all technologies
# "Europa" normalised to "EU" to match T1/T4 region values.
cols = ["Abatement Capacity (tCO2e/year)",
        "Abatement Cost (USD/tCO2e/year)","Initial Investment (USD million)",]

t3[cols] = t3[cols].apply(pd.to_numeric, errors="coerce")
t3["Region"] = t3["Region"].str.strip().replace({"Europa": "EU"})
t3_avg = t3.groupby(["Sector","Region"],as_index=False)[cols].agg({
    "Abatement Capacity (tCO2e/year)":"sum",
    "Abatement Cost (USD/tCO2e/year)":"mean",
    "Initial Investment (USD million)":"sum" 
})
# Join T3 abatement data into master on Sector+Region
master = master.merge(
    t3_avg,
    left_on=["GICS Sector","Country"],
    right_on=["Sector","Region"],
    how="left"
)
# normalize only in master
for col in ["Baseline","Endline","Reduction target", "Year"]:
    master[col] = master[col].map(normalize)

# PREPARE EMISSION PATHWAY INPUT
# Copy master as the base for pathway modelling.
# Rename Year → baseline_year to distinguish from Projection Year downstream.
# Cast all numeric model inputs, replacing blank strings with NaN first

master["Year"] = pd.to_numeric(master["Year"].astype(str).str.replace(",", ""), errors="coerce")

Emission_pathway = master.copy()

Emission_pathway = Emission_pathway.rename(
    columns={"Year": "baseline_year"}
)

model_cols = [
    "Baseline_Emission",
    "Production Volume",
    "Endline",
    "Abatement Capacity (tCO2e/year)",
    "baseline_year"
]

Emission_pathway[model_cols] = (
    Emission_pathway[model_cols]
    .replace(r'^\s*$', np.nan, regex=True)
    .apply(pd.to_numeric, errors="coerce")
)
#TARGET EMISSION CALCULATOR
# Computes the emission level the company is targeting at its endline year.
# Two calculation paths based on reduction target type:
# Intensity target (contains "tCO2e"): Target = Production Volume × intensity rate
def compute_target_emission(df):

    reduction_raw = df["Reduction target"].astype(str)

    intensity_mask = reduction_raw.str.contains("tco2e", case=False)

    reduction_val = pd.to_numeric(
        reduction_raw.str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce"
    )

    target = np.where(
        intensity_mask,
        df["Production Volume"] * reduction_val,
        df["Baseline_Emission"] * (1 - reduction_val)
    )
    return target
#EMISSION PATHWAY MODEL (2025 → 2050)
# Vectorised expansion: each company row is repeated once per projection year
def build_emission_pathway_long(df, target_year=2050):

    years = np.arange(2025, target_year + 1)
    df_exp = df.loc[df.index.repeat(len(years))].reset_index(drop=True)
    df_exp["Projection Year"] = np.tile(years, len(df))
  
    year= df_exp["Projection Year"].values
    baseline_year = df_exp["baseline_year"]
    Baseline_Emission = df_exp["Baseline_Emission"]
    endline = df_exp["Endline"].fillna(target_year)
    abatement = df_exp["Abatement Capacity (tCO2e/year)"].fillna(0)
    target_emission   = compute_target_emission(df_exp)
        # Case 1: Endline >= target year
    annual_decline_target = np.where(
        target_year != baseline_year,
        (Baseline_Emission - target_emission)/ (target_year - baseline_year),0
    )
        # Case 2: Endline < target year
    annual_decline_endline = np.where(
        endline != baseline_year,
        (Baseline_Emission - target_emission) / (endline - baseline_year),0
    )    
    path_case1 = Baseline_Emission - annual_decline_target * (year - baseline_year)
    path_to_endline = Baseline_Emission - annual_decline_endline * (year - baseline_year)
    endline_emission = Baseline_Emission - annual_decline_endline * (endline - baseline_year)
    abatement_path = endline_emission - abatement * (year - endline)

    emissions = np.where(endline >= target_year, path_case1,
                             np.where(year <= endline, path_to_endline, abatement_path)
    )

    df_exp["Projected Emission"] = np.maximum(emissions, 0)
    return df_exp

Emission_pathway_long = build_emission_pathway_long(Emission_pathway)

Emission_pathway_long = Emission_pathway_long.sort_values(
    ["UID", "Projection Year"]
).reset_index(drop=True)

Emission_pathway_long["Projection Year"] = Emission_pathway_long["Projection Year"].astype("int64")
#t4["Year"] = t4["Year"].astype(int)
#CARBON PRICE INTERPOLATION (T4)
t4 = t4[["Sector","Region","Year","Scenario","Carbon Price (USD/tCO2e)"]].copy()
t4["Year"] = pd.to_numeric(t4["Year"], errors="coerce")
t4 = t4.drop_duplicates(
    subset=["Sector","Region","Year","Scenario"]
)

t4["Carbon Price (USD/tCO2e)"] = pd.to_numeric(
    t4["Carbon Price (USD/tCO2e)"], errors="coerce"
)
#Build a full year spine 2025–2050 for every Sector/Region/Scenario combination
all_years = pd.DataFrame({"Year": np.arange(2025, 2051)})
t4_keys   = t4[["Sector","Region","Scenario"]].drop_duplicates()
t4_full   = t4_keys.merge(all_years, how="cross")
t4_full = t4_full.merge(
    t4, on=["Sector","Region","Year","Scenario"], how="left"
) 
t4_full = t4_full.sort_values(["Sector","Region","Scenario","Year"])

# Interpolate gaps between anchors, then flat-fill post-2030
t4_full["Carbon Price (USD/tCO2e)"] = (
    t4_full
    .groupby(["Sector","Region","Scenario"])["Carbon Price (USD/tCO2e)"]
    .transform(lambda g: g.interpolate(method="index")  # fills 2026/2028/2029
                          .ffill())                      # flat from 2030 onwards
)
Emission_pathway_long = Emission_pathway_long.merge(
   t4_full,
    left_on=["GICS Sector","Country","Projection Year"],
    right_on=["Sector","Region","Year"],
    how="left",
)
# Carbon Cost = what the company pays on emissions it still emits
Emission_pathway_long["Carbon Cost (USD)"] = (
    Emission_pathway_long["Carbon Price (USD/tCO2e)"] *
    Emission_pathway_long["Projected Emission"]
).round(2)

# Calculate emission reduction
# Clipped at 0 — cannot report negative reduction 
Emission_pathway_long["Emission Reduction"] = (
    Emission_pathway_long["Baseline_Emission"]
    - Emission_pathway_long["Projected Emission"]
).clip(lower=0)

# Calculate total abatement cost

intensity_mask = (
    Emission_pathway_long["Reduction target"]
    .astype(str).str.contains("tco2e", case=False)
)

beats_target = intensity_mask & (
    pd.Series(
        compute_target_emission(Emission_pathway_long),
        index=Emission_pathway_long.index
    ) > Emission_pathway_long["Baseline_Emission"]
)
Emission_pathway_long["Abatement Basis"] = np.select(
    [
        Emission_pathway_long["Emission Reduction"] > 0,
        beats_target,
    ],
    [   Emission_pathway_long["Emission Reduction"],
        np.minimum(
            Emission_pathway_long["Abatement Capacity (tCO2e/year)"],
            Emission_pathway_long["Projected Emission"]
        )
    ],
    default=0
)

Emission_pathway_long["Total Abatement Cost (USD)"] = (
    Emission_pathway_long["Abatement Basis"]
    * Emission_pathway_long["Abatement Cost (USD/tCO2e/year)"]
).round(2)
Emission_pathway_long["Carbon Cost Avoided (USD)"] = (
    Emission_pathway_long["Abatement Basis"]
    * Emission_pathway_long["Carbon Price (USD/tCO2e)"]
).round(2)
Emission_pathway_long["Net Benefit of Abatement (USD)"] = (
    Emission_pathway_long["Carbon Cost Avoided (USD)"]
    - Emission_pathway_long["Total Abatement Cost (USD)"]
).round(2)

#Breakeven -Carbon Price
Emission_pathway_long["Break_even Carbon Price (USD/tCO2e)"] =(
    Emission_pathway_long["Total Abatement Cost (USD)"]
    / Emission_pathway_long["Abatement Basis"].replace(0, np.nan)
).round(2)

#ABATEMENT ATTRACTIVE FLAG
Emission_pathway_long["Abatement Attractive"] = np.select(
    [
        Emission_pathway_long["Net Benefit of Abatement (USD)"] > 0,
        Emission_pathway_long["Net Benefit of Abatement (USD)"] <= 0,
        Emission_pathway_long["Net Benefit of Abatement (USD)"].isna()
    ],
    [
        "Yes",
        "No",
        "Insufficient Data"
    ],
    default="Insufficient Data"
)
#ABATEMENT BASIS TYPE FLAG
Emission_pathway_long["Abatement Basis Type"] = np.select(
    [
        Emission_pathway_long["Baseline_Emission"].isna(),
        Emission_pathway_long["Carbon Price (USD/tCO2e)"].isna() &
        Emission_pathway_long["Abatement Cost (USD/tCO2e/year)"].isna(),
        Emission_pathway_long["Projection Year"] == Emission_pathway_long["baseline_year"],
        beats_target,
        Emission_pathway_long["Emission Reduction"] > 0,
    ],
    [
        "No Baseline Emission Data",
        "No Carbon/Abatement Cost available",
        "Baseline Year- Hence No Reduction Yet",
        "Potential Abatement (Already Beats Target)",
        "Actual Reduction",
    ],
    default="No Reduction Target Given"
)
# SUMMARY SHEET
# Aggregates the year-by-year pathway to one row per UID × Target × Scenario.
summary = (
    Emission_pathway_long
    .groupby(["UID","GICS Sector","Country","Type of Target","Scenario"], dropna=False)
    .agg(
        Baseline_Emission           = ("Baseline_Emission",                "first"),
        Final_Projected_Emission    = ("Projected Emission",               "last"),
        Total_Emission_Reduction    = ("Emission Reduction",               "sum"),
        Cumulative_Carbon_Cost      = ("Carbon Cost (USD)",                "sum"),
        Cumulative_Abatement_Cost   = ("Total Abatement Cost (USD)",       "sum"),
        Cumulative_Carbon_Avoided   = ("Carbon Cost Avoided (USD)",        "sum"),
        Cumulative_Net_Benefit      = ("Net Benefit of Abatement (USD)",   "sum"),
        Avg_Break_even_Price        = ("Break_even Carbon Price (USD/tCO2e)", "mean"),
        Years_Abatement_Attractive  = ("Abatement Attractive",
                                       lambda x: (x == "Yes").sum()),
    )
    .reset_index()
)

# Total cost = what company actually pays (abatement + remaining carbon cost)
summary["Total Cost (USD)"] = (
    summary["Cumulative_Carbon_Cost"] + summary["Cumulative_Abatement_Cost"]
)

# Cost saving vs BAU_Carbon_Cost (no abatement, full carbon cost on baseline)
# "BAU_Carbon_Cost" = paying carbon cost on full baseline emission every year
baseline_carbon_cost = (
    Emission_pathway_long
    .groupby(["UID","Type of Target","Scenario"], dropna=False)
    .apply(lambda g: (
        g["Baseline_Emission"].iloc[0]
        * g["Carbon Price (USD/tCO2e)"]
    ).sum())
    .reset_index(name="BAU_Carbon_cost")
)

summary = summary.merge(
    baseline_carbon_cost,
    on=["UID","Type of Target","Scenario"],
    how="left"
)

summary["Cost Saving vs BAU_Carbon_cost(USD)"] = (
    summary["BAU_Carbon_cost"] - summary["Total Cost (USD)"]
).round(2)

# Attractiveness rank within each scenario — higher net benefit = more attractive
summary["Attractiveness Rank"] = (
    summary.groupby("Scenario")["Cumulative_Net_Benefit"]
    .rank(ascending=False, method="dense")
    .astype("Int64")
)
scenario_comparison = summary.pivot_table(
    index=["UID","GICS Sector","Country","Type of Target"],
    columns="Scenario",
    values=[
        "Cumulative_Carbon_Cost",
        "Cumulative_Abatement_Cost",
        "Cumulative_Net_Benefit",
        "Avg_Break_even_Price",
        "Attractiveness Rank"
    ]
).round(2)

# Flatten multi-level columns
scenario_comparison.columns = [
    f"{metric} | {scenario}"
    for metric, scenario in scenario_comparison.columns
]
scenario_comparison = scenario_comparison.reset_index()

#WRITE OUTPUT FILE
with pd.ExcelWriter("Output_Master_File.xlsx") as writer:
    t1.to_excel(writer, sheet_name="Table 1", index=False)
    t2.to_excel(writer, sheet_name="Table 2", index=False)
    t3.to_excel(writer, sheet_name="Table 3", index=False)
    t4.to_excel(writer, sheet_name="Table 4", index=False)
    master.to_excel(writer, sheet_name="Master Sheet", index=False)
    Emission_pathway_long.to_excel(writer, sheet_name='Emission_Pathway',index=False)   
    summary.to_excel(writer, sheet_name="Summary", index=False)
    scenario_comparison.to_excel(writer, sheet_name="Scenario Comparison", index=False)
