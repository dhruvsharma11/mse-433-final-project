"""
data_prep.py
============
Loads and merges the four raw datasets into country-level Olympic panel datasets
(one row per country per Olympic Games), separately for Summer and Winter Games.

Inputs  (data/input/):
  athlete_events.csv, olympic_hosts.csv, GDP_1960_2024.csv, Population_1960_2024.csv

Outputs (data/output/):
  panel_summer.csv, panel_winter.csv

Usage:
  python src/data_prep.py
  from src.data_prep import build_panels; summer, winter = build_panels()
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "data" / "input"
OUTPUT = ROOT / "data" / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

# NOC → World Bank country name mapping (handles historical NOC codes)
NOC_TO_WB = {
    # Cold War / dissolved states
    "URS": "Russian Federation",        # Soviet Union → use Russia as closest proxy
    "GDR": "Germany",                   # East Germany
    "FRG": "Germany",                   # West Germany
    "YUG": "Serbia",                    # Yugoslavia → Serbia as primary successor
    "TCH": "Czech Republic",            # Czechoslovakia → Czech Republic
    "SCG": "Serbia",                    # Serbia & Montenegro
    "BOH": None,                        # Bohemia (historical, pre-1918)
    "ANZ": "Australia",                 # Australasia (1908/1912)

    # Modern codes that differ from World Bank names
    "USA": "United States",
    "GBR": "United Kingdom",
    "GER": "Germany",
    "RUS": "Russian Federation",
    "CHN": "China",
    "KOR": "Korea, Rep.",
    "PRK": "Korea, Dem. People's Rep.",
    "IRI": "Iran, Islamic Rep.",
    "CIV": "Cote d'Ivoire",
    "CGO": "Congo, Rep.",
    "COD": "Congo, Dem. Rep.",
    "SYR": "Syrian Arab Republic",
    "VEN": "Venezuela, RB",
    "YEM": "Yemen, Rep.",
    "LAO": "Lao PDR",
    "MAS": "Malaysia",
    "PHI": "Philippines",
    "SUI": "Switzerland",
    "SLO": "Slovenia",
    "CRO": "Croatia",
    "MGL": "Mongolia",
    "TPE": "Taiwan",                    # Chinese Taipei — not in World Bank; use Taiwan
    "HKG": "Hong Kong SAR, China",
    "MAC": "Macao SAR, China",
    "TAN": "Tanzania",
    "ZIM": "Zimbabwe",
    "ZAM": "Zambia",
    "NGR": "Nigeria",
    "GHA": "Ghana",
    "GUI": "Guinea",
    "SEN": "Senegal",
    "CMR": "Cameroon",
    "ETH": "Ethiopia",
    "KEN": "Kenya",
    "UGA": "Uganda",
    "RSA": "South Africa",
    "EGY": "Egypt, Arab Rep.",
    "MAR": "Morocco",
    "ALG": "Algeria",
    "TUN": "Tunisia",
    "LIB": "Lebanon",
    "IRQ": "Iraq",
    "KUW": "Kuwait",
    "UAE": "United Arab Emirates",
    "QAT": "Qatar",
    "BRN": "Bahrain",
    "OMA": "Oman",
    "JOR": "Jordan",
    "ISR": "Israel",
    "POR": "Portugal",
    "ESP": "Spain",
    "NED": "Netherlands",
    "BEL": "Belgium",
    "DEN": "Denmark",
    "NOR": "Norway",
    "SWE": "Sweden",
    "FIN": "Finland",
    "POL": "Poland",
    "CZE": "Czech Republic",
    "SVK": "Slovak Republic",
    "HUN": "Hungary",
    "ROU": "Romania",
    "BUL": "Bulgaria",
    "GRE": "Greece",
    "TUR": "Turkey",
    "AUT": "Austria",
    "LUX": "Luxembourg",
    "IRL": "Ireland",
    "ISL": "Iceland",
    "LAT": "Latvia",
    "LTU": "Lithuania",
    "EST": "Estonia",
    "UKR": "Ukraine",
    "BLR": "Belarus",
    "KAZ": "Kazakhstan",
    "UZB": "Uzbekistan",
    "AZE": "Azerbaijan",
    "GEO": "Georgia",
    "ARM": "Armenia",
    "MDA": "Moldova",
    "ALB": "Albania",
    "BIH": "Bosnia and Herzegovina",
    "MKD": "North Macedonia",
    "MNE": "Montenegro",
    "CYP": "Cyprus",
    "MLT": "Malta",
    "ARG": "Argentina",
    "BRA": "Brazil",
    "CHI": "Chile",
    "COL": "Colombia",
    "ECU": "Ecuador",
    "PER": "Peru",
    "URU": "Uruguay",
    "PAR": "Paraguay",
    "BOL": "Bolivia",
    "CRC": "Costa Rica",
    "GUA": "Guatemala",
    "HON": "Honduras",
    "ESA": "El Salvador",
    "NCA": "Nicaragua",
    "PAN": "Panama",
    "CUB": "Cuba",
    "DOM": "Dominican Republic",
    "PUR": "Puerto Rico",
    "JAM": "Jamaica",
    "TTO": "Trinidad and Tobago",
    "BAH": "Bahamas, The",
    "IND": "India",
    "PAK": "Pakistan",
    "BAN": "Bangladesh",
    "SRI": "Sri Lanka",
    "NEP": "Nepal",
    "AFG": "Afghanistan",
    "MYA": "Myanmar",
    "THA": "Thailand",
    "VIE": "Vietnam",
    "CAM": "Cambodia",
    "INA": "Indonesia",
    "SGP": "Singapore",
    "JPN": "Japan",
    "AUS": "Australia",
    "NZL": "New Zealand",
    "CAN": "Canada",
    "MEX": "Mexico",
    "ROC": "Russian Federation",       # Russian Olympic Committee (post-2021 sanctions)
    "EUN": "Russian Federation",        # Unified Team (1992, former Soviet)
    "IOP": None,                        # Individual Olympic Participants (stateless)
    "IOA": None,                        # Individual Olympic Athletes
    "EOR": None,                        # Refugee Olympic Team
    "SGP": "Singapore",
    "BRU": "Brunei Darussalam",
    "MOZ": "Mozambique",
    "ANG": "Angola",
    "CMR": "Cameroon",
    "BEN": "Benin",
    "TOG": "Togo",
    "MLI": "Mali",
    "BUR": "Burkina Faso",
    "NIG": "Niger",
    "CHA": "Chad",
    "CAF": "Central African Republic",
    "GAB": "Gabon",
    "EQG": "Equatorial Guinea",
    "STP": "Sao Tome and Principe",
    "CPV": "Cabo Verde",
    "SLE": "Sierra Leone",
    "LBR": "Liberia",
    "GNB": "Guinea-Bissau",
    "GAM": "Gambia, The",
    "MTN": "Mauritania",
    "SOM": "Somalia",
    "DJI": "Djibouti",
    "ERI": "Eritrea",
    "SUD": "Sudan",
    "RWA": "Rwanda",
    "BDI": "Burundi",
    "COM": "Comoros",
    "SEY": "Seychelles",
    "MRI": "Mauritius",
    "MDG": "Madagascar",
    "MWI": "Malawi",
    "ZIM": "Zimbabwe",
    "BOT": "Botswana",
    "SWZ": "Eswatini",
    "LES": "Lesotho",
    "NAM": "Namibia",
    "FIJ": "Fiji",
    "PNG": "Papua New Guinea",
    "SOL": "Solomon Islands",
    "VAN": "Vanuatu",
    "SAM": "Samoa",
    "TGA": "Tonga",
    "COK": None,                        # Cook Islands (not in World Bank)
    "FSM": "Micronesia, Fed. Sts.",
    "PLW": "Palau",
    "MHL": "Marshall Islands",
    "KIR": "Kiribati",
    "TUV": "Tuvalu",
    "NRU": "Nauru",
    "LCA": "St. Lucia",
    "VIN": "St. Vincent and the Grenadines",
    "GRN": "Grenada",
    "ANT": None,                        # Netherlands Antilles (dissolved 2010)
    "ISV": "Virgin Islands (U.S.)",
    "PLE": None,                        # Palestine (not always in WB as sovereign)
    "KOS": "Kosovo",
    "MTN": "Mauritania",
    "CUB": "Cuba",
    "VEN": "Venezuela, RB",
}

def load_athletes() -> pd.DataFrame:
    """Load and clean athlete_events.csv, filtering to 1960-2016."""
    df = pd.read_csv(INPUT / "athlete_events.csv")
    df = df[df["Year"] >= 1960].copy()
    df["has_medal"] = df["Medal"].notna() & (df["Medal"] != "NA")
    df["is_gold"] = df["Medal"] == "Gold"
    df["is_silver"] = df["Medal"] == "Silver"
    df["is_bronze"] = df["Medal"] == "Bronze"
    return df


def load_hosts() -> pd.DataFrame:
    """Load olympic_hosts.csv and extract host country per game."""
    df = pd.read_csv(INPUT / "olympic_hosts.csv")
    df["Year"] = df["Year"].astype(int)
    df["Season"] = df["Type"].str.replace("games", "", regex=False).str.strip()
    df["Season"] = df["Season"].str.replace("summer", "Summer").str.replace("winter", "Winter")
    df = df[["Year", "Season", "Country", "Events"]].rename(columns={"Country": "host_country"})
    return df


def load_gdp() -> pd.DataFrame:
    """Load World Bank GDP (wide format) and reshape to long."""
    df = pd.read_csv(INPUT / "GDP_1960_2024.csv", skiprows=4)
    year_cols = [c for c in df.columns if c.isdigit()]
    df = df[["Country Name", "Country Code"] + year_cols].copy()
    df = df.melt(id_vars=["Country Name", "Country Code"],
                 var_name="Year", value_name="gdp")
    df["Year"] = df["Year"].astype(int)
    df = df.rename(columns={"Country Name": "wb_country", "Country Code": "wb_code"})
    return df


def load_population() -> pd.DataFrame:
    """Load World Bank Population (wide format) and reshape to long."""
    df = pd.read_csv(INPUT / "Population_1960_2024.csv", skiprows=4)
    year_cols = [c for c in df.columns if c.isdigit()]
    df = df[["Country Name", "Country Code"] + year_cols].copy()
    df = df.melt(id_vars=["Country Name", "Country Code"],
                 var_name="Year", value_name="population")
    df["Year"] = df["Year"].astype(int)
    df = df.rename(columns={"Country Name": "wb_country", "Country Code": "wb_code"})
    return df


def herfindahl_index(series: pd.Series) -> float:
    """
    Herfindahl-Hirschman Index on a categorical series.
    Returns value in [0, 1] where 1 = perfect concentration (one sport only).
    """
    if series.empty or series.sum() == 0:
        return 0.0
    shares = series / series.sum()
    return float((shares ** 2).sum())


def deduplicate_medals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate medals for team sports: a team event should count as ONE medal
    for the country, not N medals (one per athlete).
    Strategy: keep one row per (NOC, Year, Season, Event, Medal) combination
    where Medal is not null.
    """
    medal_rows = df[df["has_medal"]].copy()
    # For team events, multiple athletes share the same Event and Medal type.
    # Deduplicate: count each (NOC, Year, Season, Event, Medal) once.
    deduped = (
        medal_rows
        .drop_duplicates(subset=["NOC", "Year", "Season", "Event", "Medal"])
    )
    return deduped


def build_country_panel(athletes: pd.DataFrame, season: str) -> pd.DataFrame:
    """Aggregate athlete-level data to country × Olympic-game level."""
    df = athletes[athletes["Season"] == season].copy()

    delegation = (df.groupby(["NOC", "Year"])["ID"].nunique()
                  .reset_index().rename(columns={"ID": "delegation_size"}))

    gender = (df.groupby(["NOC", "Year"])
              .apply(lambda g: (g["Sex"] == "F").sum() / len(g), include_groups=False)
              .reset_index().rename(columns={0: "female_ratio"}))

    sport_count = (df.groupby(["NOC", "Year"])["Sport"].nunique()
                   .reset_index().rename(columns={"Sport": "sport_count"}))

    deduped = deduplicate_medals(df)
    medal_totals = (deduped.groupby(["NOC", "Year"])
                    .agg(total_medals=("has_medal", "sum"),
                         gold_medals=("is_gold", "sum"),
                         silver_medals=("is_silver", "sum"),
                         bronze_medals=("is_bronze", "sum"))
                    .reset_index())

    def sport_hhi(group):
        sport_medals = group.groupby("Sport")["has_medal"].sum()
        return herfindahl_index(sport_medals)

    hhi = (deduped.groupby(["NOC", "Year"])
           .apply(sport_hhi, include_groups=False)
           .reset_index().rename(columns={0: "sport_hhi"}))

    panel = delegation.copy()
    for frame in [gender, sport_count, medal_totals, hhi]:
        panel = panel.merge(frame, on=["NOC", "Year"], how="left")

    for col in ["total_medals", "gold_medals", "silver_medals", "bronze_medals"]:
        panel[col] = panel[col].fillna(0).astype(int)
    panel["sport_hhi"] = panel["sport_hhi"].fillna(0.0)
    panel["female_ratio"] = panel["female_ratio"].fillna(0.0)

    panel["medal_rate"] = panel["total_medals"] / panel["delegation_size"]
    panel["Season"] = season

    return panel


def add_host_flag(panel: pd.DataFrame, hosts: pd.DataFrame) -> pd.DataFrame:
    """Add binary is_host flag based on host country lookup."""
    HOST_NOC = {
        # Summer Games (1960-2016)
        (1960, "Summer"): "ITA",   # Rome
        (1964, "Summer"): "JPN",   # Tokyo
        (1968, "Summer"): "MEX",   # Mexico City
        (1972, "Summer"): "FRG",   # Munich → Germany
        (1976, "Summer"): "CAN",   # Montreal
        (1980, "Summer"): "URS",   # Moscow
        (1984, "Summer"): "USA",   # Los Angeles
        (1988, "Summer"): "KOR",   # Seoul
        (1992, "Summer"): "ESP",   # Barcelona
        (1996, "Summer"): "USA",   # Atlanta
        (2000, "Summer"): "AUS",   # Sydney
        (2004, "Summer"): "GRE",   # Athens
        (2008, "Summer"): "CHN",   # Beijing
        (2012, "Summer"): "GBR",   # London
        (2016, "Summer"): "BRA",   # Rio
        # Winter Games (1960-2016)
        (1960, "Winter"): "USA",   # Squaw Valley
        (1964, "Winter"): "AUT",   # Innsbruck
        (1968, "Winter"): "FRA",   # Grenoble
        (1972, "Winter"): "JPN",   # Sapporo
        (1976, "Winter"): "AUT",   # Innsbruck
        (1980, "Winter"): "USA",   # Lake Placid
        (1984, "Winter"): "YUG",   # Sarajevo
        (1988, "Winter"): "CAN",   # Calgary
        (1992, "Winter"): "FRA",   # Albertville
        (1994, "Winter"): "NOR",   # Lillehammer
        (1998, "Winter"): "JPN",   # Nagano
        (2002, "Winter"): "USA",   # Salt Lake City
        (2006, "Winter"): "ITA",   # Turin
        (2010, "Winter"): "CAN",   # Vancouver
        (2014, "Winter"): "RUS",   # Sochi
    }

    panel = panel.copy()
    panel["is_host"] = panel.apply(
        lambda r: int(HOST_NOC.get((r["Year"], r["Season"]), "") == r["NOC"]),
        axis=1
    )
    return panel


def add_macro(panel: pd.DataFrame, gdp: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Join GDP and population data using NOC_TO_WB mapping."""
    panel = panel.copy()
    panel["wb_country"] = panel["NOC"].map(NOC_TO_WB)

    gdp_slim = gdp[["wb_country", "Year", "gdp"]].dropna(subset=["gdp"])
    panel = panel.merge(gdp_slim, on=["wb_country", "Year"], how="left")

    pop_slim = pop[["wb_country", "Year", "population"]].dropna(subset=["population"])
    panel = panel.merge(pop_slim, on=["wb_country", "Year"], how="left")

    panel["gdp_per_capita"] = panel["gdp"] / panel["population"]
    panel["log_gdp"] = np.log(panel["gdp"].clip(lower=1))
    panel["log_population"] = np.log(panel["population"].clip(lower=1))
    panel["log_gdp_per_capita"] = np.log(panel["gdp_per_capita"].clip(lower=1))
    panel["log_delegation_size"] = np.log(panel["delegation_size"].clip(lower=1))
    panel["log_total_medals"] = np.log1p(panel["total_medals"])

    return panel


def build_panels(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load raw data → build features → join macro → save panels.

    Returns:
        (panel_summer, panel_winter) DataFrames
    """
    print("Loading raw data...")
    athletes = load_athletes()
    hosts = load_hosts()
    gdp = load_gdp()
    pop = load_population()

    print(f"  Athletes (1960-2016): {len(athletes):,} rows")

    panels = {}
    for season in ["Summer", "Winter"]:
        print(f"\nBuilding {season} panel...")
        panel = build_country_panel(athletes, season)
        panel = add_host_flag(panel, hosts)
        panel = add_macro(panel, gdp, pop)
        print(f"  {season} panel: {len(panel):,} rows, {panel['NOC'].nunique()} NOCs, "
              f"{panel['Year'].nunique()} games")
        print(f"  Missing GDP: {panel['gdp'].isna().sum()} rows "
              f"({panel['gdp'].isna().mean():.1%})")
        print(f"  Missing Population: {panel['population'].isna().sum()} rows")
        panels[season] = panel

    panel_summer = panels["Summer"]
    panel_winter = panels["Winter"]

    if save:
        out_s = OUTPUT / "panel_summer.csv"
        out_w = OUTPUT / "panel_winter.csv"
        panel_summer.to_csv(out_s, index=False)
        panel_winter.to_csv(out_w, index=False)
        print(f"\nSaved: {out_s}")
        print(f"Saved: {out_w}")

    return panel_summer, panel_winter


if __name__ == "__main__":
    summer, winter = build_panels(save=True)
    print("\nSummer panel columns:", list(summer.columns))
    print("Winter panel sample:\n", winter.head(3).to_string())
