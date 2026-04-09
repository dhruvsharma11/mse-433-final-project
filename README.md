# Olympic Medal Performance — MSE 433 Individual Project

**Evaluating and Optimizing Olympic Medal Performance for National Sport Agencies (1960–2016)**

A three-stage data-driven framework — descriptive → predictive → prescriptive — that helps National Sport Agencies (NSAs) understand the structural drivers of Olympic success and optimize their sport portfolio under real resource constraints.

---

## What It Does

| Stage | Tool | Output |
|---|---|---|
| **Descriptive** | `01_EDA.ipynb` | 7 figures characterizing medal distributions, key predictors, host effects, Pareto concentration |
| **Predictive** | `02_Predictive_Model.ipynb` | Zero-Inflated Negative Binomial (ZINB) model — estimates expected medals from GDP, delegation size, female ratio, etc. |
| **Prescriptive** | `03_Optimization.ipynb` | Gurobi 12 Integer Linear Program — recommends which sports to enter and how many athletes to send to maximize expected medals |
| **Interactive UI** | `src/app.py` | Streamlit app — any NSA can input their profile and get a real-time optimal sport allocation |

---

## Project Structure

```
Individual_Project/
├── data/
│   ├── input/                  # Raw datasets (athlete_events, GDP, population, hosts)
│   └── output/                 # Generated panels + saved model .pkl files
├── notebooks/
│   ├── 01_EDA.ipynb            # Data prep + exploratory analysis (Figures 1–7)
│   ├── 02_Predictive_Model.ipynb  # ZINB + NegBin models (Figures 8–10)
│   └── 03_Optimization.ipynb   # Gurobi ILP + Canada 2016 case study (Figures 11–12)
├── src/
│   ├── data_prep.py            # Feature engineering, NOC mapping
│   ├── models.py               # ZINB, NegBin, cross-validation, sensitivity
│   ├── optimization.py         # Gurobi ILP solver, sport parameter estimation
│   └── app.py                  # Streamlit UI
├── report/
│   ├── Final_Report.md         # Full written report
│   └── figures/                # All saved plots
└── requirements.txt
```

---

## Setup

**1. Create and activate a virtual environment**

```bash
python3 -m venv mse433-env
source mse433-env/bin/activate        # Mac/Linux
# mse433-env\Scripts\activate         # Windows
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Gurobi license** *(required for the optimization notebook and Streamlit app)*

Get a free academic license at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/) and activate it:

```bash
grbgetkey <your-license-key>
```

> Without a Gurobi license, the optimization falls back to a greedy heuristic automatically.

---

## How to Run

Run the notebooks **in order** — each one depends on outputs from the previous.

### Step 1 — EDA (generates panel CSVs + Figures 1–7)
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
Outputs: `data/output/panel_summer.csv`, `data/output/panel_winter.csv`, figures in `report/figures/`

### Step 2 — Predictive Model (fits ZINB, generates Figures 8–10)
```bash
jupyter notebook notebooks/02_Predictive_Model.ipynb
```
Outputs: `data/output/zinb_summer.pkl`, `data/output/zinb_winter.pkl`, `negbin_*.pkl`

### Step 3 — Optimization (Gurobi ILP + Canada case study, generates Figures 11–12)
```bash
jupyter notebook notebooks/03_Optimization.ipynb
```

### Streamlit App (interactive NSA optimizer)
```bash
streamlit run src/app.py
```
> Requires Step 2 to have been run first (loads saved `.pkl` model files).

---

## Key Results

- **Delegation size** is the dominant predictor of medal success (elasticity ~1.9 for Summer)
- **Female participation ratio** has a significant independent effect (Summer: +10pp → +13% expected medals)
- **Sport specialization** (HHI) outperforms breadth — concentrating athletes in fewer, high-yield sports wins more medals
- **Canada 2016 case study:** The ILP shows that even with a fixed 314-athlete budget, concentrating on fewer sports would be expected to outperform Canada's actual 27-sport broad-portfolio strategy

---

## Data Sources

| Dataset | Source |
|---|---|
| `athlete_events.csv` | [Kaggle — 120 Years of Olympic History](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results) |
| `GDP_1960_2024.csv` | World Bank World Development Indicators |
| `Population_1960_2024.csv` | World Bank World Development Indicators |
| `olympic_hosts.csv` | Olympic.org historical records |
