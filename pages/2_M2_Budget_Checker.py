import streamlit as st
import pandas as pd
from statistics import median
from typing import Dict, Tuple, Optional

# --- CSV/XLSX loader: flexibel inlezen & kolommen normaliseren ---
def _standardize_col(c: str) -> str:
    c = str(c).strip()
    c = c.lower().replace(" ", "_").replace("-", "_")
    c = c.replace("m²", "m2")
    return c

def _parse_bool(x):
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "y", "ja")
    return bool(x)

def load_dataset_flexible(uploaded_file):
    import io, os
    import pandas as pd
    import streamlit as st

    # ---------- helpers ----------
    def read_csv_flexible(text: str):
        # 1) auto-sniffer
        try:
            return pd.read_csv(io.StringIO(text), engine="python", sep=None)
        except Exception:
            pass
        # 2) bekende schema’s
        for args in [dict(sep=","), dict(sep=";"), dict(sep="\t"), dict(sep="|")]:
            try:
                return pd.read_csv(io.StringIO(text), **args)
            except Exception:
                continue
        return None

    def salvage_single_col_text(text: str):
        # Repareer “alles-in-één-kolom” CSV’s
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return pd.DataFrame()
        header = lines[0]
        delim = "," if "," in header else ";" if ";" in header else ("\t" if "\t" in header else None)
        if not delim:
            return pd.DataFrame()
        cols = [_standardize_col(x) for x in header.split(delim)]
        rows = []
        for ln in lines[1:]:
            cells = [s.strip() for s in ln.split(delim)]
            if len(cells) < len(cols):  # padding
                cells += [""] * (len(cols) - len(cells))
            rows.append(cells[:len(cols)])
        return pd.DataFrame(rows, columns=cols)

    def load_local_seed():
        # 1) Probeer CSV seed (detecteer scheidingsteken, BOM)
        for args in [dict(engine="python", sep=None, encoding="utf-8-sig"),
                     dict(sep=";", encoding="utf-8-sig"),
                     dict(sep=",", encoding="utf-8-sig")]:
            try:
                df_local = pd.read_csv("m2_budget_checker_seed.csv", **args)
                st.sidebar.info(f"Seed dataset (CSV) geladen: {len(df_local)} records.")
                return df_local
            except Exception:
                pass
        # 2) Probeer Excel intake als seed
        for name in ["M2_Budget_Checker_Intake.xlsx", "m2_budget_checker.xlsx"]:
            if os.path.exists(name):
                try:
                    df_local = pd.read_excel(name, engine="openpyxl")
                    st.sidebar.info(f"Seed dataset (Excel: {name}) geladen: {len(df_local)} records.")
                    return df_local
                except Exception:
                    pass
        st.sidebar.warning("Geen dataset gevonden. Upload een CSV/Excel of voeg beneden 'Actuals' toe.")
        return pd.DataFrame()

    # ---------- lezen: upload of lokale seed ----------
    if uploaded_file is not None:
        fname = (uploaded_file.name or "").lower()
        if fname.endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(uploaded_file, engine="openpyxl")
                st.sidebar.success(f"Excel geladen: {len(df)} records.")
            except Exception as e:
                st.sidebar.error(f"Kon Excel niet lezen: {e}")
                return pd.DataFrame()
        else:
            raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            df = read_csv_flexible(raw)
            # ‘alles-in-1-kolom’-herstel
            if df is None or (df.shape[1] <= 2 and any(("," in str(c) or ";" in str(c)) for c in df.columns)):
                df = salvage_single_col_text(raw)
            if df is None or df.empty:
                st.sidebar.error("Kon CSV niet lezen. Tip: exporteer als CSV UTF-8 (Comma delimited).")
                return pd.DataFrame()
            st.sidebar.success(f"CSV geladen: {len(df)} records.")
    else:
        df = load_local_seed()

    # ---------- normaliseer kolomnamen & mappen ----------
    df.columns = [_standardize_col(c) for c in df.columns]

    rename_map = {
        # type
        "type": "project_type", "projecttype": "project_type",
        # regio/jaar
        "regio": "region", "jaar": "year",
        # m2
        "gross": "gross_m2", "bruto_m2": "gross_m2", "bvo_m2": "gross_m2", "bvo": "gross_m2",
        "net": "net_m2", "netto_m2": "net_m2", "nvo_m2": "net_m2", "nvo": "net_m2",
        # kosten
        "total": "total_cost_excl_vat", "totaal": "total_cost_excl_vat",
        "totaal_excl_btw": "total_cost_excl_vat", "cost_excl_vat": "total_cost_excl_vat",
        # pm/logistiek
        "include_pm": "include_pm_logistics", "pm": "include_pm_logistics",
        # kwaliteit/scope
        "kwaliteit": "quality", "scope_werk": "scope",
    }
    df = df.rename(columns=rename_map)

    # ---------- types ----------
    for col in ["year", "gross_m2", "net_m2", "total_cost_excl_vat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["include_mep", "include_pm_logistics"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_bool)

    # ---------- verplichte kolommen (moeten ALS KOP bestaan) ----------
    needed = {"project_type", "scope", "quality", "region", "year",
              "gross_m2", "net_m2", "total_cost_excl_vat"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.sidebar.error("CSV/Excel mist kolommen: " + ", ".join(missing))
        st.sidebar.write("Gevonden kolommen:", list(df.columns))
        return pd.DataFrame()

    # Kleine sanity: toon kolommen en aantal records
    st.sidebar.caption(f"Kolommen: {', '.join(list(df.columns))}")
    return df


st.set_page_config(page_title="M² Budget Checker", page_icon="📐", layout="wide")

# ------------------------------
# Defaults / Baselines
# ------------------------------
BASE_YEAR = 2024
BASELINES_P50 = {
    "office": 600.0,
    "hospitality": 900.0,
    "retail": 800.0,
    "residential": 700.0,
}

QUALITY_FACTOR = {"basic": 0.85, "mid": 1.00, "high": 1.25}
REGION_FACTOR = {"NL-Randstad": 1.05, "NL-overig": 0.97, "EU-noord": 1.00, "EU-zuid": 0.95}
COMPLEXITY_FACTOR = {"standard": 1.00, "maatwerk": 1.10, "monument": 1.20}
SCOPE_FACTOR = {"both": 1.00, "FF&E": 0.60, "afbouw": 0.55}
PM_FACTOR = {True: 1.07, False: 1.00}

CPI_INDEX = {2021: 1.027, 2022: 1.100, 2023: 1.038, 2024: 1.030, 2025: 1.026}

# NFC OpEx
NFC_OPEX_EUR_PER_M2 = 589.0   # €/m² VVO/jaar
NFC_M2_PER_WORKPLACE_VVO = 22.3

# ------------------------------
# Helpers
# ------------------------------
# --- CSV helpers: flexibel inlezen & kolommen normaliseren ---
def _standardize_col(c: str) -> str:
    c = str(c).strip()
    c = c.lower().replace(" ", "_").replace("-", "_")
    c = c.replace("m²", "m2")
    return c

def _parse_bool(x):
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "y", "ja")
    return bool(x)

def load_dataset_flexible(uploaded_file):
    import io
    import pandas as pd
    import streamlit as st

    def read_with_fallbacks(fobj):
        # Probeer sniffer (engine='python', sep=None), anders vaste lijst
        try:
            fobj.seek(0)
            return pd.read_csv(fobj, engine="python", sep=None)
        except Exception:
            pass
        for sep in [",", ";", "\t", "|"]:
            try:
                fobj.seek(0)
                return pd.read_csv(fobj, sep=sep)
            except Exception:
                continue
        return None

    if uploaded_file is not None:
        # UploadedFile -> StringIO
        buf = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
        df = read_with_fallbacks(buf)
        if df is None:
            st.sidebar.error("Kon CSV niet lezen. Probeer opnieuw (CSV UTF-8).")
            return pd.DataFrame()
        st.sidebar.success(f"Dataset geladen: {len(df)} records (upload).")
    else:
        # Probeer lokale seed (maakt niet uit of , of ;)
        try:
            df = pd.read_csv("m2_budget_checker_seed.csv", engine="python", sep=None)
            st.sidebar.info(f"Seed dataset geladen: {len(df)} records.")
        except Exception:
            try:
                df = pd.read_csv("m2_budget_checker_seed.csv", sep=";")
                st.sidebar.info(f"Seed dataset geladen (;) : {len(df)} records.")
            except Exception:
                st.sidebar.warning("Geen dataset gevonden. Upload een CSV of voeg beneden 'Actuals' toe.")
                return pd.DataFrame()

    # Kolomnaam-normalisatie
    df.columns = [_standardize_col(c) for c in df.columns]
    rename_map = {
        "type": "project_type",
        "projecttype": "project_type",
        "regio": "region",
        "jaar": "year",
        "gross": "gross_m2", "bruto_m2": "gross_m2", "bvo_m2": "gross_m2", "bvo": "gross_m2",
        "net": "net_m2", "netto_m2": "net_m2", "nvo_m2": "net_m2", "nvo": "net_m2",
        "total": "total_cost_excl_vat", "totaal": "total_cost_excl_vat",
        "totaal_excl_btw": "total_cost_excl_vat", "cost_excl_vat": "total_cost_excl_vat",
        "include_pm": "include_pm_logistics", "pm": "include_pm_logistics",
        "kwaliteit": "quality", "scope_werk": "scope",
    }
    df = df.rename(columns=rename_map)

    # Types netjes zetten
    for col in ["year", "gross_m2", "net_m2", "total_cost_excl_vat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["include_mep", "include_pm_logistics"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_bool)

    # Vereiste kolommen voor de benchmarks
    needed = {"project_type", "scope", "quality", "region", "year", "gross_m2", "net_m2", "total_cost_excl_vat"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.sidebar.error("CSV mist kolommen: " + ", ".join(missing))
        st.sidebar.write("Gevonden kolommen:", list(df.columns))
        return pd.DataFrame()

    return df

def inflation_factor(year: int, base_year: int, cpi: Dict[int, float]) -> float:
    if year == base_year:
        return 1.0
    years = sorted(cpi.keys())
    if year not in cpi or base_year not in cpi:
        return 1.0
    earliest = years[0]
    def cumulative(to_year: int) -> float:
        fac = 1.0
        for y in range(earliest, to_year + 1):
            fac *= cpi.get(y, 1.0)
        return fac
    base_idx = cumulative(base_year)
    tgt_idx = cumulative(year)
    return tgt_idx / base_idx

def compute_model_p50_per_m2(
    project_type: str, scope: str, quality: str, region: str,
    year: int, include_mep: bool, include_pm: bool, complexity: str, cpi_index: Dict[int, float],
    mep_factor_selected: float = 1.0
) -> float:
    base = BASELINES_P50.get(project_type, 650.0)
    val = base
    val *= SCOPE_FACTOR.get(scope, 1.0)
    val *= QUALITY_FACTOR.get(quality, 1.0)
    val *= REGION_FACTOR.get(region, 1.0)
    val *= COMPLEXITY_FACTOR.get(complexity, 1.0)
    val *= (mep_factor_selected if include_mep else 1.0)
    val *= PM_FACTOR.get(include_pm, 1.0)
    val *= inflation_factor(year, BASE_YEAR, cpi_index)
    return val

def compute_model_band(p50_per_m2: float) -> Tuple[float, float]:
    return p50_per_m2 * 0.85, p50_per_m2 * 1.25

def per_m2_from_record(row: pd.Series) -> Optional[float]:
    m2 = row.get("net_m2") if pd.notnull(row.get("net_m2")) and row.get("net_m2") > 0 else row.get("gross_m2")
    if pd.isnull(m2) or m2 <= 0:
        return None
    cost = row.get("total_cost_excl_vat")
    if pd.isnull(cost) or cost <= 0:
        return None
    return float(cost) / float(m2)

def empirical_estimate(df: pd.DataFrame, project_type: str, scope: str, quality: str, region: str) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    if df is None or df.empty:
        return None, None, None, 0
    q = df.copy()
    for col, val in [("project_type", project_type), ("scope", scope), ("quality", quality), ("region", region)]:
        q = q[q[col] == val]
    if q.empty:
        return None, None, None, 0
    values = [v for v in (per_m2_from_record(r) for _, r in q.iterrows()) if v is not None]
    n = len(values)
    if n == 0:
        return None, None, None, 0
    values_sorted = sorted(values)
    def pct(p):
        idx = max(0, min(len(values_sorted)-1, int(round((p/100) * (len(values_sorted)-1)))))
        return values_sorted[idx]
    p50 = median(values_sorted)
    p20 = pct(20)
    p80 = pct(80)
    return p50, p20, p80, n

def blended(p50_model: float, p20_model: float, p80_model: float, emp):
    ep50, ep20, ep80, n = emp
    if ep50 is None or n == 0:
        return p50_model, p20_model, p80_model, 0.0, n
    w = min(0.8, n / 10.0)
    p50 = (1-w)*p50_model + w*ep50
    p20 = (1-w)*p20_model + w*(ep20 if ep20 is not None else ep50*0.85)
    p80 = (1-w)*p80_model + w*(ep80 if ep80 is not None else ep50*1.25)
    return p50, p20, p80, w, n

# ------------------------------
# Sidebar config
# ------------------------------
st.sidebar.header("⚙️ Config")
cpi_edit = st.sidebar.checkbox("CPI handmatig bewerken?", value=False)
cpi_state = CPI_INDEX.copy()
if cpi_edit:
    st.sidebar.caption("Index per jaar (1.03 betekent +3% t.o.v. vorig jaar).")
    for y in sorted(cpi_state.keys()):
        cpi_state[y] = st.sidebar.number_input(f"CPI {y}", value=float(cpi_state[y]), step=0.001, format="%.3f")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV of Excel)", type=["csv", "xlsx", "xls"])
df = load_dataset_flexible(uploaded)


# ------------------------------
# Main UI
# ------------------------------
st.title("📐 M² Budget Checker — MVP")
st.caption("Snelle realiteitscheck van budgetten per m². Toon bandbreedte en aannames.")

col1, col2, col3 = st.columns(3)

with col1:
    project_type = st.selectbox("Projecttype", ["office", "hospitality", "retail", "residential"])
    scope = st.selectbox("Scope", ["both", "FF&E", "afbouw"])
    quality = st.selectbox("Kwaliteit", ["basic", "mid", "high"])
with col2:
    region = st.selectbox("Regio", ["NL-Randstad", "NL-overig", "EU-noord", "EU-zuid"])
    year = st.number_input("Jaar (kostenpeil)", min_value=2021, max_value=2030, value=2025, step=1)
    complexity = st.selectbox("Complexiteit", ["standard", "maatwerk", "monument"])
with col3:
    include_mep = st.checkbox("Inclusief installaties (MEP: E + W/S)?", value=True, help="MEP = Mechanical, Electrical & Plumbing. NL: E/W/S = Elektrotechniek, Werktuigbouw/HVAC, Sanitair.")
    include_pm = st.checkbox("Inclusief PM & logistiek?", value=True)
    mep_intensity = st.selectbox("MEP-zwaarte", ["Licht (+5%)", "Normaal (+15%)", "Zwaar (+30%)"], index=1, help="Snelheidsknop voor installatieniveau: licht (minimale aanpassingen), normaal (typisch), zwaar (ingrijpende installaties).")
    m2 = st.number_input("Oppervlakte (m²)", min_value=10.0, max_value=50000.0, value=1000.0, step=10.0)

# --- SAFETY DEFAULTS for area basis ---
m2_vvo = m2
m2_input_type = "VVO"
nvo_to_vvo = 1.08
bvo_to_vvo = 0.80

# --- M² type & conversie (NEN 2580) ---
st.markdown("#### M² type & conversie")
m2_input_type = st.radio("Type van ingevoerde m²", ["VVO", "NVO (netto)", "BVO"], horizontal=True)
if m2_input_type == "NVO (netto)":
    nvo_to_vvo = st.slider("Conversie NVO → VVO factor", min_value=1.00, max_value=1.20, value=1.08, step=0.01)
    m2_vvo = m2 * nvo_to_vvo
elif m2_input_type == "BVO":
    bvo_to_vvo = st.slider("Conversie BVO → VVO efficiëntie", min_value=0.70, max_value=0.90, value=0.80, step=0.01)
    m2_vvo = m2 * bvo_to_vvo
else:
    m2_vvo = m2

st.caption(f"Interne rekenbasis: **{m2_vvo:,.0f} m² VVO** (input: {m2:,.0f} m² {'VVO' if m2_input_type=='VVO' else ('NVO' if m2_input_type.startswith('NVO') else 'BVO')})")

# MEP intensity factor mapping
MEP_INTENSITY_FACTOR = {"Licht (+5%)": 1.05, "Normaal (+15%)": 1.15, "Zwaar (+30%)": 1.30}
mep_factor_selected = MEP_INTENSITY_FACTOR.get(mep_intensity, 1.15)

# ------------------------------
# 👥 Office-specifieke inputs
# ------------------------------
n_employees = 0
annual_opex_facility = 0.0
annual_rent = 0.0
annual_opex_total = 0.0
eur_per_emp_year1 = None
eur_per_emp_breakdown = ""
total_opex = 0.0
horizon = 0
# Defaults for program/meeting so variables exist
wp_pct = mtg_pct = sup_pct = 0
capex_meeting_extra = 0.0
contingency_pct = 0

if project_type == "office":
    st.markdown("### 👥 Medewerkers & OpEx")
    cA, cB, cC = st.columns(3)
    with cA:
        n_employees = st.number_input("Aantal medewerkers", min_value=1, value=50, step=1)
    with cB:
        area_basis = st.radio("Bereken m² op basis van", ["Handmatig (boven)", "NFC-norm (22,3 m² VVO/pp)", "Norm-densiteit (net m²/pp)"], index=0)
    with cC:
        opex_growth = st.number_input("OpEx groei p.j. (%)", min_value=0.0, value=3.0, step=0.5)

    if area_basis == "NFC-norm (22,3 m² VVO/pp)":
        m2_vvo = n_employees * NFC_M2_PER_WORKPLACE_VVO
        st.info(f"Benodigde m² (VVO) o.b.v. NFC: **{m2_vvo:,.0f} m²**")
    elif area_basis == "Norm-densiteit (net m²/pp)":
        net_per_emp = st.slider("Norm-densiteit (netto m² per medewerker)", min_value=8.0, max_value=20.0, value=12.0, step=0.5, help="Vuistregel: 10–14 m² netto per medewerker afhankelijk van openheid/activiteit.")
        st.caption("We zetten netto om naar VVO met je NVO→VVO-factor hierboven.")
        m2_vvo = n_employees * net_per_emp * (nvo_to_vvo if m2_input_type != "BVO" else 1.0)

    st.markdown("#### 💶 Huur & OpEx")
    c1, c2 = st.columns(2)
    with c1:
        rent_per_m2_year = st.number_input("Huur €/m²/jaar (optioneel)", min_value=0.0, value=0.0, step=5.0)
    with c2:
        use_nfc_opex = st.checkbox("Gebruik NFC OpEx €589/m²/jaar (facilitair)", value=True, help="Als uit, kun je eigen OpEx invullen hieronder.")

    if use_nfc_opex:
        annual_opex_facility = NFC_OPEX_EUR_PER_M2 * m2_vvo
    else:
        annual_opex_facility = st.number_input("Eigen OpEx €/jaar (facilitair, excl. huur)", min_value=0.0, value=0.0, step=1000.0)

    annual_rent = rent_per_m2_year * m2_vvo
    annual_opex_total = annual_opex_facility + annual_rent

    horizon = st.slider("Horizon (jaren) voor TotEx", min_value=1, max_value=10, value=5, help="Aantal jaren waarover OpEx (huur + facilitair) wordt opgeteld. CapEx blijft eenmalig.")
    total_opex = sum(annual_opex_total * ((1 + opex_growth/100) ** t) for t in range(horizon))

    eur_per_emp_year1 = annual_opex_total / n_employees
    eur_per_emp_breakdown = f"(huur: €{annual_rent:,.0f} • facilitair: €{annual_opex_facility:,.0f})"

    cO1, cO2, cO3 = st.columns(3)
    with cO1:
        st.metric("Jaarlijkse OpEx totaal", f"€ {annual_opex_total:,.0f}", help="Huur + facilitair (jaar 1).")
    with cO2:
        st.metric("€/medewerker/jaar (jaar 1)", f"€ {eur_per_emp_year1:,.0f}")
    with cO3:
        st.caption(f"Waarvan: {eur_per_emp_breakdown}")

    # Programma-verdeling
    st.markdown("#### 📐 Programma-verdeling (VVO %)")
    wp_pct = st.slider("Werkplekken %", min_value=0, max_value=100, value=60, step=1)
    mtg_pct = st.slider("Vergaderruimtes %", min_value=0, max_value=100, value=25, step=1)
    sup_pct = st.slider("Support (receptie/pantry/opslag/copy) %", min_value=0, max_value=100, value=15, step=1)
    total_pct = wp_pct + mtg_pct + sup_pct
    if total_pct != 100:
        st.warning(f"Verdeling telt nu {total_pct}%. Maak samen 100% voor een kloppend programma.")

    # Vergaderruimtes add-on
    st.markdown("#### 🪑 Vergaderruimtes — € per zitplaats (optioneel extra CapEx)")
    cvs, cps = st.columns(2)
    with cvs:
        seats_small = st.number_input("Aantal zitplaatsen: kleine ruimtes", min_value=0, value=0, step=1)
        seats_med = st.number_input("Aantal zitplaatsen: medium ruimtes", min_value=0, value=0, step=1)
        seats_large = st.number_input("Aantal zitplaatsen: grote ruimtes", min_value=0, value=0, step=1)
    with cps:
        eur_small = st.number_input("€ per zitplaats (klein)", min_value=0.0, value=500.0, step=50.0)
        eur_med = st.number_input("€ per zitplaats (medium)", min_value=0.0, value=750.0, step=50.0)
        eur_large = st.number_input("€ per zitplaats (groot)", min_value=0.0, value=1000.0, step=50.0)

    capex_meeting_extra = seats_small*eur_small + seats_med*eur_med + seats_large*eur_large

    st.markdown("#### 🔁 Onvoorzien (CapEx)")
    contingency_pct = st.slider("Onvoorzien % op CapEx", min_value=0, max_value=30, value=5, step=1)

# Modelschatting (CapEx €/m²)
p50_model = compute_model_p50_per_m2(project_type, scope, quality, region, year, include_mep, include_pm, complexity, CPI_INDEX if not cpi_edit else cpi_state, mep_factor_selected)
p20_model, p80_model = compute_model_band(p50_model)

# Empirische schatting uit dataset
ep = empirical_estimate(df, project_type, scope, quality, region)

# Blend
p50, p20, p80, weight_emp, n_emp = blended(p50_model, p20_model, p80_model, ep)

# CapEx totals (basis + meeting add-on + onvoorzien)
capex_base = p50 * m2_vvo
capex_total = capex_base
if project_type == "office":
    capex_total += capex_meeting_extra
    capex_total *= (1 + contingency_pct/100.0)

# Resultaten
st.subheader("📊 Resultaat")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("P50 €/m² (blended)", f"{p50:,.0f}")
with c2:
    st.metric("Bandbreedte P20–P80 €/m²", f"{p20:,.0f} – {p80:,.0f}")
with c3:
    st.metric("Indicatieve CapEx (P50)", f"€ {capex_total:,.0f}")

# TotEx metric (alleen office): CapEx + OpEx over horizon
if project_type == "office" and horizon > 0:
    st.metric(f"TotEx (CapEx + OpEx, {horizon} jr)", f"€ {capex_total + total_opex:,.0f}")

# 📋 Programma-overzicht
if project_type == "office":
    total_pct_ok = (wp_pct + mtg_pct + sup_pct == 100)
    if total_pct_ok:
        m2_wp = m2_vvo * (wp_pct/100.0)
        m2_mtg = m2_vvo * (mtg_pct/100.0)
        m2_sup = m2_vvo * (sup_pct/100.0)

        capex_base_wp = p50 * m2_wp
        capex_base_mtg = p50 * m2_mtg
        capex_base_sup = p50 * m2_sup

        add_on_wp = 0.0
        add_on_mtg = capex_meeting_extra
        add_on_sup = 0.0

        precont_wp = capex_base_wp + add_on_wp
        precont_mtg = capex_base_mtg + add_on_mtg
        precont_sup = capex_base_sup + add_on_sup

        cont_wp = precont_wp * (contingency_pct/100.0)
        cont_mtg = precont_mtg * (contingency_pct/100.0)
        cont_sup = precont_sup * (contingency_pct/100.0)

        total_wp = precont_wp + cont_wp
        total_mtg = precont_mtg + cont_mtg
        total_sup = precont_sup + cont_sup

        opex_wp = annual_opex_total * (wp_pct/100.0)
        opex_mtg = annual_opex_total * (mtg_pct/100.0)
        opex_sup = annual_opex_total * (sup_pct/100.0)

        prog_df = pd.DataFrame([
            ["Werkplekken", wp_pct, m2_wp, capex_base_wp, add_on_wp, total_wp, opex_wp],
            ["Vergaderruimtes", mtg_pct, m2_mtg, capex_base_mtg, add_on_mtg, total_mtg, opex_mtg],
            ["Support", sup_pct, m2_sup, capex_base_sup, add_on_sup, total_sup, opex_sup],
        ], columns=["Categorie", "% VVO", "m² VVO", "CapEx basis €", "CapEx add-on €", "CapEx + onvoorzien €", "OpEx jaar 1 €"])

        st.markdown("### 📋 Programma-overzicht")
        st.dataframe(
            prog_df.style.format({
                "% VVO": "{:.0f}",
                "m² VVO": "{:,.0f}",
                "CapEx basis €": "€ {:,.0f}",
                "CapEx add-on €": "€ {:,.0f}",
                "CapEx + onvoorzien €": "€ {:,.0f}",
                "OpEx jaar 1 €": "€ {:,.0f}",
            }),
            use_container_width=True
        )
    else:
        st.info("📐 Programma-verdeling telt nu niet op tot 100%. Pas de sliders aan om het overzicht te tonen.")

with st.expander("Details & aannames"):
    if ep[0]:
        st.write(f"- Model P50 €/m²: **{p50_model:,.0f}**  | Empirisch P50: **{ep[0]:,.0f}** (n={ep[3]})")
    else:
        st.write(f"- Model P50 €/m²: **{p50_model:,.0f}**  | Empirisch: *(geen data)*")
    st.write(f"- Blending gewicht empirisch: **{weight_emp:.0%}**")
    st.write(f"- Rekenbasis m²: **VVO** = {m2_vvo:,.0f} (input {m2:,.0f} {m2_input_type}; NVO→VVO={nvo_to_vvo} | BVO→VVO={bvo_to_vvo})")
    st.write(f"- Factors: scope={scope}, quality={quality}, region={region}, complexity={complexity}, MEP_incl={include_mep}, MEP_zwaarte={mep_intensity}, PM={include_pm}")
    st.write(f"- CPI-index gebruikt (base {BASE_YEAR}): {cpi_state if cpi_edit else CPI_INDEX}")
    if project_type == "office":
        st.write(f"- Programma: werkplekken {wp_pct}%, vergaderruimtes {mtg_pct}%, support {sup_pct}%")
        st.write(f"- Meeting add-on CapEx: €{capex_meeting_extra:,.0f} • Onvoorzien: {contingency_pct}% toegepast op CapEx")
        st.write(f"- OpEx jaar 1 totaal: €{annual_opex_total:,.0f} (huur: €{annual_rent:,.0f} • facilitair: €{annual_opex_facility:,.0f})")

# Empirische matches tabel
if ep[3] > 0:
    with st.expander(f"🔎 Empirische matches (n={ep[3]})"):
        q = df.copy()
        for col, val in [("project_type", project_type), ("scope", scope), ("quality", quality), ("region", region)]:
            q = q[q[col] == val]
        q = q.assign(per_m2=q.apply(per_m2_from_record, axis=1))
        st.dataframe(q)

st.divider()

# ------------------------------
# Add Actuals
# ------------------------------
st.subheader("➕ Werkelijk besteed toevoegen (anoniem)")
st.caption("Voeg afgeronde projecten toe om de benchmarks slimmer te maken. Je kunt de geüpdatete CSV direct downloaden.")

with st.form("add_actuals"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        a_project_id = st.text_input("Project ID", value="NEW-001")
        a_project_type = st.selectbox("Type", ["office", "hospitality", "retail", "residential"], key="a_type")
        a_scope = st.selectbox("Scope", ["both", "FF&E", "afbouw"], key="a_scope")
    with c2:
        a_quality = st.selectbox("Kwaliteit", ["basic", "mid", "high"], key="a_quality")
        a_region = st.selectbox("Regio", ["NL-Randstad", "NL-overig", "EU-noord", "EU-zuid"], key="a_region")
        a_year = st.number_input("Jaar", min_value=2021, max_value=2030, value=2025, step=1, key="a_year")
    with c3:
        a_gross = st.number_input("Gross m²", min_value=0.0, value=1000.0, step=10.0, key="a_gross")
        a_net = st.number_input("Net m² (optioneel)", min_value=0.0, value=0.0, step=10.0, key="a_net")
        a_cost = st.number_input("Totaal (excl. btw)", min_value=0.0, value=500000.0, step=1000.0, key="a_cost")
    with c4:
        a_mep = st.checkbox("Incl. MEP?", value=True, key="a_mep")
        a_pm = st.checkbox("Incl. PM & logistiek?", value=True, key="a_pm")
        a_complexity = st.selectbox("Complexiteit", ["standard", "maatwerk", "monument"], key="a_complexity")
    a_notes = st.text_area("Notities (optioneel)", value="")

    submitted = st.form_submit_button("Toevoegen aan dataset")
    if submitted:
        new_row = {
            "project_id": a_project_id,
            "project_type": a_project_type,
            "scope": a_scope,
            "quality": a_quality,
            "region": a_region,
            "year": int(a_year),
            "gross_m2": float(a_gross),
            "net_m2": float(a_net) if a_net > 0 else None,
            "total_cost_excl_vat": float(a_cost),
            "include_mep": bool(a_mep),
            "include_pm_logistics": bool(a_pm),
            "complexity": a_complexity,
            "notes": a_notes,
        }
        df_new = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        csv_bytes = df_new.to_csv(index=False).encode("utf-8")
        st.success("Record toegevoegd! Download hieronder de geüpdatete CSV.")
        st.download_button("⬇️ Download geüpdatete dataset", data=csv_bytes, file_name="m2_budget_checker_dataset_updated.csv", mime="text/csv")

st.divider()
st.caption("Tip: pas de baselines en factoren in de code aan voor jouw markt.")

st.write("— M² Budget Checker MVP • © 2025")
