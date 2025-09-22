# ============================================
# app_v12.py — Notion-look + Good/Better/Best + Vergelijking + Export (CSV/PDF)
# ============================================

import streamlit as st
import pandas as pd
import pulp
import io
import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.pdfgen import canvas as rl_canvas
from pypdf import PdfReader, PdfWriter

# ----------------- Page settings & clean Notion-like CSS -----------------
st.set_page_config(page_title="The Studio - Budget Optimizer Voor Interieur Professionals", page_icon="🧾", layout="wide")

st.markdown("""
<style id="final-override">
/* Inter font import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* Reset */
html, body, .stApp {
  background: #ffffff !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  color: #0f172a !important;
}

/* Kopteksten */
h1, h2, h3, h4 {
  font-size: 28px !important;
  font-weight: 600 !important;
  line-height: 1.3 !important;
  margin-bottom: .6rem !important;
}

/* Body tekst */
.block-container,
[data-testid="stAppViewContainer"],
[data-testid="stSidebarContent"],
[data-testid="stMarkdownContainer"],
[data-testid="stDataFrame"] table,
.stTextInput input, .stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stRadio label, .stCheckbox label,
.stSlider, .stFileUploader,
div.stButton > button, .stDownloadButton button,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"],
label, p, span, li, a, legend, small, code, pre {
  font-size: 16px !important;
  line-height: 1.5 !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  font-weight: 400 !important;
}

/* Tabelkop iets zwaarder */
.dataframe th { font-weight: 600 !important; }

/* Tabelcellen luchtiger */
div[data-testid="stDataFrame"] th,
div[data-testid="stDataFrame"] td { padding: 6px 8px !important; }
            
/* Tekst onder uploader rood */
div[data-testid="stFileUploaderDropzone"] p {
    color: #dc2626 !important;   /* alleen hier rood */
    font-size: 15px !important;
    font-weight: 500 !important;
}

/* Knoppen zwart/wit */
div.stButton > button, .stDownloadButton button {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
}
div.stButton > button:hover, .stDownloadButton button:hover {
    background-color: #1e293b !important;
}

/* Radio & checkboxes donker */
.stRadio label, .stCheckbox label {
    color: #0f172a !important;
    font-weight: 400 !important;
}

/* Spacing bij labels/uploader */
.step-label p { margin-bottom: 0.5rem !important; }
.step-label + div[data-testid="stFileUploader"] { margin-top: 0 !important; }


</style>
""", unsafe_allow_html=True)


# ----------------- PDF CONST -----------------
PAGE_SIZE = A4
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 20*mm, 20*mm, 50*mm, 18*mm

# ----------------- Utils -----------------
def euro(x, decimals=2):
    try:
        return f"€ {float(x):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "€ 0,00"

def int_to_eu(n: int) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return "0"

def eu_to_int(s: str) -> int:
    if s is None:
        return 0
    s = str(s).strip()
    if s == "":
        return 0
    s = s.replace(".", "").replace(" ", "")
    s = s.replace(",", ".")
    try:
        return int(float(s))
    except Exception:
        return 0

def pct_to_float(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    s = str(v).strip().replace("%", "").replace(",", ".")
    try:
        val = float(s)
        return val/100.0 if val > 1.0 else val
    except Exception:
        return 0.0

def format_date(d=None):
    if d is None:
        d = datetime.date.today()
    return d.strftime("%d-%m-%Y")

# ----------------- Editor helpers -----------------
def _normalize_editor(df: pd.DataFrame, zero_inactive: bool = False) -> pd.DataFrame:
    df = df.copy()
    defaults = {"soort":"", "actief":False, "aantal":None, "min_klasse":None, "max_klasse":None, "niet_mixen":False}
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v
    df["soort"] = df["soort"].astype(str)
    df["actief"] = df["actief"].astype(bool)
    df["aantal"] = pd.to_numeric(df["aantal"], errors="coerce")
    df["min_klasse"] = pd.to_numeric(df["min_klasse"], errors="coerce")
    df["max_klasse"] = pd.to_numeric(df["max_klasse"], errors="coerce")
    df["niet_mixen"] = df["niet_mixen"].astype(bool)
    if zero_inactive:
        df.loc[~df["actief"], "aantal"] = 0
    return df

def _to_int_safe(v, default=0):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return int(float(v))
    except Exception:
        return default

def _to_float_safe(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return float(v)
    except Exception:
        return default

# ----------------- PDF helpers -----------------
def draw_letterhead_image(c: rl_canvas.Canvas, letterhead_img, page_size=PAGE_SIZE):
    try:
        w, h = page_size
        c.drawImage(letterhead_img, 0, 0, width=w, height=h, preserveAspectRatio=False, mask='auto')
    except Exception:
        pass

def on_page(letterhead_img=None, footer_text=""):
    def _cb(c, doc):
        if letterhead_img:
            draw_letterhead_image(c, letterhead_img, PAGE_SIZE)
        c.setFont("Helvetica", 8)
        c.setFillColorRGB(0.45, 0.48, 0.55)
        c.drawRightString(PAGE_SIZE[0]-MARGIN_R, 10*mm, f"{footer_text}  •  pagina {c.getPageNumber()}")
    return _cb

def build_offer_pdf(
    project_meta: dict,
    result_df: "pd.DataFrame",
    summary_df: "pd.DataFrame",
    show_prices: bool = True,
    letterhead_image_bytes: bytes | None = None,
    internal: bool = False
) -> bytes:
    buf = io.BytesIO()
    top_margin_value = MARGIN_T
    if letterhead_image_bytes:
        top_margin_value = 100
    doc = SimpleDocTemplate(
        buf, pagesize=PAGE_SIZE,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=top_margin_value, bottomMargin=MARGIN_B
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, leading=22, spaceAfter=6))
    styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=16, spaceAfter=4))
    styles.add(ParagraphStyle("Meta", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14))

    story = []
    titel = project_meta.get("title") or ("Rapport (intern)" if internal else "Offerte")
    story.append(Paragraph(titel, styles["H1"]))
    meta_bits = []
    for key, label in [
        ("client","Klant"), ("reference","Referentie"), ("date_str","Datum"),
        ("mode_str","Doel"), ("discount_client_str","Klantkorting"), ("budget_str","Budget"),
    ]:
        val = project_meta.get(key)
        if val:
            meta_bits.append(f"<b>{label}</b>: {val}")
    if meta_bits:
        story.append(Paragraph(" &nbsp; • &nbsp; ".join(meta_bits), styles["Meta"]))
    story.append(Spacer(1, 8))

    # Samenvatting
    story.append(Paragraph("Samenvatting per soort", styles["H2"]))
    summary_df = summary_df.copy()
    if internal:
        sum_cols = ["soort","totaal_stuks","omzet","inkoop","marge","marge_pct","gem_klasse_per_soort"]
        sum_headers = ["Soort","Stuks","Omzet","Inkoop","Marge","Marge %","Gem. klasse"]
    else:
        sum_cols = ["soort","totaal_stuks","omzet","gem_klasse_per_soort"]
        sum_headers = ["Soort","Stuks","Omzet","Gem. klasse"]
    for col in sum_cols:
        if col not in summary_df.columns:
            summary_df[col] = ""
    sum_tbl = [sum_headers] + summary_df[sum_cols].astype(str).values.tolist()
    t = Table(sum_tbl, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 10),
        ("FONT", (0,1), (-1,-1), "Helvetica", 9),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#0F172A")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#E2E8F0")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FCFDFF")]),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Details
    story.append(Paragraph("Details", styles["H2"]))
    detail = result_df.copy()

    def _eur(v):
        try:
            return f"€ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(v)

    if "Brutoprijs" not in detail.columns and "prijs" in detail.columns:
        detail["Brutoprijs"] = detail["prijs"].apply(_eur)
    if "Verkoop/klant" not in detail.columns and "price_effective" in detail.columns:
        detail["Verkoop/klant"] = detail["price_effective"].apply(_eur)
    if "Inkoop/netto" not in detail.columns and "price_buy_unit" in detail.columns:
        detail["Inkoop/netto"] = detail["price_buy_unit"].apply(_eur)
    if "Subtotaal klant" not in detail.columns and {"aantal","price_effective"}.issubset(detail.columns):
        detail["Subtotaal klant"] = (detail["aantal"] * detail["price_effective"]).apply(_eur)
    if "Subtotaal inkoop" not in detail.columns and {"aantal","price_buy_unit"}.issubset(detail.columns):
        detail["Subtotaal inkoop"] = (detail["aantal"] * detail["price_buy_unit"]).apply(_eur)

    if internal:
        det_cols = ["soort","artikel","merk","klasse","aantal",
                    "Brutoprijs","Verkoop/klant","Inkoop/netto","Subtotaal klant","Subtotaal inkoop"]
        det_headers = ["Soort","Artikel","Merk","Kl.","Aantal","Bruto","Verkoop","Inkoop","Subtot. klant","Subtot. inkoop"]
    else:
        det_cols = ["soort","artikel","merk","klasse","aantal","Verkoop/klant","Subtotaal klant"]
        det_headers = ["Soort","Artikel","Merk","Kl.","Aantal","Prijs/stuk","Subtotaal"]

    for col in det_cols:
        if col not in detail.columns:
            detail[col] = ""
    det_tbl = [det_headers] + detail[det_cols].astype(str).values.tolist()
    td = Table(det_tbl, hAlign="LEFT", repeatRows=1)
    td.setStyle(TableStyle([
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("FONT", (0,1), (-1,-1), "Helvetica", 8),
        ("ALIGN", (len(det_headers)-2,1), (-1,-1), "RIGHT"),
        ("ALIGN", (0,0), (2,-1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#E2E8F0")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FCFDFF")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(td)

    letterhead_img = io.BytesIO(letterhead_image_bytes) if letterhead_image_bytes else None
    footer = project_meta.get("footer", "Gegenereerd met Quote Optimizer")
    doc.build(story, onFirstPage=on_page(letterhead_img, footer), onLaterPages=on_page(letterhead_img, footer))
    return buf.getvalue()

def overlay_pdf_letterhead(base_pdf_bytes: bytes, letterhead_pdf_bytes: bytes, first_page_only=True) -> bytes:
    base_reader = PdfReader(io.BytesIO(base_pdf_bytes))
    letter_reader = PdfReader(io.BytesIO(letterhead_pdf_bytes))
    out = PdfWriter()
    overlay_page = letter_reader.pages[0]
    for i, page in enumerate(base_reader.pages):
        new_page = page
        if (not first_page_only) or (first_page_only and i == 0):
            new_page.merge_page(overlay_page)
        out.add_page(new_page)
    out_buf = io.BytesIO()
    out.write(out_buf)
    return out_buf.getvalue()

# ----------------- Solver helpers -----------------
def _build_common(df, needs):
    df = df.copy()
    df.columns = df.columns.str.lower()
    price_col = "price_effective" if "price_effective" in df.columns else "prijs"
    types = sorted(df["soort"].astype(str).unique().tolist())
    idx_by_type = {s: [i for i in range(len(df)) if str(df.loc[i,"soort"]) == s] for s in types}
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(len(df))]
    y = [None]*len(df)
    price_sum = pulp.lpSum(x[i] * float(df.loc[i, price_col]) for i in range(len(df)))
    class_sum = pulp.lpSum(x[i] * float(df.loc[i, "klasse"]) for i in range(len(df)))
    return df, price_col, types, idx_by_type, x, y, price_sum, class_sum

def _add_core_constraints(model, df, needs, idx_by_type, x, y,
                          min_class_by_type, max_class_by_type, no_mix_by_type):
    for s, qty in needs.items():
        if int(qty) <= 0:
            continue
        idx = idx_by_type.get(s, [])
        model += pulp.lpSum(x[i] for i in idx) == int(qty)
    min_class_by_type = min_class_by_type or {}
    max_class_by_type = max_class_by_type or {}
    for s, qty in needs.items():
        if int(qty) <= 0:
            continue
        idx = idx_by_type.get(s, [])
        mn = float(min_class_by_type.get(s, 0.0) or 0.0)
        mx = float(max_class_by_type.get(s, 0.0) or 0.0)
        if mn > 0:
            model += pulp.lpSum(x[i] * float(df.loc[i,"klasse"]) for i in idx) >= mn * int(qty)
        if mx > 0:
            model += pulp.lpSum(x[i] * float(df.loc[i,"klasse"]) for i in idx) <= mx * int(qty)
    no_mix_by_type = no_mix_by_type or {}
    for s, qty in needs.items():
        if int(qty) <= 0:
            continue
        if no_mix_by_type.get(s, False):
            idx = idx_by_type.get(s, [])
            for i in idx:
                if y[i] is None:
                    y[i] = pulp.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat="Binary")
            model += pulp.lpSum(y[i] for i in idx) == 1
            for i in idx:
                model += x[i] <= int(qty) * y[i]

def solve_min_cost_with_quality(
    df, needs, min_avg,
    min_class_by_type=None, max_class_by_type=None, no_mix_by_type=None,
    budget_cap=None
):
    df = df.copy()
    df.columns = df.columns.str.lower()
    required = {"artikel","soort","merk","prijs","klasse"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Ontbrekende kolommen: {missing}")
    active_types = [s for s,q in needs.items() if int(q) > 0]
    if not active_types:
        raise ValueError("Geen aantallen ingevuld (>0).")
    df = df[df["soort"].isin(active_types)].reset_index(drop=True)

    N = int(sum(needs.values()))
    df, price_col, types, idx_by_type, x, y, price_sum, class_sum = _build_common(df, needs)

    model = pulp.LpProblem("min_cost_quality", pulp.LpMinimize)
    model += price_sum
    _add_core_constraints(model, df, needs, idx_by_type, x, y,
                          min_class_by_type, max_class_by_type, no_mix_by_type)
    if min_avg is not None and min_avg > 0:
        model += class_sum >= float(min_avg) * N
    if budget_cap is not None and budget_cap > 0:
        model += price_sum <= float(budget_cap)

    res = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[res] != "Optimal":
        raise RuntimeError(f"Geen optimale oplossing (min_cost_with_quality), status: {pulp.LpStatus[res]}")

    qtys = [int(v.value()) for v in x]
    df["aantal"] = qtys
    df = df[df["aantal"] > 0].copy()
    df["omzet_subtotaal"]  = df["aantal"] * (df["price_effective"] if "price_effective" in df.columns else df["prijs"])
    df["inkoop_subtotaal"] = df["aantal"] * (df["price_buy_unit"] if "price_buy_unit" in df.columns else 0.0)
    total_price = float(df["omzet_subtotaal"].sum())
    avg_class = float((df["aantal"] * df["klasse"]).sum() / N)
    return df, total_price, avg_class

def solve_max_quality_under_budget(
    df, needs, budget,
    min_class_by_type=None, max_class_by_type=None, no_mix_by_type=None
):
    df = df.copy()
    df.columns = df.columns.str.lower()
    required = {"artikel","soort","merk","prijs","klasse"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Ontbrekende kolommen: {missing}")
    active_types = [s for s,q in needs.items() if int(q) > 0]
    if not active_types:
        raise ValueError("Geen aantallen ingevuld (>0).")
    df = df[df["soort"].isin(active_types)].reset_index(drop=True)

    N = int(sum(needs.values()))
    df, price_col, types, idx_by_type, x, y, price_sum, class_sum = _build_common(df, needs)

    model = pulp.LpProblem("max_quality_budget", pulp.LpMaximize)
    model += class_sum
    _add_core_constraints(model, df, needs, idx_by_type, x, y,
                          min_class_by_type, max_class_by_type, no_mix_by_type)
    model += price_sum <= float(budget)

    res = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[res] != "Optimal":
        raise RuntimeError(f"Geen optimale oplossing (max_quality_budget), status: {pulp.LpStatus[res]}")

    qtys = [int(v.value()) for v in x]
    df["aantal"] = qtys
    df = df[df["aantal"] > 0].copy()
    df["omzet_subtotaal"]  = df["aantal"] * (df["price_effective"] if "price_effective" in df.columns else df["prijs"])
    df["inkoop_subtotaal"] = df["aantal"] * (df["price_buy_unit"] if "price_buy_unit" in df.columns else 0.0)
    total_price = float(df["omzet_subtotaal"].sum())
    avg_class = float((df["aantal"] * df["klasse"]).sum() / N)
    return df, total_price, avg_class

# ----------------- UI -----------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:1rem;">
        <img src="https://raw.githubusercontent.com/jeanpaulpeek/quote-optimizer/refs/heads/main/The_Studio_Logo.png"
             alt="The Studio Logo"
             style="height:96px;">
          </div>
    """,
    unsafe_allow_html=True
)

# Introblok met quote-style en divider
st.markdown(
    """
    <div style="
        width:60%;
        margin-bottom:1rem;
        padding:0.8rem 1.2rem;
        border-left:4px solid #0f172a;
        background-color:#fafafa;
    ">
        <p style="margin:0; font-size:16px; line-height:1.5; color:#334155;">
        The Studio Quote Optimizer is een tool die automatisch de kwalitatief beste combinatie interieurproducten maakt binnen het gestelde budget én/of de beste combinatie binnen een gestelde kwalitateitseis.
        </p>
    </div>
    <hr style="border:0; border-top:1px solid #e5e7eb; margin:1.5rem 0;">
    """,
    unsafe_allow_html=True
)

# === START: CLEAN STAP 1 & 2 BLOK ===

# 0) Budget-callback
def _format_budget_cb():
    raw = st.session_state.get("budget_input_str", "")
    n = eu_to_int(raw)
    st.session_state["budget_input_str"] = int_to_eu(n)
    st.session_state["budget_value"] = n

# 1) STAP 1 — bron kiezen (demo of upload)
st.markdown(
    '<div class="step-label"><strong>Stap 1. Upload prijslijst (.xlsx) of gebruik de demo-dataset.</strong></div>',
    unsafe_allow_html=True
)

use_demo = st.toggle(
    "Demo-dataset gebruiken",
    value=False,
    help="Schakel in om met de ingebouwde demo-prijslijst te werken.",
    key="use_demo_toggle",
)

DEMO_XLSX_PATH = Path(__file__).with_name("Demo_prijslijst.xlsx")

@st.cache_data(show_spinner=False)
def _load_demo_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.lower()
    df["prijs"]  = pd.to_numeric(df["prijs"], errors="coerce")
    df["klasse"] = pd.to_numeric(df["klasse"], errors="coerce")
    if "korting_leverancier" in df.columns:
        df["korting_leverancier"] = pd.to_numeric(df["korting_leverancier"], errors="coerce")
        if df["korting_leverancier"].dropna().gt(1).any():
            df["korting_leverancier"] /= 100.0
    else:
        df["korting_leverancier"] = 0.0
    return df

# Zorg dat 'uploaded' altijd bestaat
uploaded = None

if use_demo:
    try:
        df_raw = _load_demo_df(DEMO_XLSX_PATH)
    except Exception as e:
        st.error(f"Kon demo-dataset niet laden: {e}")
        st.stop()

    init_rows = [
        {"soort": s, "actief": False, "aantal": 0,
         "min_klasse": None, "max_klasse": None, "niet_mixen": False}
        for s in sorted(pd.Series(df_raw["soort"]).dropna().unique().tolist())
    ]

else:
    uploaded = st.file_uploader("Upload prijslijst (.xlsx)", type=["xlsx"], key="uploader_main")

    if uploaded is None:
        st.warning("Upload een prijslijst (.xlsx) om te starten of schakel de demo-dataset in.")
        try:
            with open("Prijslijst_Template.xlsx", "rb") as f:
                st.download_button(
                    label="⬇Download prijslijst template",
                    data=f,
                    file_name="Prijslijst_Template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_template_btn"
                )
        except FileNotFoundError:
            st.info("📄 Template-bestand ontbreekt in de repo (root). Voeg 'Prijslijst_Template.xlsx' toe.")
        st.stop()

    # ===== Excel inladen + normaliseren =====
    df_raw = pd.read_excel(uploaded)
    df_raw.columns = df_raw.columns.str.lower()

    needed = {"artikel","soort","merk","prijs","klasse"}  # korting optioneel
    missing = needed - set(df_raw.columns)
    if missing:
        st.error("Je Excel mist één of meer kolommen: " + ", ".join(sorted(missing)))
        st.stop()

    # types naar numeriek waar nodig
    df_raw["prijs"]  = pd.to_numeric(df_raw["prijs"], errors="coerce")
    df_raw["klasse"] = pd.to_numeric(df_raw["klasse"], errors="coerce")
    if "korting_leverancier" in df_raw.columns:
        df_raw["korting_leverancier"] = pd.to_numeric(df_raw["korting_leverancier"], errors="coerce")
        # Als gebruiker 10 of 35 invult i.p.v. 0.10 / 0.35 → normaliseer
        if df_raw["korting_leverancier"].dropna().gt(1).any():
            df_raw["korting_leverancier"] = df_raw["korting_leverancier"] / 100.0
    else:
        df_raw["korting_leverancier"] = 0.0

    # ===== HIER: init_rows ZEKER ZETTEN =====
    init_rows = [
        {"soort": s, "actief": False, "aantal": 0,
         "min_klasse": None, "max_klasse": None, "niet_mixen": False}
        for s in sorted(pd.Series(df_raw["soort"]).dropna().unique().tolist())
    ]


st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)


# 2) STAP 2 — Doelkeuze
st.markdown(
    '<div class="step-label"><strong>Stap 2. Kies optimalisatie-doel:</strong></div>',
    unsafe_allow_html=True
)
DOEL_KLASSE = "Minimale prijs bij de gekozen kwaliteitsdrempel"
DOEL_BUDGET = "Maximale kwaliteit binnen budget"
doel = st.radio(
    " ",
    [DOEL_KLASSE, DOEL_BUDGET],
    index=(1 if use_demo else 0),
    label_visibility="collapsed",
    key="doel_radio"
)
# Context-hint bij doelkeuze
if doel == DOEL_KLASSE:
    st.caption("Budgetvelden zijn niet van toepassing bij dit doel: de optimizer zoekt de laagste prijs die aan de klasse-eis voldoet.")
else:
    st.caption("Vul een budget in; de optimizer maximaliseert de kwaliteit binnen dat budget.")


# 3) STAP 3 — Klantkorting
st.markdown(
    '<div class="step-label"><strong>Stap 3. Geef aan hoeveel procent korting de klant ontvangt:</strong></div>',
    unsafe_allow_html=True
)
col1, col2 = st.columns([1, 2])
with col1:
    klant_korting_pct = st.number_input(
        "De klant ontvangt een kortingspercentage van",
        min_value=0.0, max_value=100.0, value=0.0, step=1.0,
        key="klant_korting_pct_input"
    )
with col2:
    st.caption("")

# 4) COMMON — df_work opbouwen (geldt voor demo én upload)
df_work = df_raw.copy()
df_work.columns = df_work.columns.str.lower()
if "korting_leverancier" in df_work.columns:
    df_work["korting_leverancier"] = df_work["korting_leverancier"].apply(pct_to_float)
else:
    df_work["korting_leverancier"] = 0.0
klant_korting = pct_to_float(klant_korting_pct)
df_work["price_buy_unit"]  = df_work["prijs"] * (1 - df_work["korting_leverancier"])
df_work["price_effective"] = df_work["prijs"] * (1 - klant_korting)
if st.session_state.get("df_work") is None:
    st.session_state["df_work"] = df_work

# 5) Doel-specifieke invoer
target_avg, budget, budget_basis = None, None, None
if doel == DOEL_KLASSE:
    target_avg = st.slider("Doelklasse (gemiddeld)", min_value=1.0, max_value=5.0, value=2.7, step=0.05, key="klasse_slider")
else:
    default_budget = 18000 if use_demo else 100000
    if "budget_input_str" not in st.session_state:
        st.session_state["budget_input_str"] = int_to_eu(default_budget)
    if "budget_value" not in st.session_state:
        st.session_state["budget_value"] = eu_to_int(st.session_state["budget_input_str"])
    st.text_input(
        "Budget (excl. btw)",
        key="budget_input_str",
        on_change=_format_budget_cb,
        help="Gebruik punten voor duizendtallen, bijv. 100.000"
    )
    budget = st.session_state["budget_value"]
    budget_basis = st.radio(
        "Budget is opgegeven als",
        ["Netto voor klant (na korting)", "Bruto advies (vóór korting)"],
        index=0, horizontal=True, key="budget_basis_radio"
    )

st.markdown("---")

# 6) Editor per soort
types_key = tuple(sorted({r["soort"] for r in init_rows}))
if "editor_types" not in st.session_state or st.session_state["editor_types"] != types_key:
    st.session_state["editor_types"] = types_key
    df_init = pd.DataFrame(init_rows).assign(actief=False)  # forceer uit
    st.session_state["editor_df"] = df_init

st.markdown("### Invoer per soort")
with st.form("opt_form", clear_on_submit=False):
    edited_raw = st.data_editor(
        st.session_state["editor_df"],
        key="editor_df_widget",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config={
            "soort": st.column_config.TextColumn("Soort", disabled=True),
            "actief": st.column_config.CheckboxColumn("Actief"),
            "aantal": st.column_config.NumberColumn("Aantal", min_value=None, step=1),
            "min_klasse": st.column_config.NumberColumn("Min. klasse", min_value=None, step=0.1),
            "max_klasse": st.column_config.NumberColumn("Max. klasse (0=geen)", min_value=None, step=0.1),
            "niet_mixen": st.column_config.CheckboxColumn("Niet mixen"),
        },
    )
    submitted = st.form_submit_button("Optimaliseren", use_container_width=False, type="primary", help="Bereken Good/Better/Best", disabled=False)

# 7) OPTIMALISEREN
if submitted:
    prev = _normalize_editor(st.session_state["editor_df"], zero_inactive=False)
    new = _normalize_editor(edited_raw, zero_inactive=False)
    for col in ["aantal", "min_klasse", "max_klasse"]:
        if col in new.columns and col in prev.columns:
            mask = new[col].isna()
            new.loc[mask, col] = prev.loc[mask, col]
    new["aantal"] = new["aantal"].fillna(0)
    new["min_klasse"] = new["min_klasse"].fillna(0.0)
    new["max_klasse"] = new["max_klasse"].fillna(0.0)
    # Auto-activeren: aantal > 0 ⇒ actief
    new["actief"] = new["actief"] | (new["aantal"].fillna(0) > 0)
    st.session_state["editor_df"] = new.copy()

    ed = new.copy()
    needs = {row["soort"]: (_to_int_safe(row["aantal"]) if (bool(row["actief"]) or _to_int_safe(row["aantal"]) > 0) else 0) for _, row in ed.iterrows()}
    per_type_min = {row["soort"]: _to_float_safe(row["min_klasse"]) for _, row in ed.iterrows()}
    per_type_max = {row["soort"]: _to_float_safe(row["max_klasse"]) for _, row in ed.iterrows()}
    no_mix_by_type = {row["soort"]: bool(row["niet_mixen"]) for _, row in ed.iterrows()}

    problems = []
    N_total = sum(needs.values())
    if N_total <= 0:
        problems.append("Geen aantallen ingevuld (>0).")

    if doel == DOEL_BUDGET:
        budget_net = budget if budget_basis.startswith("Netto") else budget * (1 - klant_korting)
        if budget_net is None or budget_net <= 0:
            problems.append("Geen (geldig) budget ingevuld.")
    else:
        budget_net = None

    if problems:
        st.error("Corrigeer de volgende punten:")
        for p in problems:
            st.write("• " + p)
        st.stop()

    options = []
    try:
        if doel == DOEL_BUDGET:
            best_df, best_total, best_avg = solve_max_quality_under_budget(
                df_work, needs, budget_net,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )
            options.append({"name": "BEST • Max kwaliteit", "result": best_df, "total": best_total, "avg": best_avg})

            thr_better = max(1.0, best_avg - 0.05)
            try:
                b_df, b_total, b_avg = solve_min_cost_with_quality(
                    df_work, needs, thr_better,
                    min_class_by_type=per_type_min, max_class_by_type=per_type_max,
                    no_mix_by_type=no_mix_by_type, budget_cap=budget_net
                )
                options.append({"name": "BETTER • Zuinig (−0,05 klasse)", "result": b_df, "total": b_total, "avg": b_avg})
            except Exception:
                pass

            thr_good = max(1.0, best_avg - 0.10)
            try:
                g_df, g_total, g_avg = solve_min_cost_with_quality(
                    df_work, needs, thr_good,
                    min_class_by_type=per_type_min, max_class_by_type=per_type_max,
                    no_mix_by_type=no_mix_by_type, budget_cap=budget_net
                )
                options.append({"name": "GOOD • Budgetvriendelijk (−0,10 klasse)", "result": g_df, "total": g_total, "avg": g_avg})
            except Exception:
                pass

            options = sorted(options, key=lambda d: d["avg"], reverse=True)
            badge = f"Kwaliteit binnen budget ({euro(budget_net, 0)})"
            st.session_state["last_budget_shown"] = budget_net

        else:
            g_df, g_total, g_avg = solve_min_cost_with_quality(
                df_work, needs, target_avg,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )
            options.append({"name": "GOOD • Min prijs (voldoet)", "result": g_df, "total": g_total, "avg": g_avg})

            try:
                b_df, b_total, b_avg = solve_min_cost_with_quality(
                    df_work, needs, target_avg + 0.05,
                    min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
                )
                options.append({"name": "BETTER • Klasse +0,05", "result": b_df, "total": b_total, "avg": b_avg})
            except Exception:
                pass

            try:
                best_df, best_total, best_avg = solve_min_cost_with_quality(
                    df_work, needs, target_avg + 0.10,
                    min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
                )
                options.append({"name": "BEST • Klasse +0,10", "result": best_df, "total": best_total, "avg": best_avg})
            except Exception:
                pass

            options = sorted(options, key=lambda d: d["avg"], reverse=True)
            badge = f"Min. prijs bij klasse ≥ {target_avg:.2f}".replace(".", ",")

        # Bewaar state voor resultaatsectie
        st.session_state["gbb_options"] = options
        st.session_state["badge"] = badge
        st.session_state["doel"] = doel
        st.session_state["klant_korting_value"] = float(klant_korting_pct)
        st.session_state["df_work"] = df_work
        st.session_state["needs"] = needs
        st.session_state["per_type_min"] = per_type_min
        st.session_state["per_type_max"] = per_type_max
        st.session_state["no_mix_by_type"] = no_mix_by_type
        st.session_state["target_avg"] = target_avg
        st.session_state["last_budget_shown"] = st.session_state.get("last_budget_shown", None)

        st.success("Opties berekend. Je kunt nu hieronder vergelijken en exporteren.")
        st.rerun()

    except Exception as e:
        st.error(str(e))

# === EINDE: CLEAN STAP 1 & 2 BLOK ===

# ===================== B) RESULTATEN / VERGELIJKING / EXPORT =====================
if "gbb_options" in st.session_state and st.session_state["gbb_options"]:
    options = st.session_state["gbb_options"]
    badge = st.session_state.get("badge", "")
    doel_state = st.session_state.get("doel", "")
    klant_korting_pct = st.session_state.get("klant_korting_value", 0.0)

    df_work_state = st.session_state.get("df_work", None)
    df_work = df_work_state if df_work_state is not None else None
    needs = st.session_state.get("needs", {})
    per_type_min = st.session_state.get("per_type_min", {})
    per_type_max = st.session_state.get("per_type_max", {})
    no_mix_by_type = st.session_state.get("no_mix_by_type", {})
    target_avg_state = st.session_state.get("target_avg", None)
    last_budget_shown = st.session_state.get("last_budget_shown", None)

    if df_work is None:
        st.warning("Interne staat mist de werk-DF. Vul aantallen in en klik ‘Optimaliseren’ opnieuw.")
        st.stop()

    st.subheader("✨ Resultaten (3 opties)")
    ref_avg = options[0]["avg"]
    ref_total = options[0]["total"]
    for opt in options:
        delta_eur = opt["total"] - ref_total
        delta_cls = opt["avg"] - ref_avg
        d_eur = ("+" if delta_eur > 0 else "") + euro(delta_eur, 0) if abs(delta_eur) > 1e-6 else "± € 0"
        d_cls = ("+" if delta_cls > 0 else "") + f"{delta_cls:.2f}".replace(".", ",")
        st.markdown(
            f"**{opt['name']}**  \n"
            f"Prijs: {euro(opt['total'], 0)}  •  Gem. klasse: {opt['avg']:.2f}".replace(".", ",") + "  \n"
            f"_Δ t.o.v. BEST_: {d_eur} • {d_cls}"
        )
    st.caption(f"Doel: {badge}")

    st.markdown("---")
    cmp_on = st.checkbox(
        "🔁 Vergelijking (Good vs Best)",
        value=False,
        help="Toon naast elkaar: goedkoopste die aan klasse voldoet (Good) vs maximale kwaliteit binnen budget (Best).",
        key="compare_checkbox"
    )

    if cmp_on:
        if doel_state == DOEL_BUDGET:
            best_opt = next((o for o in options if o["name"].startswith("BEST")), options[0])
            best_avg = float(best_opt["avg"])
            thr_good = max(1.0, best_avg - 0.10)
            budget_net = last_budget_shown

            good_df, good_total, good_avg = solve_min_cost_with_quality(
                df_work, needs, thr_good,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max,
                no_mix_by_type=no_mix_by_type, budget_cap=budget_net
            )
            better_df, better_total, better_avg = solve_max_quality_under_budget(
                df_work, needs, budget_net,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### **Good** — Min. prijs bij vaste klasse-eis")
                st.metric("Totaal (klant)", euro(good_total, 0))
                st.metric("Gem. klasse", f"{good_avg:.2f}".replace(".", ","))
            with c2:
                st.markdown("#### **Best** — Max. kwaliteit binnen budget")
                st.metric("Totaal (klant)", euro(better_total, 0))
                st.metric("Gem. klasse", f"{better_avg:.2f}".replace(".", ","))

            st.caption(
                f"Vergelijking binnen hetzelfde budget ({euro(budget_net,0)}). "
                f"Klasse-eis Good = BEST − 0,10 = {thr_good:.2f}".replace(".", ",")
            )
        else:
            cmp_budget = st.number_input(
                "Vergelijking – budget voor Best (netto, na klantkorting)",
                min_value=0, value=int(last_budget_shown or 100000), step=1000, format="%i",
                help="Best = maximale kwaliteit binnen dit budget.",
                key="cmp_budget_input"
            )
            good_df, good_total, good_avg = solve_min_cost_with_quality(
                df_work, needs, float(target_avg_state or 2.7),
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )
            better_df, better_total, better_avg = solve_max_quality_under_budget(
                df_work, needs, cmp_budget,
                min_class_by_type=per_type_min, max_class_by_type=per_type_max, no_mix_by_type=no_mix_by_type
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### **Good** — Min. prijs die aan de klasse-eis voldoet")
                st.metric("Totaal (klant)", euro(good_total, 0))
                st.metric("Gem. klasse", f"{good_avg:.2f}".replace(".", ","))
            with c2:
                st.markdown("#### **Best** — Max. kwaliteit binnen budget")
                st.metric("Totaal (klant)", euro(better_total, 0))
                st.metric("Gem. klasse", f"{better_avg:.2f}".replace(".", ","))
            # Extra toelichting specifiek voor DOEL_KLASSE
            st.caption("Let op: 'Good' negeert budget en voldoet minimaal aan jouw klasse-eis; 'Best' gebruikt het opgegeven budget.")

            st.caption(
                f"Verschil (Best − Good): "
                f"{'+' if (better_total - good_total) > 0 else ''}{euro(better_total - good_total, 0)} • "
                f"{'+' if (better_avg - good_avg) > 0 else ''}{(better_avg - good_avg):.2f}".replace(".", ",")
            )

    # ======= EXPORT =======
    st.markdown("---")
    st.subheader("Exporteer / Download")

    names = [o["name"] for o in options]
    choice = st.selectbox("Kies een optie voor export", names, index=0, key="export_choice")
    sel = options[names.index(choice)]
    sel_df = sel["result"].copy()

    if "price_buy_unit" not in sel_df.columns or "price_effective" not in sel_df.columns or "prijs" not in sel_df.columns:
        merge_cols = ["artikel","merk","soort"]
        sel_df = sel_df.merge(
            df_work[merge_cols + ["price_buy_unit","price_effective","prijs"]],
            on=merge_cols, how="left"
        )

    if "omzet_subtotaal" not in sel_df.columns:
        sel_df["omzet_subtotaal"]  = sel_df["aantal"] * sel_df["price_effective"]
    if "inkoop_subtotaal" not in sel_df.columns:
        sel_df["inkoop_subtotaal"] = sel_df["aantal"] * sel_df["price_buy_unit"]
    total_omzet  = float(sel_df["omzet_subtotaal"].sum())
    total_inkoop = float(sel_df["inkoop_subtotaal"].sum())
    marge_eur    = total_omzet - total_inkoop
    marge_pct    = (marge_eur / total_omzet) * 100.0 if total_omzet > 0 else 0.0

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        st.metric("Totaal (klant)", euro(total_omzet, 0))
    with colB:
        st.metric("Gem. klasse", f"{sel['avg']:.2f}".replace(".", ","))
    with colC:
        st.metric("Marge (intern)", f"{euro(marge_eur,0)}  ·  {marge_pct:.1f}%")

    st.markdown("### Samenvatting per soort (met aantallen per artikel)")
    sel_df["artikel_merk"] = sel_df["artikel"].astype(str) + " (" + sel_df["merk"].astype(str) + ")"
    sel_df["gew_klasse"] = sel_df["aantal"] * sel_df["klasse"]

    art_counts = (
        sel_df.groupby(["soort", "artikel_merk"], as_index=False)["aantal"].sum()
        .sort_values(["soort","aantal"], ascending=[True, False])
    )
    def format_art_row(df_local):
        parts = [f"{int(r['aantal'])}× {r['artikel_merk']}" for _, r in df_local.iterrows() if r["aantal"] > 0]
        return " • ".join(parts) if parts else ""

    artikelen_per_soort = art_counts.groupby("soort").apply(format_art_row).reset_index(name="artikelen")

    grp = sel_df.groupby("soort", as_index=False).agg(
        totaal_stuks=("aantal", "sum"),
        omzet=("omzet_subtotaal", "sum"),
        inkoop=("inkoop_subtotaal", "sum"),
        som_gew_klasse=("gew_klasse", "sum"),
        som_aantal=("aantal", "sum"),
    )
    grp["gem_klasse_per_soort"] = grp["som_gew_klasse"] / grp["som_aantal"]
    grp["marge"] = grp["omzet"] - grp["inkoop"]
    grp["marge_pct"] = grp.apply(lambda r: (r["marge"] / r["omzet"] * 100.0) if r["omzet"] > 0 else 0.0, axis=1)
    grp = grp.merge(artikelen_per_soort, on="soort", how="left")

    def _eur0(v):
        try:
            return f"€ {float(v):,.0f}".replace(",", ".")
        except Exception:
            return "€ 0"
    grp_show = grp.copy()
    grp_show["omzet"]  = grp_show["omzet"].apply(_eur0)
    grp_show["inkoop"] = grp_show["inkoop"].apply(_eur0)
    grp_show["marge"]  = grp_show["marge"].apply(_eur0)
    grp_show["marge_pct"] = grp_show["marge_pct"].apply(lambda x: f"{x:.1f}%")
    grp_show["gem. kl."] = grp_show["gem_klasse_per_soort"].apply(lambda x: f"{x:.2f}".replace(".", ","))

    cols_show = ["soort", "artikelen", "totaal_stuks", "omzet", "inkoop", "marge", "marge_pct", "gem. kl."]
    cols_show = [c for c in cols_show if c in grp_show.columns]
    st.dataframe(grp_show[cols_show], use_container_width=True, hide_index=True)

    # CSV
    out = sel_df[[
        "artikel","soort","merk","prijs","klasse","aantal",
        "price_effective","price_buy_unit","omzet_subtotaal","inkoop_subtotaal"
    ]].rename(columns={
        "prijs": "bruto_prijs",
        "price_effective": "verkoop_per_stuk",
        "price_buy_unit": "inkoop_per_stuk",
    })
    st.download_button("⬇️ Download offerte (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name="offerte.csv", mime="text/csv")

    # PDF export
    st.markdown("### PDF export")
    c1, c2 = st.columns([2, 1])
    with c1:
        project_title  = st.text_input("Titel op de offerte/rapport", value=st.session_state.get("pdf_title", ""), key="pdf_title")
        project_client = st.text_input("Klantnaam (optioneel)", value=st.session_state.get("pdf_client", ""), key="pdf_client")
        project_ref    = st.text_input("Referentie (optioneel)", value=st.session_state.get("pdf_ref", ""), key="pdf_ref")
    with c2:
        pdf_variant = st.radio("PDF-type", ["Offerte (klant)", "Rapport (intern)"], horizontal=True, index=0, key="pdf_variant_radio")
        only_first_page = st.checkbox("Alleen 1e pagina bij PDF-briefpapier", value=st.session_state.get("pdf_firstpage", True), key="pdf_firstpage")
    internal = (pdf_variant == "Rapport (intern)")
    letterhead_file = st.file_uploader("Optioneel: upload briefpapier (PNG, JPG of PDF)", type=["png","jpg","jpeg","pdf"], key="pdf_letterhead")

    if st.button("📄 Genereer PDF", key="pdf_generate"):
        meta = {
            "title": (project_title or ("Rapport (intern)" if internal else "Offerte")),
            "client": (project_client or "").strip(),
            "reference": (project_ref or "").strip(),
            "date_str": format_date(),
            "discount_client_str": f"{float(st.session_state.get('klant_korting_value', klant_korting_pct)):.0f}%",
            "budget_str": euro(last_budget_shown or 0, 0) if doel_state == DOEL_BUDGET else "",
            "mode_str": badge,
            "footer": "© jouw-merk – gegenereerd met Quote Optimizer",
        }

        res_pdf = sel_df.copy()
        def _eur(v):
            try:
                return f"€ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            except Exception:
                return str(v)
        res_pdf["Brutoprijs"]        = res_pdf["prijs"].apply(_eur)
        res_pdf["Verkoop/klant"]     = res_pdf["price_effective"].apply(_eur)
        res_pdf["Inkoop/netto"]      = res_pdf["price_buy_unit"].apply(_eur)
        res_pdf["Subtotaal klant"]   = (res_pdf["aantal"] * res_pdf["price_effective"]).apply(_eur)
        res_pdf["Subtotaal inkoop"]  = (res_pdf["aantal"] * res_pdf["price_buy_unit"]).apply(_eur)

        pdf_grp = grp.copy()
        pdf_grp["omzet"]  = pdf_grp["omzet"].round(2)
        pdf_grp["inkoop"] = pdf_grp["inkoop"].round(2)
        pdf_grp["marge"]  = pdf_grp["marge"].round(2)
        pdf_grp["marge_pct"] = pdf_grp["marge_pct"].round(1)

        try:
            base_pdf = build_offer_pdf(
                project_meta=meta,
                result_df=res_pdf,
                summary_df=(pdf_grp[["soort","totaal_stuks","omzet","inkoop","marge","marge_pct","gem_klasse_per_soort"]] if internal
                            else pdf_grp[["soort","totaal_stuks","omzet","gem_klasse_per_soort"]]),
                show_prices=True,
                letterhead_image_bytes=(letterhead_file.read() if letterhead_file and letterhead_file.type != "application/pdf" else None),
                internal=internal
            )
            final_pdf = base_pdf
            if letterhead_file and letterhead_file.type == "application/pdf":
                final_pdf = overlay_pdf_letterhead(base_pdf, letterhead_file.read(), first_page_only=only_first_page)

            st.success("PDF klaar om te downloaden.")
            st.download_button(
                "⬇️ Download PDF",
                data=final_pdf,
                file_name=f"{'Rapport' if internal else 'Offerte'}_{(project_ref or format_date()).replace(' ', '_')}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"PDF-generatie mislukte: {e}")

else:
    st.info("Upload een prijslijst of zet de demo-dataset aan om te starten.")

# ─────────────────────────────────────────────────────────
# Onderaan: Resetknop (zachte reset: houdt prijslijst vast)
# ─────────────────────────────────────────────────────────
st.markdown("---")

def _soft_reset():
    keys_to_clear = [
        "doel_radio", "klasse_slider", "budget_input_str", "budget_value",
        "budget_input", "budget_basis_radio",
        "klant_korting_pct_input", "klant_korting_value",
        "editor_types", "editor_df", "editor_df_widget",
        "gbb_options", "badge", "doel",
        "target_avg", "last_budget_shown",
        "df_work", "needs", "per_type_min", "per_type_max", "no_mix_by_type",
        "pdf_title", "pdf_client", "pdf_ref", "pdf_variant_radio", "pdf_firstpage",
        "pdf_letterhead", "compare_checkbox", "export_choice", "cmp_budget_input",
        "uploader_main", "use_demo_toggle",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    st.rerun()

if st.button("Reset invoer", help="Zet aantallen/keuzes/resultaten terug, behoudt geüploade prijslijst."):
    _soft_reset()
