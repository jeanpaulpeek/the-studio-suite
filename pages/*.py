from pathlib import Path
import pandas as pd
import streamlit as st

# repo-root: ga 1 map omhoog vanaf /pages
ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_prijslijst.xlsx"

if DEMO.exists():
    # Lezen
    df = pd.read_excel(DEMO)

    # Downloadknop
    st.download_button(
        "Download demo-prijslijst.xlsx",
        DEMO.read_bytes(),
        file_name="demo_prijslijst.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Demo-bestand niet gevonden. Zet 'm in /examples of gebruik de demo-generator.")
