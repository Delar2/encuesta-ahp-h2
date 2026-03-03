import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import json

# -----------------------------
# AHP consistency (Saaty)
# -----------------------------
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

CONFIDENCE_OPTIONS = ["Muy seguro", "Moderadamente seguro", "Poco seguro"]

def slider_to_ratio(slider_value: int) -> float:
    # 5 = igual; 1..4 prioriza A; 6..9 prioriza B
    # 1->9, 2->7, 3->5, 4->3, 5->1, 6->1/3, 7->1/5, 8->1/7, 9->1/9
    mapping_left = {1: 9, 2: 7, 3: 5, 4: 3, 5: 1}
    if slider_value <= 5:
        return float(mapping_left[slider_value])
    inv = mapping_left[10 - slider_value]
    return 1.0 / float(inv)

def build_matrix(items, pair_answers):
    n = len(items)
    A = np.ones((n, n), dtype=float)
    for (i, j), r in pair_answers.items():
        A[i, j] = float(r)
        A[j, i] = 1.0 / float(r)
    return A

def ahp_weights_eigen(A):
    vals, vecs = np.linalg.eig(A)
    idx = np.argmax(vals.real)
    w = np.abs(vecs[:, idx].real)
    return w / w.sum()

def ahp_cr(A):
    n = A.shape[0]
    if n <= 2:
        return 0.0, 0.0, 0.0
    vals, _ = np.linalg.eig(A)
    lam = float(np.max(vals.real))
    CI = (lam - n) / (n - 1)
    RI = RI_TABLE.get(n, 0.0)
    CR = 0.0 if RI == 0 else CI / RI
    return lam, float(CI), float(CR)

# -----------------------------
# Optional: Google Sheets export (service account)
# -----------------------------
def append_to_gsheet(df: pd.DataFrame, sheet_id: str, worksheet_name: str, service_account_json: str):
    """
    Writes rows at end of an existing worksheet.
    Requires: pip install gspread google-auth
    service_account_json: content of your Google service account JSON key file (as text).
    """
    import gspread
    from google.oauth2.service_account import Credentials

    creds_dict = json.loads(service_account_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="50")
        ws.append_row(list(df.columns))

    # Ensure header exists (basic check)
    existing = ws.row_values(1)
    if not existing or existing != list(df.columns):
        ws.clear()
        ws.append_row(list(df.columns))

    ws.append_rows(df.values.tolist())

# -----------------------------
# Survey structure (from your PDF)
# -----------------------------
MAIN_ITEMS = ["Cobertura del muestreo", "Analítica en laboratorio", "Capacidad de sensores"]
MAIN_COMPARISONS = [
    ("Cobertura del muestreo", "Analítica en laboratorio", "P1"),
    ("Cobertura del muestreo", "Capacidad de sensores", "P2"),
    ("Analítica en laboratorio", "Capacidad de sensores", "P3"),
]

COV_ITEMS = ["Densidad de puntos", "Profundidad del muestreo", "Duración del muestreo"]
COV_COMPARISONS = [
    ("Densidad de puntos", "Profundidad del muestreo", "C1"),
    ("Densidad de puntos", "Duración del muestreo", "C2"),
    ("Profundidad del muestreo", "Duración del muestreo", "C3"),
]

LAB_ITEMS = [
    "Caracterización isotópica",
    "Cromatografía de gases",
    "Caracterización mineralógica",
    "Análisis biogeoquímicos",
    "Análisis fisicoquímicos",
]
# 5 items => 10 comparaciones
LAB_COMPARISONS = [
    ("Caracterización isotópica", "Cromatografía de gases", "L1"),
    ("Caracterización isotópica", "Caracterización mineralógica", "L2"),
    ("Caracterización isotópica", "Análisis biogeoquímicos", "L3"),
    ("Caracterización isotópica", "Análisis fisicoquímicos", "L4"),
    ("Cromatografía de gases", "Caracterización mineralógica", "L5"),
    ("Cromatografía de gases", "Análisis biogeoquímicos", "L6"),
    ("Cromatografía de gases", "Análisis fisicoquímicos", "L7"),
    ("Caracterización mineralógica", "Análisis biogeoquímicos", "L8"),
    ("Caracterización mineralógica", "Análisis fisicoquímicos", "L9"),
    ("Análisis biogeoquímicos", "Análisis fisicoquímicos", "L10"),
]

SENS_ITEMS = ["Límite de detección", "Capacidad multigas"]
SENS_COMPARISONS = [("Límite de detección", "Capacidad multigas", "S1")]

SECTIONS = [
    ("Criterios principales", MAIN_ITEMS, MAIN_COMPARISONS),
    ("Cobertura del muestreo (Campo)", COV_ITEMS, COV_COMPARISONS),
    ("Analítica en laboratorio", LAB_ITEMS, LAB_COMPARISONS),
    ("Capacidad de sensores (Campo)", SENS_ITEMS, SENS_COMPARISONS),
]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Encuesta AHP H2 (CR en tiempo real)", layout="centered")
st.title("Encuesta AHP — Metodología de detección de H₂ (con consistencia en tiempo real)")
st.caption("Reproduce tu Google Form y bloquea el envío si CR es alto por sección.")

with st.sidebar:
    st.header("Configuración")
    cr_threshold = st.number_input("Umbral CR", 0.01, 0.30, 0.10, 0.01)
    respondent_id = st.text_input("Nombre completo / ID", value="")
    st.divider()
    st.subheader("Guardar en Google Sheets (opcional)")
    enable_gsheets = st.checkbox("Activar guardado a Google Sheets", value=False)
    sheet_id = st.text_input("Google Sheet ID (la parte entre /d/ y /edit)", value="", disabled=not enable_gsheets)
    worksheet_pairs = st.text_input("Nombre hoja Pairwise", value="Pairwise", disabled=not enable_gsheets)
    worksheet_cr = st.text_input("Nombre hoja CR", value="CR_por_seccion", disabled=not enable_gsheets)
    worksheet_w = st.text_input("Nombre hoja Pesos", value="Pesos_AHP", disabled=not enable_gsheets)
    service_account_json = st.text_area(
        "Service Account JSON (pegar aquí el contenido completo)",
        value="",
        height=140,
        disabled=not enable_gsheets
    )

if "step" not in st.session_state:
    st.session_state.step = 0

if not respondent_id.strip():
    st.warning("Escribe tu Nombre/ID en la barra lateral (arriba) para continuar.")

# progress
progress = (st.session_state.step) / (len(SECTIONS))
st.progress(progress)

# storage
if "rows" not in st.session_state:
    st.session_state.rows = []
if "cr_by_section" not in st.session_state:
    st.session_state.cr_by_section = {}
if "weights_by_section" not in st.session_state:
    st.session_state.weights_by_section = {}
if "matrices_by_section" not in st.session_state:
    st.session_state.matrices_by_section = {}

def render_section(section_name, items, comparisons):
    st.header(section_name)

    idx = {name: i for i, name in enumerate(items)}
    pair_answers = {}
    local_rows = []

    for (a, b, qid) in comparisons:
        st.subheader(f"{a}  vs  {b}")

        slider = st.slider(
            "Escala 1–9 (5 = iguales). Izquierda = prioridad a A, derecha = prioridad a B.",
            min_value=1, max_value=9, value=5, step=1,
            key=f"s_{section_name}_{qid}"
        )
        conf = st.selectbox(
            "¿Qué tan seguro está de esta evaluación?",
            CONFIDENCE_OPTIONS,
            key=f"c_{section_name}_{qid}"
        )

        ratio = slider_to_ratio(slider)  # A/B
        i, j = idx[a], idx[b]
        pair_answers[(i, j)] = ratio

        local_rows.append({
            "Respondent_ID": respondent_id.strip(),
            "Section": section_name,
            "A": a,
            "B": b,
            "Slider_1to9": slider,
            "Ratio_A_over_B": ratio,
            "Confidence": conf
        })

        st.caption(f"Ratio A/B usado para consistencia: {ratio:.4g}")
        st.divider()

    A = build_matrix(items, pair_answers)
    w = ahp_weights_eigen(A)
    lam, CI, CR = ahp_cr(A)

    c1, c2, c3 = st.columns(3)
    c1.metric("λmax", f"{lam:.4f}")
    c2.metric("CI", f"{CI:.4f}")
    c3.metric("CR", f"{CR:.4f}")

    ok = CR <= cr_threshold
    if ok:
        st.success(f"✅ Consistencia aceptable: CR ≤ {cr_threshold:.2f}")
    else:
        st.error(f"❌ Consistencia alta: CR > {cr_threshold:.2f} — ajusta respuestas para continuar")

    weights_df = pd.DataFrame({"Item": items, "Weight": w}).sort_values("Weight", ascending=False)
    st.write("Pesos (AHP clásico) — informativo:")
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    with st.expander("Ver matriz pareada A"):
        st.dataframe(pd.DataFrame(A, index=items, columns=items), use_container_width=True)

    return local_rows, CR, weights_df, pd.DataFrame(A, index=items, columns=items), ok

# Current section
section_name, items, comparisons = SECTIONS[st.session_state.step]
local_rows, CR, wdf, Adf, ok = render_section(section_name, items, comparisons)

# Save current section snapshot into session_state (without duplicating rows)
# We'll overwrite rows for that section each render
def upsert_section_rows(all_rows, new_rows, sec_name):
    all_rows = [r for r in all_rows if r.get("Section") != sec_name]
    all_rows.extend(new_rows)
    return all_rows

st.session_state.rows = upsert_section_rows(st.session_state.rows, local_rows, section_name)
st.session_state.cr_by_section[section_name] = CR
st.session_state.weights_by_section[section_name] = wdf
st.session_state.matrices_by_section[section_name] = Adf

# Nav buttons
col_prev, col_next = st.columns([1, 1])

with col_prev:
    if st.button("⬅️ Anterior", disabled=(st.session_state.step == 0)):
        st.session_state.step -= 1
        st.rerun()

with col_next:
    if st.session_state.step < len(SECTIONS) - 1:
        if st.button("Siguiente ➡️", disabled=not ok or not respondent_id.strip()):
            st.session_state.step += 1
            st.rerun()

# Final submit
if st.session_state.step == len(SECTIONS) - 1:
    st.divider()
    st.header("Enviar / Exportar")

    all_ok = all(st.session_state.cr_by_section.get(s[0], 999) <= cr_threshold for s in SECTIONS)
    if all_ok:
        st.success("✅ Todas las secciones cumplen CR. Puedes enviar.")
    else:
        st.error("❌ Alguna sección no cumple CR. Vuelve atrás y corrige.")

    if st.button("Guardar y descargar Excel", disabled=not all_ok):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df_pairs = pd.DataFrame(st.session_state.rows)
        df_pairs.insert(1, "Timestamp", ts)

        df_cr = pd.DataFrame([{
            "Respondent_ID": respondent_id.strip(),
            "Timestamp": ts,
            "Section": sec_name,
            "CR": float(st.session_state.cr_by_section.get(sec_name, np.nan))
        } for sec_name, _, _ in SECTIONS])

        def w_out(section, wdf_):
            out = wdf_.copy()
            out.insert(0, "Respondent_ID", respondent_id.strip())
            out.insert(1, "Timestamp", ts)
            out.insert(2, "Section", section)
            return out

        df_w = pd.concat([w_out(sec_name, st.session_state.weights_by_section[sec_name]) for sec_name, _, _ in SECTIONS],
                         ignore_index=True)

        # Excel
        xlsx_name = f"respuestas_AHP_H2_{respondent_id.strip().replace(' ','_')}.xlsx"
        with pd.ExcelWriter(xlsx_name, engine="openpyxl") as writer:
            df_pairs.to_excel(writer, index=False, sheet_name="Pairwise")
            df_cr.to_excel(writer, index=False, sheet_name="CR_por_seccion")
            df_w.to_excel(writer, index=False, sheet_name="Pesos_AHP")
            for sec_name, _, _ in SECTIONS:
                st.session_state.matrices_by_section[sec_name].to_excel(
                    writer, sheet_name=("Matriz_" + sec_name[:20]).replace(" ", "_")[:31]
                )

        # Optional: Google Sheets
        if enable_gsheets and sheet_id.strip() and service_account_json.strip():
            try:
                append_to_gsheet(df_pairs, sheet_id.strip(), worksheet_pairs, service_account_json.strip())
                append_to_gsheet(df_cr, sheet_id.strip(), worksheet_cr, service_account_json.strip())
                append_to_gsheet(df_w, sheet_id.strip(), worksheet_w, service_account_json.strip())
                st.success("✅ También guardado en Google Sheets.")
            except Exception as e:
                st.warning(f"No pude guardar en Google Sheets: {e}")

        st.download_button(
            "Descargar Excel",
            data=open(xlsx_name, "rb").read(),
            file_name=xlsx_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )