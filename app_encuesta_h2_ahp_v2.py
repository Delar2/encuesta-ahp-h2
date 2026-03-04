import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO

# ============================================================
# FIXED SETTINGS
# ============================================================
CR_THRESHOLD = 0.10  # fixed, not editable

RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
}

CONFIDENCE_DELTAS = {
    "Muy seguro": 0.5,
    "Moderadamente seguro": 1.0,
    "Poco seguro": 2.0
}
CONFIDENCE_OPTIONS = list(CONFIDENCE_DELTAS.keys())

DEFINICIONES = {
    # --- Criterios principales ---
    "Cobertura del muestreo": "la cantidad, profundidad y duración de las mediciones en campo",
    "Analítica en laboratorio": "la capacidad de obtener información sobre la calidad, el origen y la composición del gas",
    "Capacidad de sensores": "la sensibilidad, el rango de medición y la capacidad de los equipos de campo para detectar múltiples gases",

    # --- Cobertura del muestreo (Campo) ---
    "Densidad de puntos": "el número de mediciones realizadas dentro de cada círculo de Hada",
    "Profundidad del muestreo": "el nivel de penetración en el suelo (p. ej., 1 m, 4 m o 6 m)",
    "Duración del muestreo": "el tiempo de medición continua por punto (p. ej., 1 hora, 24 horas o 1 semana)",

    # --- Analítica en laboratorio (subcriterios) ---
    "Caracterización isotópica": "la identificación del origen y la fuente del gas",
    "Cromatografía de gases": "la determinación de la composición y la pureza del gas",
    "Caracterización mineralógica": "la identificación de minerales (p. ej., DRX/FRX) y la evaluación de la reactividad del suelo",
    "Análisis biogeoquímicos": "la determinación de la actividad biológica en el suelo asociada a la presencia de hidrógeno",
    "Análisis fisicoquímicos": "la determinación de propiedades del suelo como porosidad, permeabilidad, pH y potencial redox",

    # --- Capacidad de sensores (Campo) ---
    "Límite de detección": "el rango, la sensibilidad y la precisión del sensor para medir diferentes concentraciones de hidrógeno",
    "Capacidad multigas": "la detección simultánea de otros gases además del hidrógeno, como helio y radón (trazadores de fuente)",
}

# ============================================================
# Secrets safe
# ============================================================
def get_secrets_safe():
    try:
        _ = st.secrets
        return st.secrets
    except Exception:
        return None

SECRETS = get_secrets_safe()

def email_enabled():
    if SECRETS is None:
        return False
    needed = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "ADMIN_EMAIL"]
    return all(k in SECRETS for k in needed)

# ============================================================
# Crisp mapping: k -> ratio
# ============================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def slider_to_ratio_int(k: int) -> float:
    mapping_left = {1: 9, 2: 7, 3: 5, 4: 3, 5: 1}
    if k <= 5:
        return float(mapping_left[k])
    inv = mapping_left[10 - k]
    return 1.0 / float(inv)

# Anchor map for continuous interpolation (k may be float due to ±0.5 etc)
ANCHOR_K = np.array([1,2,3,4,5,6,7,8,9], dtype=float)
ANCHOR_R = np.array([9,7,5,3,1,1/3,1/5,1/7,1/9], dtype=float)

def ratio_from_k(k: float) -> float:
    """
    Continuous mapping from k (float in [1..9]) -> ratio A/B
    Uses log-linear interpolation between anchors.
    """
    k = clamp(float(k), 1.0, 9.0)
    if float(int(k)) == k:
        return float(ANCHOR_R[int(k)-1])
    i = int(np.floor(k))
    j = int(np.ceil(k))
    if i == j:
        return float(ANCHOR_R[i-1])
    ri = ANCHOR_R[i-1]
    rj = ANCHOR_R[j-1]
    t = (k - i) / (j - i)
    return float(np.exp((1-t)*np.log(ri) + t*np.log(rj)))

# ============================================================
# Crisp AHP: matrices, CR, weights
# ============================================================
def build_matrix(n: int, answers: dict) -> np.ndarray:
    A = np.ones((n, n), dtype=float)
    for (i,j), r in answers.items():
        A[i,j] = float(r)
        A[j,i] = 1.0 / float(r)
    return A

def ahp_cr(A: np.ndarray):
    n = A.shape[0]
    if n <= 2:
        return 0.0, 0.0, 0.0
    vals, _ = np.linalg.eig(A)
    lam = float(np.max(vals.real))
    CI = (lam - n) / (n - 1)
    RI = RI_TABLE.get(n, 0.0)
    CR = 0.0 if RI == 0 else CI / RI
    return lam, float(CI), float(CR)

def ahp_weights_eigen(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(A)
    idx = np.argmax(vals.real)
    w = np.abs(vecs[:, idx].real)
    w = w / w.sum()
    return w

def triad_inconsistency_report(A: np.ndarray, labels: list, top_k: int = 5):
    n = A.shape[0]
    rows = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                implied = A[i,j]*A[j,k]
                actual = A[i,k]
                err = abs(np.log(actual) - np.log(implied))
                rows.append({
                    "Triada": f"{labels[i]} → {labels[j]} → {labels[k]}",
                    "Actual (i/k)": actual,
                    "Implicado": implied,
                    "Sugerencia": f"Ajustar {labels[i]} vs {labels[k]} hacia ~ {implied:.3g}",
                    "Error_log": err
                })
    rows.sort(key=lambda x: x["Error_log"], reverse=True)
    return rows[:min(top_k, len(rows))]

# ============================================================
# Fuzzy TFN operations + fuzzy weights (Buckley GM)
# ============================================================
def tfn_mul(a, b): return (a[0]*b[0], a[1]*b[1], a[2]*b[2])
def tfn_pow(a, p): return (a[0]**p, a[1]**p, a[2]**p)
def tfn_add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def tfn_inv(a): return (1.0/a[2], 1.0/a[1], 1.0/a[0])
def tfn_div(a, b): return tfn_mul(a, tfn_inv(b))
def tfn_defuzz(a): return (a[0]+a[1]+a[2]) / 3.0

def fuzzy_weights_geometric_mean(fuzzy_A):
    n = len(fuzzy_A)
    g = []
    for i in range(n):
        prod = (1.0, 1.0, 1.0)
        for j in range(n):
            prod = tfn_mul(prod, fuzzy_A[i][j])
        g.append(tfn_pow(prod, 1.0/n))

    sum_g = (0.0, 0.0, 0.0)
    for gi in g:
        sum_g = tfn_add(sum_g, gi)

    w = [tfn_div(gi, sum_g) for gi in g]
    w_def = np.array([tfn_defuzz(wi) for wi in w], dtype=float)
    w_def = w_def / w_def.sum()
    return w, w_def.tolist()

# ============================================================
# Email sender
# ============================================================
def send_email(to_email: str, subject: str, body: str, attachment_bytes: bytes, attachment_name: str):
    import smtplib
    from email.message import EmailMessage

    smtp_host = SECRETS["SMTP_HOST"]
    smtp_port = int(SECRETS["SMTP_PORT"])
    smtp_user = SECRETS["SMTP_USER"]
    smtp_pass = SECRETS["SMTP_PASS"]
    from_email = SECRETS.get("SMTP_FROM", smtp_user)

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    msg.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=attachment_name
    )

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

# ============================================================
# Survey structure
# ============================================================
MAIN = ("Criterios principales",
        ["Cobertura del muestreo", "Analítica en laboratorio", "Capacidad de sensores"],
        [("Cobertura del muestreo", "Analítica en laboratorio", "P1"),
         ("Cobertura del muestreo", "Capacidad de sensores", "P2"),
         ("Analítica en laboratorio", "Capacidad de sensores", "P3")]
       )

COV  = ("Cobertura del muestreo (Campo)",
        ["Densidad de puntos", "Profundidad del muestreo", "Duración del muestreo"],
        [("Densidad de puntos", "Profundidad del muestreo", "C1"),
         ("Densidad de puntos", "Duración del muestreo", "C2"),
         ("Profundidad del muestreo", "Duración del muestreo", "C3")]
       )

LAB  = ("Analítica en laboratorio",
        ["Caracterización isotópica", "Cromatografía de gases", "Caracterización mineralógica",
         "Análisis biogeoquímicos", "Análisis fisicoquímicos"],
        [("Caracterización isotópica", "Cromatografía de gases", "L1"),
         ("Caracterización isotópica", "Caracterización mineralógica", "L2"),
         ("Caracterización isotópica", "Análisis biogeoquímicos", "L3"),
         ("Caracterización isotópica", "Análisis fisicoquímicos", "L4"),
         ("Cromatografía de gases", "Caracterización mineralógica", "L5"),
         ("Cromatografía de gases", "Análisis biogeoquímicos", "L6"),
         ("Cromatografía de gases", "Análisis fisicoquímicos", "L7"),
         ("Caracterización mineralógica", "Análisis biogeoquímicos", "L8"),
         ("Caracterización mineralógica", "Análisis fisicoquímicos", "L9"),
         ("Análisis biogeoquímicos", "Análisis fisicoquímicos", "L10")]
       )

SENS = ("Capacidad de sensores (Campo)",
        ["Límite de detección", "Capacidad multigas"],
        [("Límite de detección", "Capacidad multigas", "S1")]
       )

SECTIONS = [MAIN, COV, LAB, SENS]

# Parent mapping for global weights
PARENT_OF = {
    "Cobertura del muestreo (Campo)": "Cobertura del muestreo",
    "Analítica en laboratorio": "Analítica en laboratorio",
    "Capacidad de sensores (Campo)": "Capacidad de sensores",
}

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Encuesta AHP H2", layout="centered")
st.title("Encuesta AHP — Metodología de detección de H₂")
st.caption("Encuesta realizada por Juan Pardo y Salim Shalom")

st.markdown("""
### Descripción de la encuesta

Esta encuesta tiene como objetivo evaluar la importancia relativa de diferentes criterios y subcriterios 
utilizados en la **metodología de detección de hidrógeno natural (H₂)**.  

Las comparaciones se realizan mediante el método **AHP (Analytic Hierarchy Process)** y su extensión 
**Fuzzy AHP**, lo que permite incorporar la incertidumbre asociada a las decisiones de los expertos.

Durante la encuesta se le pedirá comparar pares de criterios utilizando una escala de **1 a 9**, donde:
- **5** indica que ambos criterios tienen igual importancia.
- Valores hacia la izquierda indican mayor importancia del primer criterio.
- Valores hacia la derecha indican mayor importancia del segundo criterio.

Además, deberá indicar su **nivel de confianza** en cada comparación:
- **Muy seguro**
- **Moderadamente seguro**
- **Poco seguro**

El sistema verificará automáticamente la **consistencia de sus respuestas (CR ≤ 0.10)**.  
Si la consistencia supera este valor, se le solicitará ajustar algunas comparaciones antes de continuar.

La información recopilada será utilizada exclusivamente con el fin de generar progresos en el proyecto.

Muchas gracias por su colaboración.
""")


respondent_id = st.text_input("Nombre completo", value="", placeholder="Ej: Juan Pardo")
if not respondent_id.strip():
    st.info("Escribe tu nombre/ID para comenzar.")
    st.stop()

if not email_enabled():
    st.warning(
        "⚠️ Correo no configurado (faltan Secrets). La encuesta funciona, pero no enviará resultados."
    )

if "step" not in st.session_state:
    st.session_state.step = 0
if "blocks" not in st.session_state:
    st.session_state.blocks = {}  # sec_name -> dict(rows, CR, A_crisp, items, w_crisp, fuzzy_mats, w_fuzzy)
if "cr_by_section" not in st.session_state:
    st.session_state.cr_by_section = {}

st.progress(st.session_state.step / len(SECTIONS))

def render_section(sec_name, items, comps):
    st.header(sec_name)

    idx = {name:i for i,name in enumerate(items)}
    crisp_answers = {}
    rows = []

    for (a,b,qid) in comps:
        st.subheader(f"{a} vs {b}")
        def_a = DEFINICIONES.get(a, "—")
        def_b = DEFINICIONES.get(b, "—")

        st.markdown(f"""
        <div style="
        background-color:#1f2937;
        padding:15px;
        border-radius:10px;
        border-left:6px solid #22c55e;
        font-size:16px;
        line-height:1.6;
        ">

        Recuerde que la <b>{a}</b> hace referencia a <b>{def_a}</b>, mientras que la <b>{b}</b> es <b>{def_b}</b>.  
        <br><br>
        Utilice la escala lineal para desplazar el marcador hacia el criterio que considere predominante.  
        Marque <b>5</b> si cree que ambos son igualmente necesarios.

        </div>
        """, unsafe_allow_html=True)
        k = st.slider("Escala 1–9 (5 = iguales)", 1, 9, 5, 1, key=f"k_{sec_name}_{qid}")
        conf = st.selectbox("Confianza", CONFIDENCE_OPTIONS, key=f"conf_{sec_name}_{qid}")

        # Crisp for CR
        r_crisp = slider_to_ratio_int(int(k))
        i, j = idx[a], idx[b]
        crisp_answers[(i,j)] = r_crisp

        # Fuzzy on k
        d = CONFIDENCE_DELTAS[conf]
        k_l = clamp(k - d, 1.0, 9.0)
        k_m = float(k)
        k_u = clamp(k + d, 1.0, 9.0)

        # Convert to ratio TFN (monotonic decreasing)
        r_l = ratio_from_k(k_u)
        r_m = ratio_from_k(k_m)
        r_u = ratio_from_k(k_l)

        rows.append({
            "Respondent_ID": respondent_id.strip(),
            "Section": sec_name,
            "A": a, "B": b,
            "k": int(k),
            "Confidence": conf,
            "Delta": float(d),
            "TFN_k_l": float(k_l), "TFN_k_m": float(k_m), "TFN_k_u": float(k_u),
            "Ratio_crisp_for_CR": float(r_crisp),
            "TFN_ratio_l": float(r_l), "TFN_ratio_m": float(r_m), "TFN_ratio_u": float(r_u),
        })

        #st.caption(f"CR usa ratio={r_crisp:.4g}.  Fuzzy TFN_ratio=({r_l:.4g}, {r_m:.4g}, {r_u:.4g})")
        st.divider()

    # Crisp matrix
    A_crisp = build_matrix(len(items), crisp_answers)
    lam, CI, CR = ahp_cr(A_crisp)

    c1,c2,c3 = st.columns(3)
    c1.metric("λmax", f"{lam:.4f}")
    c2.metric("CI", f"{CI:.4f}")
    c3.metric("CR - CONSISTENCIA", f"{CR:.4f}")

    ok = CR <= CR_THRESHOLD
    if ok:
        st.success("✅ Consistencia aceptable. Puedes continuar.")
    else:
        st.error("❌ Consistencia alta. Ajusta respuestas para continuar.")
        triads = triad_inconsistency_report(A_crisp, items, top_k=5)
        st.subheader("Sugerencias para corregir")
        st.dataframe(pd.DataFrame(triads), use_container_width=True, hide_index=True)

    # Crisp weights
    w_crisp = ahp_weights_eigen(A_crisp)

    # Fuzzy matrices L/M/U from TFN_ratio in rows
    n = len(items)
    L = np.ones((n,n), dtype=float)
    M = np.ones((n,n), dtype=float)
    U = np.ones((n,n), dtype=float)

    for r in rows:
        i = idx[r["A"]]
        j = idx[r["B"]]
        l,m,u = r["TFN_ratio_l"], r["TFN_ratio_m"], r["TFN_ratio_u"]
        L[i,j], M[i,j], U[i,j] = l,m,u
        L[j,i], M[j,i], U[j,i] = 1.0/u, 1.0/m, 1.0/l

    fuzzy_A = [[(L[i,j], M[i,j], U[i,j]) for j in range(n)] for i in range(n)]
    w_fuzzy_tfn, w_fuzzy_def = fuzzy_weights_geometric_mean(fuzzy_A)

    return rows, CR, ok, A_crisp, w_crisp, (L,M,U), (w_fuzzy_tfn, w_fuzzy_def)

# run section
sec_name, sec_items, sec_comps = SECTIONS[st.session_state.step]
rows, CR, ok, A_crisp, w_crisp, fuzzy_mats, fuzzy_w = render_section(sec_name, sec_items, sec_comps)

st.session_state.blocks[sec_name] = {
    "rows": rows,
    "CR": CR,
    "A_crisp": A_crisp,
    "items": sec_items,
    "w_crisp": w_crisp,
    "fuzzy_mats": fuzzy_mats,
    "w_fuzzy": fuzzy_w
}
st.session_state.cr_by_section[sec_name] = CR

# navigation
col_prev, col_next = st.columns([1,1])
with col_prev:
    if st.button("⬅️ Anterior", disabled=(st.session_state.step==0)):
        st.session_state.step -= 1
        st.rerun()
with col_next:
    if st.session_state.step < len(SECTIONS)-1:
        if st.button("Siguiente ➡️", disabled=not ok):
            st.session_state.step += 1
            st.rerun()

# ============================================================
# FINAL SUBMIT: create Excel and email it
# ============================================================
if st.session_state.step == len(SECTIONS)-1:
    st.header("Finalizar")
    all_ok = all(st.session_state.cr_by_section.get(s[0], 999) <= CR_THRESHOLD for s in SECTIONS)

    if not all_ok:
        st.error("❌ Alguna sección no cumple CR.")
    else:
        if st.button("Enviar respuestas"):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Pairwise + CR tables
            all_pairwise = []
            cr_rows = []
            sheets = {}

            # ---- Crisp & fuzzy weights per section ----
            crisp_weights_rows = []
            fuzzy_weights_rows = []

            for (name, _, _) in SECTIONS:
                block = st.session_state.blocks[name]
                all_pairwise.extend(block["rows"])
                cr_rows.append({
                    "Respondent_ID": respondent_id.strip(),
                    "Timestamp": ts,
                    "Section": name,
                    "CR": float(block["CR"])
                })

                items = block["items"]

                # Crisp matrix
                sheets[f"Crisp_{name[:20]}"] = pd.DataFrame(block["A_crisp"], index=items, columns=items)

                # Crisp weights
                for it, wv in zip(items, block["w_crisp"]):
                    crisp_weights_rows.append({
                        "Respondent_ID": respondent_id.strip(),
                        "Timestamp": ts,
                        "Section": name,
                        "Item": it,
                        "Weight_crisp": float(wv)
                    })

                # Fuzzy matrices
                L,M,U = block["fuzzy_mats"]
                sheets[f"FuzzyL_{name[:18]}"] = pd.DataFrame(L, index=items, columns=items)
                sheets[f"FuzzyM_{name[:18]}"] = pd.DataFrame(M, index=items, columns=items)
                sheets[f"FuzzyU_{name[:18]}"] = pd.DataFrame(U, index=items, columns=items)

                # Fuzzy weights
                w_tfn, w_def = block["w_fuzzy"]
                for it, wi, wd in zip(items, w_tfn, w_def):
                    fuzzy_weights_rows.append({
                        "Respondent_ID": respondent_id.strip(),
                        "Timestamp": ts,
                        "Section": name,
                        "Item": it,
                        "w_l": float(wi[0]),
                        "w_m": float(wi[1]),
                        "w_u": float(wi[2]),
                        "w_defuzz_norm": float(wd)
                    })

                # Add weights sheets per section for easy viewing
                dfw_c = pd.DataFrame([r for r in crisp_weights_rows if r["Section"] == name])[["Item","Weight_crisp"]] \
                        .sort_values("Weight_crisp", ascending=False)
                sheets[f"CrispW_{name[:18]}"] = dfw_c

                dfw_f = pd.DataFrame([r for r in fuzzy_weights_rows if r["Section"] == name])[["Item","w_l","w_m","w_u","w_defuzz_norm"]] \
                        .sort_values("w_defuzz_norm", ascending=False)
                sheets[f"FuzzyW_{name[:18]}"] = dfw_f

            df_pairwise = pd.DataFrame(all_pairwise)
            df_pairwise.insert(1, "Timestamp", ts)
            df_cr = pd.DataFrame(cr_rows)

            # ========================================================
            # GLOBAL WEIGHTS (crisp + fuzzy)
            # ========================================================
            # 1) Main criterion weights come from "Criterios principales"
            main_block = st.session_state.blocks["Criterios principales"]
            main_items = main_block["items"]

            # crisp main weights
            w_main_crisp = main_block["w_crisp"]
            main_crisp_map = {it: float(w) for it, w in zip(main_items, w_main_crisp)}

            # fuzzy main weights (defuzz normalized)
            w_main_fuzzy_def = main_block["w_fuzzy"][1]
            main_fuzzy_map = {it: float(w) for it, w in zip(main_items, w_main_fuzzy_def)}

            # 2) Local weights inside sub-blocks => multiply by parent weight
            global_crisp_rows = []
            global_fuzzy_rows = []

            for sub_sec in ["Cobertura del muestreo (Campo)", "Analítica en laboratorio", "Capacidad de sensores (Campo)"]:
                parent = PARENT_OF[sub_sec]
                block = st.session_state.blocks[sub_sec]
                items = block["items"]

                # crisp globals
                for it, wloc in zip(items, block["w_crisp"]):
                    global_crisp_rows.append({
                        "Parent": parent,
                        "Subcriterion": it,
                        "Local_weight_crisp": float(wloc),
                        "Parent_weight_crisp": main_crisp_map[parent],
                        "Global_weight_crisp": float(wloc) * main_crisp_map[parent]
                    })

                # fuzzy globals (use defuzz normalized)
                wloc_f_def = block["w_fuzzy"][1]
                for it, wloc_def in zip(items, wloc_f_def):
                    global_fuzzy_rows.append({
                        "Parent": parent,
                        "Subcriterion": it,
                        "Local_weight_fuzzy_def": float(wloc_def),
                        "Parent_weight_fuzzy_def": main_fuzzy_map[parent],
                        "Global_weight_fuzzy_def": float(wloc_def) * main_fuzzy_map[parent]
                    })

            df_main_weights = pd.DataFrame({
                "Criterion": main_items,
                "Weight_crisp": [main_crisp_map[it] for it in main_items],
                "Weight_fuzzy_def": [main_fuzzy_map[it] for it in main_items],
            }).sort_values("Weight_fuzzy_def", ascending=False)

            df_global_crisp = pd.DataFrame(global_crisp_rows).sort_values("Global_weight_crisp", ascending=False)
            df_global_fuzzy = pd.DataFrame(global_fuzzy_rows).sort_values("Global_weight_fuzzy_def", ascending=False)

            # ========================================================
            # Write Excel to memory
            # ========================================================
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as writer:
                df_pairwise.to_excel(writer, index=False, sheet_name="Pairwise")
                df_cr.to_excel(writer, index=False, sheet_name="CR_por_seccion")
                df_main_weights.to_excel(writer, index=False, sheet_name="Main_Weights")
                df_global_crisp.to_excel(writer, index=False, sheet_name="Global_Weights_Crisp")
                df_global_fuzzy.to_excel(writer, index=False, sheet_name="Global_Weights_Fuzzy")

                # matrices + weight sheets
                for sname, sdf in sheets.items():
                    safe = sname.replace(" ", "_")[:31]
                    sdf.to_excel(writer, sheet_name=safe)

            excel_bytes = bio.getvalue()
            filename = f"AHP_Fuzzy_FULL_{respondent_id.strip().replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            # Email
            if email_enabled():
                admin = SECRETS["ADMIN_EMAIL"]
                subject = f"Respuesta AHP/Fuzzy (FULL): {respondent_id.strip()}"
                body = (
                    f"Se recibió una nueva respuesta.\n\n"
                    f"Encuestado: {respondent_id.strip()}\n"
                    f"Timestamp: {ts}\n\n"
                    f"CR por sección:\n{df_cr.to_string(index=False)}\n\n"
                    f"Adjunto: Excel con respuestas + matrices crisp/fuzzy + pesos crisp/fuzzy + pesos globales.\n"
                )
                try:
                    send_email(admin, subject, body, excel_bytes, filename)
                except Exception as e:
                    st.warning(f"No pude enviar el correo: {e}")

            st.success("¡Gracias por su participación!")
            st.session_state.step = 0
            st.session_state.blocks = {}
            st.session_state.cr_by_section = {}
            st.stop()