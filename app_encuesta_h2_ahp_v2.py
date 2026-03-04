import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO


# ============================================================

def interpret_pair(k, a, b):

    a_fmt = f"<b><u>{a}</u></b>"
    b_fmt = f"<b><u>{b}</u></b>"

    if k == 5:
        return f"{a_fmt} y {b_fmt} tienen importancia similar."
    elif k < 5:
        return f"{a_fmt} es más importante que {b_fmt}."
    else:
        return f"{b_fmt} es más importante que {a_fmt}."



def k_to_text(k: int, a: str, b: str) -> str:
    if k == 5:
        return f"Iguales ({a} ≈ {b})"
    if k < 5:
        return f"Favorece {a} (k={k})"
    return f"Favorece {b} (k={k})"


# ============================================================

def invert_slider(k: int) -> int:
    return 10 - int(k)


def ratio_to_slider_nearest(r: float) -> int:
    """Convierte ratio A/B a escala 1–9 (5=igual) eligiendo el k más cercano."""
    mapping = {
        1: 9, 2: 7, 3: 5, 4: 3, 5: 1,
        6: 1/3, 7: 1/5, 8: 1/7, 9: 1/9
    }
    best_k, best_err = 5, float("inf")
    for k, rk in mapping.items():
        err = abs(np.log(r) - np.log(rk))
        if err < best_err:
            best_k, best_err = k, err
    return best_k


def all_fixes_3x3(A: np.ndarray, items: list):
    """Tres opciones exactas para arreglar 3×3: ajustar AB o AC o BC manteniendo las otras dos."""
    if A.shape[0] != 3:
        return []

    A12, A13, A23 = A[0,1], A[0,2], A[1,2]

    implied_13 = A12 * A23      # mantener 12 y 23 -> ajustar 13
    implied_23 = A13 / A12      # mantener 12 y 13 -> ajustar 23
    implied_12 = A13 / A23      # mantener 13 y 23 -> ajustar 12

    fixes = [
        {"Cambiar": f"{items[0]} vs {items[2]}",
         "Mantener": f"{items[0]} vs {items[1]} y {items[1]} vs {items[2]}",
         "Sugerido_k": ratio_to_slider_nearest(implied_13),
         "Ratio_implicado": implied_13},

        {"Cambiar": f"{items[1]} vs {items[2]}",
         "Mantener": f"{items[0]} vs {items[1]} y {items[0]} vs {items[2]}",
         "Sugerido_k": ratio_to_slider_nearest(implied_23),
         "Ratio_implicado": implied_23},

        {"Cambiar": f"{items[0]} vs {items[1]}",
         "Mantener": f"{items[0]} vs {items[2]} y {items[1]} vs {items[2]}",
         "Sugerido_k": ratio_to_slider_nearest(implied_12),
         "Ratio_implicado": implied_12},
    ]
    return fixes


def pref_text(r: float, left: str, right: str) -> str:
    """Texto tipo A>B, A≈B, B>A según ratio A/B."""
    if r > 1.15:
        return f"**{left}** > **{right}**"
    if r < (1/1.15):
        return f"**{right}** > **{left}**"
    return f"**{left}** ≈ **{right}**"


def top_pair_suggestions(A: np.ndarray, items: list, top_n: int = 3):
    """
    Para n>=4: encuentra qué comparaciones (pares) aportan más a inconsistencia
    agregando errores de tríadas. Devuelve top_n sugerencias con k recomendado.
    """
    n = A.shape[0]
    # Acumuladores por par (i,k) con i<k
    acc = {}  # (i,k) -> dict(weight_sum, log_suggest_sum)

    def add_suggestion(i, k, implied_ratio, weight):
        if i > k:
            i, k = k, i
            implied_ratio = 1.0 / implied_ratio  # mantener orientación i/k
        key = (i, k)
        if key not in acc:
            acc[key] = {"w": 0.0, "logr": 0.0}
        acc[key]["w"] += weight
        acc[key]["logr"] += weight * np.log(implied_ratio)

    # Recorre tríadas i<j<k
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # ratios actuales
                a_ij, a_jk, a_ik = A[i,j], A[j,k], A[i,k]

                # 1) edge (i,k) implied by (i,j)*(j,k)
                implied_ik = a_ij * a_jk
                err_ik = abs(np.log(a_ik) - np.log(implied_ik))

                # 2) edge (i,j) implied by (i,k)/(j,k)
                implied_ij = a_ik / a_jk
                err_ij = abs(np.log(a_ij) - np.log(implied_ij))

                # 3) edge (j,k) implied by (i,k)/(i,j)
                implied_jk = a_ik / a_ij
                err_jk = abs(np.log(a_jk) - np.log(implied_jk))

                # Agregar sugerencias (peso=error)
                add_suggestion(i, k, implied_ik, err_ik)
                add_suggestion(i, j, implied_ij, err_ij)
                add_suggestion(j, k, implied_jk, err_jk)

    # Construye ranking
    rows = []
    for (i,k), v in acc.items():
        if v["w"] <= 0:
            continue
        suggested_ratio = float(np.exp(v["logr"] / v["w"]))  # promedio geométrico ponderado
        rows.append({
            "i": i, "k": k,
            "Comparación": f"{items[i]} vs {items[k]}",
            "Severidad": v["w"],
            "Sugerido_k": ratio_to_slider_nearest(suggested_ratio),
            "Ratio_sugerido": suggested_ratio
        })

    rows.sort(key=lambda x: x["Severidad"], reverse=True)
    return rows[:min(top_n, len(rows))]


def top_conflict_explanation(A: np.ndarray, items: list):
    """
    Devuelve una explicación humana para la tríada más conflictiva:
    A>B, B>C, pero C>A (o equivalente).
    """
    n = A.shape[0]
    best = None
    best_err = -1.0

    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                implied_ik = A[i,j] * A[j,k]
                err = abs(np.log(A[i,k]) - np.log(implied_ik))
                if err > best_err:
                    best_err = err
                    best = (i, j, k)

    if best is None:
        return None

    i, j, k = best
    return {
        "triad": (items[i], items[j], items[k]),
        "texts": [
            pref_text(A[i,j], items[i], items[j]),
            pref_text(A[j,k], items[j], items[k]),
            pref_text(A[i,k], items[i], items[k]),
        ],
        "severity": best_err
    }
# ============================================================

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
- Valores hacia la izquierda indican mayor importancia del primer criterio (1,2).
- Valores hacia la derecha indican mayor importancia del segundo criterio (8,9).

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
    
    if "pending_updates" not in st.session_state:
        st.session_state.pending_updates = {}
    if "last_changes" not in st.session_state:
        st.session_state.last_changes = []   # lista de cambios recientes

    # aplica actualizaciones pendientes ANTES de instanciar sliders
    for (qid, new_k) in st.session_state.pending_updates.get(sec_name, []):
        st.session_state[f"k_{sec_name}_{qid}"] = int(new_k)
    st.session_state.pending_updates[sec_name] = []

    qid_by_pair = {(a,b): qid for (a,b,qid) in comps}
    qid_by_pair.update({(b,a): qid for (a,b,qid) in comps})

# Pendientes de aplicar (evita el error de modificar después de instanciar widgets)
    if "pending_updates" not in st.session_state:
        st.session_state.pending_updates = {}

    # Si hay una actualización pendiente para esta sección, la dejamos en session_state
    # ANTES de crear sliders (esto sí es permitido).
    for (qid, new_k) in st.session_state.pending_updates.get(sec_name, []):
        st.session_state[f"k_{sec_name}_{qid}"] = int(new_k)

    # Ya aplicadas -> limpiar
    st.session_state.pending_updates[sec_name] = []

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
        k = st.slider("Escala de preferencia", 1, 9, 5, 1, key=f"k_{sec_name}_{qid}")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"<div style='text-align:left; font-size:20px'><b>1 → {a}</b></div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div style='text-align:right; font-size:20px'><b>{b} ← 9</b></div>", unsafe_allow_html=True)
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
    # completar CR nuevo si hubo cambio aplicado
    for c in reversed(st.session_state.last_changes):
        if c["sec"] == sec_name and c["cr_new"] is None:
            c["cr_new"] = float(CR)
            break
    recent = [c for c in st.session_state.last_changes if c["sec"] == sec_name]

    if recent:
        st.subheader("Cambios aplicados")

        for c in recent[-3:]:
            st.markdown(f"""
    <div style="
    background-color:#0b1220;
    border:1px solid #334155;
    border-left:8px solid #22c55e;
    padding:14px;
    border-radius:12px;
    margin:10px 0;">

    <b>Comparación:</b> {c['pair']}<br>
    <b>Antes:</b> {c['k_old']} — {c['old_txt']}<br>
    <b>Ahora:</b> {c['k_new']} — {c['new_txt']}<br>
    <b>CR:</b> {c['cr_old']:.3f} → <b>{c['cr_new']:.3f}</b>

    </div>
    """, unsafe_allow_html=True)


    ok = CR <= CR_THRESHOLD
    if ok:
        st.success("✅ Consistencia aceptable. Puedes continuar.")
    if not ok:
        st.error("❌ Consistencia alta. Ajusta respuestas para continuar.")

        # Explicación del conflicto (humana)
        conflict = top_conflict_explanation(A_crisp, items)
        if conflict:
            a1, a2, a3 = conflict["triad"]
            st.markdown("### ¿Qué está pasando?")
            st.markdown(
                "Hay una contradicción en una tríada de comparaciones (regla de transitividad). "
                "Por ejemplo, en una parte de tus respuestas se ve algo como:"
            )
            st.markdown(
                f"- {conflict['texts'][0]}\n"
                f"- {conflict['texts'][1]}\n"
                f"- pero también {conflict['texts'][2]}"
            )
            st.caption("Esto hace que el sistema no pueda asignar pesos coherentes sin ajustar alguna comparación.")

        st.markdown("### Opciones para corregir (elige la que menos afecte tu opinión)")

        # Caso 3×3: todas las opciones exactas
        if len(items) == 3:
            fixes = all_fixes_3x3(A_crisp, items)

            st.markdown("""
            <div style="padding:10px 0 5px 0; font-size:16px;">
            Elige <b>UNA</b> opción y ajusta <b>solo</b> la comparación indicada.
            </div>
            """, unsafe_allow_html=True)

            for opt_i, fx in enumerate(fixes, start=1):
                cambiar = fx["Cambiar"]
                mantener = fx["Mantener"]
                k_sug = int(fx["Sugerido_k"])

                st.markdown(f"""
            <div style="
            background-color:#111827;
            border:1px solid #334155;
            border-left:8px solid #22c55e;
            padding:16px;
            border-radius:14px;
            margin:12px 0;
            ">

            <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
            <div style="font-size:18px; font-weight:800;">Opción {opt_i}</div>
            <div style="
                background-color:#22c55e;
                color:#111827;
                padding:8px 12px;
                border-radius:999px;
                font-weight:900;
                font-size:18px;">
                Mover a: {k_sug}
            </div>
            </div>

            <div style="margin-top:10px; font-size:16px; line-height:1.55;">
            <b>Qué cambiar:</b> {cambiar}<br>
            <b>Qué mantener:</b> {mantener}
            </div>

            <div style="margin-top:10px; font-size:14px; color:#cbd5e1;">
            Tip: Ajusta esa sola comparación, revisa el CR y continúa.
            </div>

            </div>
            """, unsafe_allow_html=True)

                 # Parsear "X vs Y" para obtener X y Y
                left, right = cambiar.split(" vs ", 1)

                qid = qid_by_pair.get((left, right))
                k_to_set = k_sug

                if qid is None:
                    qid = qid_by_pair.get((right, left))
                    if qid is not None:
                        k_to_set = invert_slider(k_sug)

                if qid is not None:
                    if st.button(f"Aplicar sugerencia {opt_i}", key=f"apply_{sec_name}_{opt_i}"):
                        key_slider = f"k_{sec_name}_{qid}"
                        k_old = int(st.session_state.get(key_slider, 5))
                        cr_old = float(CR)

                        # Registrar "antes" (para mostrar después del rerun)
                        st.session_state.last_changes.append({
                            "sec": sec_name,
                            "pair": f"{left} vs {right}",
                            "k_old": k_old,
                            "k_new": int(k_to_set),
                            "old_txt": k_to_text(k_old, left, right),
                            "new_txt": k_to_text(int(k_to_set), left, right),
                            "cr_old": cr_old,
                            "cr_new": None  # lo llenaremos tras recalcular
                        })
                        st.session_state.pending_updates.setdefault(sec_name, []).append((qid, int(k_to_set)))
                        st.rerun()

            st.info("Guía rápida: 1 favorece el criterio de la izquierda, 9 favorece el de la derecha, 5 = iguales.")

 

        # Caso 4×4 o mayor: Top 3 comparaciones más responsables
        else:
            sugg = top_pair_suggestions(A_crisp, items, top_n=3)

            st.markdown("""
            <div style="padding:10px 0 5px 0; font-size:16px;">
            Elige <b>UNA</b> sugerencia (la que menos afecte tu opinión) y ajusta <b>solo</b> esa comparación.
            </div>
            """, unsafe_allow_html=True)

            for opt_i, s in enumerate(sugg, start=1):
                comp = s["Comparación"]              # "Item_i vs Item_k"
                k_sug = int(s["Sugerido_k"])
                left = items[s["i"]]
                right = items[s["k"]]

                st.markdown(f"""
            <div style="
            background-color:#111827;
            border:1px solid #334155;
            border-left:8px solid #22c55e;
            padding:16px;
            border-radius:14px;
            margin:12px 0;
            ">

            <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
            <div style="font-size:18px; font-weight:800;">Sugerencia {opt_i}</div>
            <div style="
                background-color:#22c55e;
                color:#111827;
                padding:8px 12px;
                border-radius:999px;
                font-weight:900;
                font-size:18px;">
                Mover a: {k_sug}
            </div>
            </div>

            <div style="margin-top:10px; font-size:16px; line-height:1.55;">
            <b>Qué ajustar:</b> {left} vs {right}
            </div>

            <div style="margin-top:10px; font-size:14px; color:#cbd5e1;">
            Tip: Ajusta solo esta comparación y revisa el CR.
            </div>

            </div>
            """, unsafe_allow_html=True)

                # Botón para aplicar automáticamente
                qid = qid_by_pair.get((left, right))
                k_to_set = k_sug

                # Si el par está en orden invertido en tus preguntas, invierte el k
                if qid is None:
                    qid = qid_by_pair.get((right, left))
                    if qid is not None:
                        k_to_set = invert_slider(k_sug)

                if qid is not None:
                    if st.button(f"Aplicar sugerencia {opt_i}", key=f"apply_{sec_name}_{opt_i}"):
                        # Guardar la actualización para el siguiente rerun (antes de crear widgets)
                        key_slider = f"k_{sec_name}_{qid}"
                        k_old = int(st.session_state.get(key_slider, 5))
                        cr_old = float(CR)

                        # Registrar "antes" (para mostrar después del rerun)
                        st.session_state.last_changes.append({
                            "sec": sec_name,
                            "pair": f"{left} vs {right}",
                            "k_old": k_old,
                            "k_new": int(k_to_set),
                            "old_txt": k_to_text(k_old, left, right),
                            "new_txt": k_to_text(int(k_to_set), left, right),
                            "cr_old": cr_old,
                            "cr_new": None  # lo llenaremos tras recalcular
                        })
                        st.session_state.pending_updates.setdefault(sec_name, []).append((qid, int(k_to_set)))
                        st.rerun()
                else:
                    st.caption("No pude mapear automáticamente esta comparación a una pregunta (pero puedes ajustarla manualmente).")

            st.info("Guía rápida: 1 favorece el criterio de la izquierda, 9 favorece el de la derecha, 5 = iguales.")


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

    items_html = ""
    for a, b, qid in comps:
        k = st.session_state.get(f"k_{sec_name}_{qid}", 5)
        txt = interpret_pair(k, a, b)
        items_html += f"<li style='margin:6px 0;'>{txt}</li>"

    st.markdown(f"""
    <div style="
    background-color:#0b1220;
    border-left:8px solid #3b82f6;
    padding:16px;
    border-radius:10px;
    margin-top:10px;">
    <b style="font-size:18px;">Interpretación de sus respuestas</b>
    <ul style="margin-top:10px; margin-bottom:0; padding-left:22px;">
    {items_html}
    </ul>
    </div>
    """, unsafe_allow_html=True)

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

