import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO
from itertools import combinations


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
CR_THRESHOLD = 0.10  # fijo

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

ACADEMIC_LEVELS = ["Pregrado", "Especialización", "Maestría", "Doctorado"]

CRITERIA = [
    "Costo",
    "Rango de detección de H₂",
    "Sensibilidad en la detección de H₂",
    "Detección multigas",
    "Portabilidad y autonomía energética",
    "Robustez operativa"
]

DEFINICIONES = {
    "Costo": "el precio de adquisición del equipo y sus accesorios",
    "Rango de detección de H₂": "el intervalo de medición del hidrógeno natural (por ejemplo: ppm o %vol), desde el valor mínimo hasta el máximo",
    "Sensibilidad en la detección de H₂": "la capacidad de detectar bajas concentraciones de hidrógeno natural (ppm o ppb) y pequeñas variaciones en la señal",
    "Detección multigas": "la capacidad de medir simultáneamente otros gases como CH₄ y N₂",
    "Portabilidad y autonomía energética": "el peso del equipo, la facilidad de transporte y la duración de la batería",
    "Robustez operativa": "la capacidad del equipo para funcionar en condiciones de campo, como temperatura y humedad"
}


# ============================================================
# UTILIDADES DE TEXTO
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


def invert_slider(k: int) -> int:
    return 10 - int(k)


def pref_text(r: float, left: str, right: str) -> str:
    if r > 1.15:
        return f"**{left}** > **{right}**"
    if r < (1 / 1.15):
        return f"**{right}** > **{left}**"
    return f"**{left}** ≈ **{right}**"


# ============================================================
# UTILIDADES MATEMÁTICAS
# ============================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def slider_to_ratio_int(k: int) -> float:
    mapping_left = {1: 9, 2: 7, 3: 5, 4: 3, 5: 1}
    if k <= 5:
        return float(mapping_left[k])
    inv = mapping_left[10 - k]
    return 1.0 / float(inv)


ANCHOR_R = np.array([9, 7, 5, 3, 1, 1/3, 1/5, 1/7, 1/9], dtype=float)


def ratio_from_k(k: float) -> float:
    """
    Mapeo continuo de k (1..9) -> ratio A/B
    usando interpolación log-lineal entre anclas.
    """
    k = clamp(float(k), 1.0, 9.0)

    if float(int(k)) == k:
        return float(ANCHOR_R[int(k) - 1])

    i = int(np.floor(k))
    j = int(np.ceil(k))

    if i == j:
        return float(ANCHOR_R[i - 1])

    ri = ANCHOR_R[i - 1]
    rj = ANCHOR_R[j - 1]
    t = (k - i) / (j - i)

    return float(np.exp((1 - t) * np.log(ri) + t * np.log(rj)))


def build_matrix(n: int, answers: dict) -> np.ndarray:
    A = np.ones((n, n), dtype=float)
    for (i, j), r in answers.items():
        A[i, j] = float(r)
        A[j, i] = 1.0 / float(r)
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


def ratio_to_slider_nearest(r: float) -> int:
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


def top_pair_suggestions(A: np.ndarray, items: list, top_n: int = 3):
    """
    Para n>=4: encuentra qué comparaciones aportan más a la inconsistencia,
    agregando errores de tríadas. Devuelve top_n sugerencias con k recomendado.
    """
    n = A.shape[0]
    acc = {}

    def add_suggestion(i, k, implied_ratio, weight):
        if i > k:
            i, k = k, i
            implied_ratio = 1.0 / implied_ratio
        key = (i, k)
        if key not in acc:
            acc[key] = {"w": 0.0, "logr": 0.0}
        acc[key]["w"] += weight
        acc[key]["logr"] += weight * np.log(implied_ratio)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a_ij, a_jk, a_ik = A[i, j], A[j, k], A[i, k]

                implied_ik = a_ij * a_jk
                err_ik = abs(np.log(a_ik) - np.log(implied_ik))

                implied_ij = a_ik / a_jk
                err_ij = abs(np.log(a_ij) - np.log(implied_ij))

                implied_jk = a_ik / a_ij
                err_jk = abs(np.log(a_jk) - np.log(implied_jk))

                add_suggestion(i, k, implied_ik, err_ik)
                add_suggestion(i, j, implied_ij, err_ij)
                add_suggestion(j, k, implied_jk, err_jk)

    rows = []
    for (i, k), v in acc.items():
        if v["w"] <= 0:
            continue
        suggested_ratio = float(np.exp(v["logr"] / v["w"]))
        rows.append({
            "i": i,
            "k": k,
            "Comparación": f"{items[i]} vs {items[k]}",
            "Severidad": v["w"],
            "Sugerido_k": ratio_to_slider_nearest(suggested_ratio),
            "Ratio_sugerido": suggested_ratio
        })

    rows.sort(key=lambda x: x["Severidad"], reverse=True)
    return rows[:min(top_n, len(rows))]


def top_conflict_explanation(A: np.ndarray, items: list):
    """
    Devuelve una explicación humana para la tríada más conflictiva.
    """
    n = A.shape[0]
    best = None
    best_err = -1.0

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                implied_ik = A[i, j] * A[j, k]
                err = abs(np.log(A[i, k]) - np.log(implied_ik))
                if err > best_err:
                    best_err = err
                    best = (i, j, k)

    if best is None:
        return None

    i, j, k = best
    return {
        "triad": (items[i], items[j], items[k]),
        "texts": [
            pref_text(A[i, j], items[i], items[j]),
            pref_text(A[j, k], items[j], items[k]),
            pref_text(A[i, k], items[i], items[k]),
        ],
        "severity": best_err
    }


# ============================================================
# FUZZY AHP (TFN)
# ============================================================
def tfn_mul(a, b):
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def tfn_pow(a, p):
    return (a[0] ** p, a[1] ** p, a[2] ** p)


def tfn_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def tfn_inv(a):
    return (1.0 / a[2], 1.0 / a[1], 1.0 / a[0])


def tfn_div(a, b):
    return tfn_mul(a, tfn_inv(b))


def tfn_defuzz(a):
    return (a[0] + a[1] + a[2]) / 3.0


def fuzzy_weights_geometric_mean(fuzzy_A):
    n = len(fuzzy_A)
    g = []

    for i in range(n):
        prod = (1.0, 1.0, 1.0)
        for j in range(n):
            prod = tfn_mul(prod, fuzzy_A[i][j])
        g.append(tfn_pow(prod, 1.0 / n))

    sum_g = (0.0, 0.0, 0.0)
    for gi in g:
        sum_g = tfn_add(sum_g, gi)

    w = [tfn_div(gi, sum_g) for gi in g]
    w_def = np.array([tfn_defuzz(wi) for wi in w], dtype=float)
    w_def = w_def / w_def.sum()
    return w, w_def.tolist()


# ============================================================
# SECRETS / EMAIL
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
# ESTRUCTURA DE LA ENCUESTA
# ============================================================
def generate_comparisons(criteria):
    comps = []
    for idx, (a, b) in enumerate(combinations(criteria, 2), start=1):
        qid = f"Q{idx}"
        comps.append((a, b, qid))
    return comps


COMPARISONS = generate_comparisons(CRITERIA)


# ============================================================
# UI BASE
# ============================================================
st.set_page_config(page_title="Encuesta AHP H₂", layout="centered")
st.title("Encuesta AHP — Selección de medidores para la detección de hidrógeno natural en campo")
st.caption("Encuesta realizada por Juan Pardo y Salim Shalom")

st.markdown("""
### Descripción de la encuesta

Esta encuesta tiene como objetivo determinar, de manera estructurada y transparente, cuáles son los criterios técnicos más relevantes para la **selección de medidores para la detección de hidrógeno natural en campo**.

Para establecer la importancia relativa de los criterios se utiliza el método **AHP (Analytic Hierarchy Process)** y su extensión **Fuzzy AHP**, lo que permite incorporar también la incertidumbre asociada a las decisiones de los expertos.

Los pesos obtenidos representarán las prioridades técnicas del estudio y podrán ser utilizados posteriormente en un modelo de optimización para seleccionar la mejor combinación de equipos, considerando tanto el desempeño técnico como las restricciones económicas del proyecto.

### Consideraciones importantes
- No existe una respuesta correcta o incorrecta.
- Las comparaciones deben realizarse pensando en: **¿Qué criterio es más importante para garantizar la calidad técnica en la caracterización de filtraciones de hidrógeno?**
- La escala va de **1 a 9**.
- **5** indica que ambos criterios tienen igual importancia.
- Valores hacia la izquierda favorecen el criterio de la izquierda.
- Valores hacia la derecha favorecen el criterio de la derecha.
- También deberá indicar su **nivel de confianza** en cada comparación.
- El sistema verificará automáticamente la **consistencia** de sus respuestas (**CR ≤ 0.10**).
""")

# ============================================================
# DATOS DEL ENCUESTADO
# ============================================================
st.header("Datos del participante")

respondent_name = st.text_input("Nombre completo *", value="", placeholder="Ej: Juan Pardo")
profession = st.text_input("Profesión *", value="", placeholder="Ej: Ingeniero químico")
academic_level = st.selectbox("Nivel máximo de formación académica *", ACADEMIC_LEVELS)

if not respondent_name.strip() or not profession.strip():
    st.info("Complete su nombre y profesión para comenzar.")
    st.stop()

if not email_enabled():
    st.warning("⚠️ Correo no configurado (faltan Secrets). La encuesta funciona, pero no enviará resultados por email.")

if "pending_updates" not in st.session_state:
    st.session_state.pending_updates = []

if "last_changes" not in st.session_state:
    st.session_state.last_changes = []


# ============================================================
# APLICAR ACTUALIZACIONES PENDIENTES ANTES DE CREAR WIDGETS
# ============================================================
for qid, new_k in st.session_state.pending_updates:
    st.session_state[f"k_{qid}"] = int(new_k)
st.session_state.pending_updates = []


# ============================================================
# ENCUESTA PRINCIPAL
# ============================================================
st.header("Comparaciones pareadas")

idx = {name: i for i, name in enumerate(CRITERIA)}
qid_by_pair = {(a, b): qid for (a, b, qid) in COMPARISONS}
qid_by_pair.update({(b, a): qid for (a, b, qid) in COMPARISONS})

crisp_answers = {}
rows = []

for q_num, (a, b, qid) in enumerate(COMPARISONS, start=1):
    st.subheader(f"Pregunta #{q_num}: {a} vs {b}")

    def_a = DEFINICIONES.get(a, "—")
    def_b = DEFINICIONES.get(b, "—")

    st.markdown(f"""
    <div style="
    background-color:#1f2937;
    padding:15px;
    border-radius:10px;
    border-left:6px solid #22c55e;
    font-size:16px;
    line-height:1.6;">

    Recuerde que <b>{a}</b> corresponde a <b>{def_a}</b>, mientras que <b>{b}</b> corresponde a <b>{def_b}</b>.
    <br><br>
    Utilice la escala lineal para desplazar el marcador hacia el criterio que considere predominante.
    <br>
    Marque <b>5</b> si cree que ambos son igualmente necesarios.

    </div>
    """, unsafe_allow_html=True)

    k = st.slider("Escala de preferencia", 1, 9, 5, 1, key=f"k_{qid}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div style='text-align:left; font-size:20px'><b>1 → {a}</b></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='text-align:right; font-size:20px'><b>{b} ← 9</b></div>", unsafe_allow_html=True)

    conf = st.selectbox("¿Qué tan seguro está de esta evaluación?", CONFIDENCE_OPTIONS, key=f"conf_{qid}")

    # Crisp para CR
    r_crisp = slider_to_ratio_int(int(k))
    i, j = idx[a], idx[b]
    crisp_answers[(i, j)] = r_crisp

    # Fuzzy
    d = CONFIDENCE_DELTAS[conf]
    k_l = clamp(k - d, 1.0, 9.0)
    k_m = float(k)
    k_u = clamp(k + d, 1.0, 9.0)

    # Convertir a ratio TFN
    r_l = ratio_from_k(k_u)
    r_m = ratio_from_k(k_m)
    r_u = ratio_from_k(k_l)

    rows.append({
        "Respondent_Name": respondent_name.strip(),
        "Profession": profession.strip(),
        "Academic_Level": academic_level,
        "Criterion_A": a,
        "Criterion_B": b,
        "Question_ID": qid,
        "k": int(k),
        "Confidence": conf,
        "Delta": float(d),
        "TFN_k_l": float(k_l),
        "TFN_k_m": float(k_m),
        "TFN_k_u": float(k_u),
        "Ratio_crisp_for_CR": float(r_crisp),
        "TFN_ratio_l": float(r_l),
        "TFN_ratio_m": float(r_m),
        "TFN_ratio_u": float(r_u),
    })

    st.divider()


# ============================================================
# CÁLCULOS
# ============================================================
A_crisp = build_matrix(len(CRITERIA), crisp_answers)
lam, CI, CR = ahp_cr(A_crisp)
w_crisp = ahp_weights_eigen(A_crisp)

# Completar CR nuevo si hubo cambio aplicado
for c in reversed(st.session_state.last_changes):
    if c["cr_new"] is None:
        c["cr_new"] = float(CR)
        break

# Fuzzy matrices
n = len(CRITERIA)
L = np.ones((n, n), dtype=float)
M = np.ones((n, n), dtype=float)
U = np.ones((n, n), dtype=float)

for r in rows:
    i = idx[r["Criterion_A"]]
    j = idx[r["Criterion_B"]]
    l, m, u = r["TFN_ratio_l"], r["TFN_ratio_m"], r["TFN_ratio_u"]

    L[i, j], M[i, j], U[i, j] = l, m, u
    L[j, i], M[j, i], U[j, i] = 1.0 / u, 1.0 / m, 1.0 / l

fuzzy_A = [[(L[i, j], M[i, j], U[i, j]) for j in range(n)] for i in range(n)]
w_fuzzy_tfn, w_fuzzy_def = fuzzy_weights_geometric_mean(fuzzy_A)


# ============================================================
# MÉTRICAS Y FEEDBACK
# ============================================================
st.header("Consistencia")

c1, c2, c3 = st.columns(3)
c1.metric("λmax", f"{lam:.4f}")
c2.metric("CI", f"{CI:.4f}")
c3.metric("CR", f"{CR:.4f}")

recent = st.session_state.last_changes
if recent:
    st.subheader("Cambios aplicados")
    for c in recent[-3:]:
        if c["cr_new"] is None:
            cr_new_txt = "Recalculando..."
        else:
            cr_new_txt = f"{c['cr_new']:.3f}"

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
        <b>CR:</b> {c['cr_old']:.3f} → <b>{cr_new_txt}</b>

        </div>
        """, unsafe_allow_html=True)

ok = CR <= CR_THRESHOLD

if ok:
    st.success("✅ Consistencia aceptable. Puede finalizar la encuesta.")
else:
    st.error("❌ La consistencia es alta. Ajuste algunas comparaciones antes de finalizar.")

    conflict = top_conflict_explanation(A_crisp, CRITERIA)
    if conflict:
        st.markdown("### ¿Qué está pasando?")
        st.markdown(
            "Hay una contradicción en una tríada de comparaciones (regla de transitividad). "
            "Por ejemplo, una parte de sus respuestas se interpreta así:"
        )
        st.markdown(
            f"- {conflict['texts'][0]}\n"
            f"- {conflict['texts'][1]}\n"
            f"- pero también {conflict['texts'][2]}"
        )
        st.caption("Esto hace que el sistema no pueda asignar pesos completamente coherentes sin ajustar alguna comparación.")

    st.markdown("### Sugerencias de ajuste")
    st.markdown("Elija una sugerencia y ajuste solo esa comparación, o aplíquela automáticamente.")

    sugg = top_pair_suggestions(A_crisp, CRITERIA, top_n=3)

    for opt_i, s in enumerate(sugg, start=1):
        left = CRITERIA[s["i"]]
        right = CRITERIA[s["k"]]
        k_sug = int(s["Sugerido_k"])

        st.markdown(f"""
        <div style="
        background-color:#111827;
        border:1px solid #334155;
        border-left:8px solid #22c55e;
        padding:16px;
        border-radius:14px;
        margin:12px 0;">

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
        Tip: Ajuste solo esta comparación y vuelva a revisar el CR.
        </div>

        </div>
        """, unsafe_allow_html=True)

        qid = qid_by_pair.get((left, right))
        k_to_set = k_sug

        if qid is None:
            qid = qid_by_pair.get((right, left))
            if qid is not None:
                k_to_set = invert_slider(k_sug)

        if qid is not None:
            if st.button(f"Aplicar sugerencia {opt_i}", key=f"apply_{opt_i}"):
                key_slider = f"k_{qid}"
                k_old = int(st.session_state.get(key_slider, 5))
                cr_old = float(CR)

                st.session_state.last_changes.append({
                    "pair": f"{left} vs {right}",
                    "k_old": k_old,
                    "k_new": int(k_to_set),
                    "old_txt": k_to_text(k_old, left, right),
                    "new_txt": k_to_text(int(k_to_set), left, right),
                    "cr_old": cr_old,
                    "cr_new": None
                })

                st.session_state.pending_updates.append((qid, int(k_to_set)))
                st.rerun()

    st.info("Guía rápida: 1 favorece el criterio de la izquierda, 9 favorece el de la derecha, 5 = iguales.")


# ============================================================
# INTERPRETACIÓN DE RESPUESTAS
# ============================================================
items_html = ""
for a, b, qid in COMPARISONS:
    k = st.session_state.get(f"k_{qid}", 5)
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


# ============================================================
# RESUMEN DE PESOS
# ============================================================
st.header("Pesos resultantes")

df_crisp_weights = pd.DataFrame({
    "Criterio": CRITERIA,
    "Peso_crisp": w_crisp
}).sort_values("Peso_crisp", ascending=False)

df_fuzzy_weights = pd.DataFrame({
    "Criterio": CRITERIA,
    "w_l": [float(w[0]) for w in w_fuzzy_tfn],
    "w_m": [float(w[1]) for w in w_fuzzy_tfn],
    "w_u": [float(w[2]) for w in w_fuzzy_tfn],
    "Peso_fuzzy_defuzz": w_fuzzy_def
}).sort_values("Peso_fuzzy_defuzz", ascending=False)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Pesos crisp")
    st.dataframe(df_crisp_weights, use_container_width=True)

with col_b:
    st.subheader("Pesos fuzzy")
    st.dataframe(df_fuzzy_weights[["Criterio", "Peso_fuzzy_defuzz"]], use_container_width=True)


# ============================================================
# EXPORTACIÓN Y ENVÍO FINAL
# ============================================================
st.header("Finalizar")

if not ok:
    st.error("❌ No puede finalizar mientras el CR sea mayor a 0.10.")
else:
    if st.button("Enviar respuestas"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df_pairwise = pd.DataFrame(rows)
        df_pairwise.insert(1, "Timestamp", ts)

        df_participant = pd.DataFrame([{
            "Respondent_Name": respondent_name.strip(),
            "Profession": profession.strip(),
            "Academic_Level": academic_level,
            "Timestamp": ts,
            "CR": float(CR),
            "Lambda_max": float(lam),
            "CI": float(CI)
        }])

        df_cr = pd.DataFrame([{
            "Respondent_Name": respondent_name.strip(),
            "Timestamp": ts,
            "CR": float(CR),
            "Lambda_max": float(lam),
            "CI": float(CI)
        }])

        df_crisp_matrix = pd.DataFrame(A_crisp, index=CRITERIA, columns=CRITERIA)
        df_L = pd.DataFrame(L, index=CRITERIA, columns=CRITERIA)
        df_M = pd.DataFrame(M, index=CRITERIA, columns=CRITERIA)
        df_U = pd.DataFrame(U, index=CRITERIA, columns=CRITERIA)

        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df_participant.to_excel(writer, index=False, sheet_name="Participante")
            df_pairwise.to_excel(writer, index=False, sheet_name="Pairwise")
            df_cr.to_excel(writer, index=False, sheet_name="Consistencia")
            df_crisp_weights.to_excel(writer, index=False, sheet_name="Pesos_Crisp")
            df_fuzzy_weights.to_excel(writer, index=False, sheet_name="Pesos_Fuzzy")
            df_crisp_matrix.to_excel(writer, sheet_name="Matriz_Crisp")
            df_L.to_excel(writer, sheet_name="Fuzzy_L")
            df_M.to_excel(writer, sheet_name="Fuzzy_M")
            df_U.to_excel(writer, sheet_name="Fuzzy_U")

        excel_bytes = bio.getvalue()
        filename = f"AHP_Fuzzy_Medidores_H2_{respondent_name.strip().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        st.download_button(
            label="📥 Descargar archivo Excel",
            data=excel_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        if email_enabled():
            admin = SECRETS["ADMIN_EMAIL"]
            subject = f"Respuesta AHP/Fuzzy Medidores H2: {respondent_name.strip()}"
            body = (
                f"Se recibió una nueva respuesta.\n\n"
                f"Participante: {respondent_name.strip()}\n"
                f"Profesión: {profession.strip()}\n"
                f"Nivel académico: {academic_level}\n"
                f"Timestamp: {ts}\n"
                f"CR: {CR:.4f}\n\n"
                f"Se adjunta el archivo Excel con respuestas, matrices y pesos."
            )
            try:
                send_email(admin, subject, body, excel_bytes, filename)
                st.success("✅ Respuestas enviadas correctamente y correo enviado.")
            except Exception as e:
                st.warning(f"Se generó el archivo, pero no se pudo enviar el correo: {e}")
        else:
            st.success("✅ Respuestas procesadas correctamente.")
