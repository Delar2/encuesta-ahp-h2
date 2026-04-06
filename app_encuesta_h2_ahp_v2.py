import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO
from itertools import combinations


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
CR_THRESHOLD = 0.10

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
# FUNCIONES DE TEXTO
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


# ============================================================
# FUNCIONES MATEMÁTICAS
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


# ============================================================
# FUZZY AHP
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
# EMAIL
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
# ESTRUCTURA ENCUESTA
# ============================================================
def generate_comparisons(criteria):
    return [(a, b, f"Q{idx}") for idx, (a, b) in enumerate(combinations(criteria, 2), start=1)]


COMPARISONS = generate_comparisons(CRITERIA)
TOTAL_QUESTIONS = len(COMPARISONS)


# ============================================================
# MÉTRICAS DE IMPACTO DE LA PREGUNTA ACTUAL
# ============================================================
def pair_local_inconsistency(A: np.ndarray, pair_i: int, pair_j: int) -> float:
    """
    Mide cuánto contribuye el par (i,j) a la inconsistencia,
    promediando el error logarítmico en todas las tríadas que lo contienen.
    """
    n = A.shape[0]
    errs = []

    for k in range(n):
        if k == pair_i or k == pair_j:
            continue

        i, j = pair_i, pair_j

        implied_ij = A[i, k] / A[j, k]
        err = abs(np.log(A[i, j]) - np.log(implied_ij))
        errs.append(err)

    if not errs:
        return 0.0
    return float(np.mean(errs))


def pair_cr_without_current(A: np.ndarray, pair_i: int, pair_j: int):
    """
    Calcula un CR de referencia quitando el efecto directo del par actual,
    reemplazándolo por el promedio geométrico implicado por los demás criterios.
    """
    n = A.shape[0]
    implied = []

    for k in range(n):
        if k == pair_i or k == pair_j:
            continue
        implied.append(A[pair_i, k] / A[pair_j, k])

    if not implied:
        return ahp_cr(A)[2]

    implied_ratio = float(np.exp(np.mean(np.log(implied))))
    A2 = A.copy()
    A2[pair_i, pair_j] = implied_ratio
    A2[pair_j, pair_i] = 1.0 / implied_ratio

    return ahp_cr(A2)[2]


# ============================================================
# RECOLECTAR RESULTADOS
# ============================================================
def collect_all_rows_and_results():
    idx_map = {name: i for i, name in enumerate(CRITERIA)}
    crisp_answers = {}
    rows = []

    for q_num, (a, b, qid) in enumerate(COMPARISONS, start=1):
        k = int(st.session_state.get(f"k_{qid}", 5))
        conf = st.session_state.get(f"conf_{qid}", CONFIDENCE_OPTIONS[0])

        r_crisp = slider_to_ratio_int(k)
        i, j = idx_map[a], idx_map[b]
        crisp_answers[(i, j)] = r_crisp

        d = CONFIDENCE_DELTAS[conf]
        k_l = clamp(k - d, 1.0, 9.0)
        k_m = float(k)
        k_u = clamp(k + d, 1.0, 9.0)

        r_l = ratio_from_k(k_u)
        r_m = ratio_from_k(k_m)
        r_u = ratio_from_k(k_l)

        rows.append({
            "Question_Number": q_num,
            "Question_ID": qid,
            "Criterion_A": a,
            "Criterion_B": b,
            "k": k,
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

    A_crisp = build_matrix(len(CRITERIA), crisp_answers)
    lam, CI, CR = ahp_cr(A_crisp)
    w_crisp = ahp_weights_eigen(A_crisp)

    n = len(CRITERIA)
    L = np.ones((n, n), dtype=float)
    M = np.ones((n, n), dtype=float)
    U = np.ones((n, n), dtype=float)

    idx_map = {name: i for i, name in enumerate(CRITERIA)}
    for r in rows:
        i = idx_map[r["Criterion_A"]]
        j = idx_map[r["Criterion_B"]]
        l, m, u = r["TFN_ratio_l"], r["TFN_ratio_m"], r["TFN_ratio_u"]
        L[i, j], M[i, j], U[i, j] = l, m, u
        L[j, i], M[j, i], U[j, i] = 1.0 / u, 1.0 / m, 1.0 / l

    fuzzy_A = [[(L[i, j], M[i, j], U[i, j]) for j in range(n)] for i in range(n)]
    w_fuzzy_tfn, w_fuzzy_def = fuzzy_weights_geometric_mean(fuzzy_A)

    return rows, A_crisp, lam, CI, CR, w_crisp, L, M, U, w_fuzzy_tfn, w_fuzzy_def


# ============================================================
# CALLBACKS DE NAVEGACIÓN
# ============================================================
def go_next():
    if st.session_state.current_question < TOTAL_QUESTIONS:
        st.session_state.current_question += 1


def go_prev():
    if st.session_state.current_question > 1:
        st.session_state.current_question -= 1


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Encuesta AHP H₂", layout="centered")

if "current_question" not in st.session_state:
    st.session_state.current_question = 1

for _, _, qid in COMPARISONS:
    if f"k_{qid}" not in st.session_state:
        st.session_state[f"k_{qid}"] = 5
    if f"conf_{qid}" not in st.session_state:
        st.session_state[f"conf_{qid}"] = CONFIDENCE_OPTIONS[0]

st.title("Encuesta AHP — Selección de medidores para la detección de hidrógeno natural en campo")
st.caption("Encuesta realizada por Juan Pardo y Salim Shalom")

st.markdown("""
### Instrucciones
- Cada pantalla muestra **una sola pregunta**.
- Las respuestas **no se modifican automáticamente**.
- La **consistencia global** se calcula en tiempo real.
- También se muestra cómo la **pregunta actual** influye en la consistencia.
""")

st.header("Datos del participante")
respondent_name = st.text_input("Nombre completo *", value="", placeholder="Ej: Juan Pardo")
profession = st.text_input("Profesión *", value="", placeholder="Ej: Ingeniero químico")
academic_level = st.selectbox("Nivel máximo de formación académica *", ACADEMIC_LEVELS)

if not respondent_name.strip() or not profession.strip():
    st.info("Complete su nombre y profesión para comenzar.")
    st.stop()

if not email_enabled():
    st.warning("⚠️ Correo no configurado. La encuesta funciona, pero no podrá enviar resultados al email.")

rows, A_crisp, lam, CI, CR, w_crisp, L, M, U, w_fuzzy_tfn, w_fuzzy_def = collect_all_rows_and_results()
ok = CR <= CR_THRESHOLD

# ============================================================
# NAVEGACIÓN
# ============================================================
st.header("Navegación")
st.progress(st.session_state.current_question / TOTAL_QUESTIONS)
st.caption(f"Pregunta {st.session_state.current_question} de {TOTAL_QUESTIONS}")

nav1, nav2, nav3 = st.columns([1, 1, 1])
with nav1:
    st.metric("λmax", f"{lam:.4f}")
with nav2:
    st.metric("CI", f"{CI:.4f}")
with nav3:
    st.metric("CR global", f"{CR:.4f}")

if ok:
    st.success("✅ Consistencia global aceptable.")
else:
    st.warning("⚠️ La consistencia global aún es alta.")

# ============================================================
# PREGUNTA ACTUAL
# ============================================================
current_idx = st.session_state.current_question - 1
a, b, qid = COMPARISONS[current_idx]
i = CRITERIA.index(a)
j = CRITERIA.index(b)

st.header(f"Pregunta #{st.session_state.current_question}")
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
line-height:1.6;">

Recuerde que <b>{a}</b> corresponde a <b>{def_a}</b>, mientras que <b>{b}</b> corresponde a <b>{def_b}</b>.
<br><br>
Utilice la escala lineal para desplazar el marcador hacia el criterio que considere predominante.
<br>
Marque <b>5</b> si cree que ambos son igualmente necesarios.

</div>
""", unsafe_allow_html=True)

st.slider("Escala de preferencia", 1, 9, key=f"k_{qid}")

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"<div style='text-align:left; font-size:20px'><b>1 → {a}</b></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='text-align:right; font-size:20px'><b>{b} ← 9</b></div>", unsafe_allow_html=True)

st.selectbox("¿Qué tan seguro está de esta evaluación?", CONFIDENCE_OPTIONS, key=f"conf_{qid}")

k_current = int(st.session_state[f"k_{qid}"])
st.markdown(f"""
<div style="
background-color:#0b1220;
border-left:8px solid #3b82f6;
padding:14px;
border-radius:10px;
margin-top:10px;">
<b>Interpretación actual:</b> {interpret_pair(k_current, a, b)}
</div>
""", unsafe_allow_html=True)

# ============================================================
# IMPACTO DE LA PREGUNTA ACTUAL
# ============================================================
st.header("Impacto de esta pregunta en la consistencia")

local_err = pair_local_inconsistency(A_crisp, i, j)
cr_without_current = pair_cr_without_current(A_crisp, i, j)
delta_cr = CR - cr_without_current

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("CR global actual", f"{CR:.4f}")
with m2:
    st.metric("CR si esta pregunta estuviera alineada", f"{cr_without_current:.4f}")
with m3:
    st.metric("Impacto estimado de esta pregunta", f"{delta_cr:+.4f}")

if delta_cr > 0:
    st.info("Esta respuesta parece aumentar la inconsistencia global.")
elif delta_cr < 0:
    st.info("Esta respuesta parece ayudar a reducir la inconsistencia global.")
else:
    st.info("Esta respuesta tiene un efecto neutro o muy pequeño sobre la inconsistencia global.")

st.caption(f"Error local promedio de esta comparación en sus tríadas: {local_err:.4f}")

# ============================================================
# BOTONES DE NAVEGACIÓN
# ============================================================
st.markdown("### Navegador entre preguntas")
b1, b2, b3 = st.columns([1, 2, 1])

with b1:
    st.button(
        "⬅️ Anterior",
        on_click=go_prev,
        disabled=(st.session_state.current_question == 1),
        use_container_width=True
    )

with b2:
    st.markdown(
        f"<div style='text-align:center; padding-top:8px; font-weight:700;'>"
        f"Pregunta {st.session_state.current_question} / {TOTAL_QUESTIONS}"
        f"</div>",
        unsafe_allow_html=True
    )

with b3:
    st.button(
        "Siguiente ➡️",
        on_click=go_next,
        disabled=(st.session_state.current_question == TOTAL_QUESTIONS),
        use_container_width=True
    )

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

p1, p2 = st.columns(2)
with p1:
    st.subheader("Pesos crisp")
    st.dataframe(df_crisp_weights, use_container_width=True)

with p2:
    st.subheader("Pesos fuzzy")
    st.dataframe(df_fuzzy_weights[["Criterio", "Peso_fuzzy_defuzz"]], use_container_width=True)

# ============================================================
# ENVÍO FINAL SOLO POR EMAIL
# ============================================================
st.header("Finalizar")

if not ok:
    st.error("❌ No puede finalizar mientras el CR sea mayor a 0.10.")
else:
    if st.button("Enviar respuestas"):
        if not email_enabled():
            st.error("No se puede enviar el archivo porque faltan los datos SMTP en st.secrets.")
        else:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            df_pairwise = pd.DataFrame(rows)
            df_pairwise.insert(0, "Respondent_Name", respondent_name.strip())
            df_pairwise.insert(1, "Profession", profession.strip())
            df_pairwise.insert(2, "Academic_Level", academic_level)
            df_pairwise.insert(3, "Timestamp", ts)

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
                st.success("✅ Respuestas enviadas correctamente al email configurado.")
            except Exception as e:
                st.error(f"No se pudo enviar el correo: {e}")
