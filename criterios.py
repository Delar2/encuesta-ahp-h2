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

ACADEMIC_LEVELS = ["Pregrado", "Especialización", "Maestría", "Doctorado"]

# Escala solicitada por el usuario
SCORE_OPTIONS = [-9, -8, -7, -6, -5, -4, -3, -2, 1, 2, 3, 4, 5, 6, 7, 9]

# La confianza mueve "pasos" dentro de SCORE_OPTIONS
CONFIDENCE_STEPS = {
    "Muy seguro": 0,
    "Moderadamente seguro": 1,
    "Poco seguro": 2,
}
CONFIDENCE_OPTIONS = list(CONFIDENCE_STEPS.keys())

# ============================================================
# CRITERIOS
# ============================================================
CRITERIA = [
    "Muestreo",
    "Analítica"
]

DEFINICIONES = {
    "Muestreo": "conjunto de estrategias y procedimientos para recolectar muestras o mediciones representativas en campo",
    "Analítica": "conjunto de técnicas de laboratorio o caracterización empleadas para analizar e interpretar las muestras recolectadas"
}


# ============================================================
# TEXTO
# ============================================================
def interpret_pair(score: int, a: str, b: str) -> str:
    a_fmt = f"<b><u>{a}</u></b>"
    b_fmt = f"<b><u>{b}</u></b>"

    if score == 1:
        return f"{a_fmt} y {b_fmt} tienen importancia similar."
    elif score < 0:
        return f"{a_fmt} es más importante que {b_fmt}."
    return f"{b_fmt} es más importante que {a_fmt}."


# ============================================================
# FUNCIONES MATEMÁTICAS
# ============================================================
def score_to_ratio(score: int) -> float:
    """
    Conversión discreta directa:
    - negativos -> favorecen al criterio izquierdo
    - positivos -> favorecen al criterio derecho
    - 1 -> igualdad
    """
    score = int(score)

    if score == 1:
        return 1.0

    if score < 0:
        return float(abs(score))
    else:
        return float(1 / score)


def move_score_steps(score: int, steps: int) -> int:
    idx = SCORE_OPTIONS.index(int(score))
    new_idx = max(0, min(len(SCORE_OPTIONS) - 1, idx + steps))
    return SCORE_OPTIONS[new_idx]


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
TOTAL_STEPS = TOTAL_QUESTIONS + 1


# ============================================================
# ESTADO
# ============================================================
def ensure_answer_state():
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    for _, _, qid in COMPARISONS:
        if f"answer_score_{qid}" not in st.session_state:
            st.session_state[f"answer_score_{qid}"] = 1
        if f"answer_conf_{qid}" not in st.session_state:
            st.session_state[f"answer_conf_{qid}"] = CONFIDENCE_OPTIONS[0]

    if "initial_ranking" not in st.session_state:
        st.session_state["initial_ranking"] = CRITERIA[:]

    if "pending_question_load" not in st.session_state:
        st.session_state["pending_question_load"] = False

    if "ui_loaded_qid" not in st.session_state:
        st.session_state["ui_loaded_qid"] = None


def load_current_question_into_ui(force: bool = False):
    if st.session_state.current_step == 0:
        return

    _, _, qid = COMPARISONS[st.session_state.current_step - 1]

    if force or st.session_state.get("ui_loaded_qid") != qid:
        st.session_state["ui_score"] = st.session_state[f"answer_score_{qid}"]
        st.session_state["ui_conf"] = st.session_state[f"answer_conf_{qid}"]
        st.session_state["ui_loaded_qid"] = qid


def save_current_question_from_ui():
    if st.session_state.current_step == 0:
        return
    if "ui_score" not in st.session_state or "ui_conf" not in st.session_state:
        return

    _, _, qid = COMPARISONS[st.session_state.current_step - 1]
    st.session_state[f"answer_score_{qid}"] = int(st.session_state["ui_score"])
    st.session_state[f"answer_conf_{qid}"] = st.session_state["ui_conf"]


def get_initial_ranking():
    return st.session_state["initial_ranking"][:]


def move_rank_item_up(index: int):
    ranking = st.session_state["initial_ranking"][:]
    if index > 0:
        ranking[index - 1], ranking[index] = ranking[index], ranking[index - 1]
    st.session_state["initial_ranking"] = ranking


def move_rank_item_down(index: int):
    ranking = st.session_state["initial_ranking"][:]
    if index < len(ranking) - 1:
        ranking[index + 1], ranking[index] = ranking[index], ranking[index + 1]
    st.session_state["initial_ranking"] = ranking


def request_question_load(step: int):
    step = max(0, min(int(step), TOTAL_QUESTIONS))
    st.session_state.current_step = step
    st.session_state["pending_question_load"] = True


def go_next():
    if st.session_state.current_step == 0:
        request_question_load(1)
        return

    save_current_question_from_ui()
    if st.session_state.current_step < TOTAL_QUESTIONS:
        request_question_load(st.session_state.current_step + 1)


def go_prev():
    if st.session_state.current_step == 0:
        return

    save_current_question_from_ui()

    if st.session_state.current_step > 1:
        request_question_load(st.session_state.current_step - 1)
    else:
        st.session_state.current_step = 0
        st.session_state["pending_question_load"] = False
        st.session_state["ui_loaded_qid"] = None


# ============================================================
# RESULTADOS
# ============================================================
def collect_all_rows_and_results():
    idx_map = {name: i for i, name in enumerate(CRITERIA)}
    crisp_answers = {}
    rows = []

    for q_num, (a, b, qid) in enumerate(COMPARISONS, start=1):
        score = int(st.session_state[f"answer_score_{qid}"])
        conf = st.session_state[f"answer_conf_{qid}"]

        r_crisp = score_to_ratio(score)
        i, j = idx_map[a], idx_map[b]
        crisp_answers[(i, j)] = r_crisp

        step_delta = CONFIDENCE_STEPS[conf]

        score_left = move_score_steps(score, -step_delta)
        score_mid = score
        score_right = move_score_steps(score, step_delta)

        candidate_ratios = [
            score_to_ratio(score_left),
            score_to_ratio(score_mid),
            score_to_ratio(score_right),
        ]
        r_l, r_m, r_u = min(candidate_ratios), sorted(candidate_ratios)[1], max(candidate_ratios)

        rows.append({
            "Question_Number": q_num,
            "Question_ID": qid,
            "Criterion_A": a,
            "Criterion_B": b,
            "score": score,
            "Confidence": conf,
            "Confidence_steps": int(step_delta),
            "TFN_score_left": int(score_left),
            "TFN_score_mid": int(score_mid),
            "TFN_score_right": int(score_right),
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

    for r in rows:
        i = idx_map[r["Criterion_A"]]
        j = idx_map[r["Criterion_B"]]
        l, m, u = r["TFN_ratio_l"], r["TFN_ratio_m"], r["TFN_ratio_u"]
        L[i, j], M[i, j], U[i, j] = l, m, u
        L[j, i], M[j, i], U[j, i] = 1.0 / u, 1.0 / m, 1.0 / l

    fuzzy_A = [[(L[i, j], M[i, j], U[i, j]) for j in range(n)] for i in range(n)]
    w_fuzzy_tfn, w_fuzzy_def = fuzzy_weights_geometric_mean(fuzzy_A)

    return rows, A_crisp, lam, CI, CR, w_crisp, L, M, U, w_fuzzy_tfn, w_fuzzy_def


def current_ahp_ranking(w_crisp):
    ranking = list(zip(CRITERIA, w_crisp))
    ranking.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranking]


# ============================================================
# ESTILOS
# ============================================================
st.set_page_config(page_title="Encuesta AHP Estrategia", layout="centered")

st.markdown("""
<style>
/* Oculta el label del select_slider */
div[data-testid="stSelectSlider"] label {
    display: none !important;
}

/* Aumenta tamaño del valor seleccionado */
div[data-testid="stSelectSlider"] * {
    font-size: 1.08rem !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# UI
# ============================================================
ensure_answer_state()

if st.session_state.current_step > 0:
    if (
        "ui_score" not in st.session_state
        or "ui_conf" not in st.session_state
        or st.session_state.get("pending_question_load", False)
    ):
        load_current_question_into_ui(force=True)
        st.session_state["pending_question_load"] = False

st.title("Encuesta AHP — Comparación entre Muestreo y Analítica")
st.caption("Encuesta realizada por Juan Pardo y Salim Shalom")

st.markdown("""
### Descripción de la encuesta

El objetivo de esta encuesta es determinar, de manera estructurada y transparente, la importancia relativa entre los criterios **Muestreo** y **Analítica** para definir el valor técnico de la estrategia de caracterización de filtraciones superficiales de H₂ en “círculos de hadas”.

### Enfoque metodológico

Se utilizará el método **AHP (Analytic Hierarchy Process)** para comparar ambos criterios de forma pareada y obtener su peso relativo dentro de la estructura de decisión.

### Consideraciones importantes

- No existe una respuesta correcta o incorrecta.
- La comparación debe realizarse pensando en:  
  **¿Qué criterio es más importante para definir el valor técnico de la estrategia de caracterización?**
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

if (
    st.session_state.current_step > 0
    and "ui_score" in st.session_state
    and "ui_conf" in st.session_state
    and not st.session_state.get("pending_question_load", False)
):
    save_current_question_from_ui()

rows, A_crisp, lam, CI, CR, w_crisp, L, M, U, w_fuzzy_tfn, w_fuzzy_def = collect_all_rows_and_results()

st.header("Progreso")
step_num = st.session_state.current_step + 1
st.progress(step_num / TOTAL_STEPS)
st.caption(f"Paso {step_num} de {TOTAL_STEPS}")

if st.session_state.current_step == 0:
    st.header("Paso 1 — Orden inicial de criterios")
    st.markdown("Use los botones para mover los criterios. **Arriba = más importante**, **abajo = menos importante**.")

    ranking = get_initial_ranking()

    for idx, item in enumerate(ranking):
        c1, c2, c3 = st.columns([6, 1, 1])
        with c1:
            st.markdown(
                f"""
                <div style="
                background-color:#111827;
                border:1px solid #334155;
                padding:14px;
                border-radius:12px;
                margin:6px 0;">
                <b>{idx + 1}.</b> {item}
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.button("↑", key=f"rank_up_{idx}", on_click=move_rank_item_up, args=(idx,), disabled=(idx == 0), use_container_width=True)
        with c3:
            st.button("↓", key=f"rank_down_{idx}", on_click=move_rank_item_down, args=(idx,), disabled=(idx == len(ranking) - 1), use_container_width=True)

    st.markdown("### Navegación")
    _, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        st.markdown(
            "<div style='text-align:center; padding-top:8px; font-weight:700;'>Orden inicial de criterios</div>",
            unsafe_allow_html=True
        )
    with right_col:
        st.button("Siguiente ➡️", key="next_from_ranking", on_click=go_next, use_container_width=True)

else:
    current_idx = st.session_state.current_step - 1
    a, b, qid = COMPARISONS[current_idx]

    st.header(f"Pregunta #{st.session_state.current_step}")
    st.subheader(f"{a} vs {b}")

    def_a = DEFINICIONES[a]
    def_b = DEFINICIONES[b]

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
    Seleccione la intensidad de preferencia.
    <br>
    Marque <b>1</b> si cree que ambos son igualmente necesarios.

    </div>
    """, unsafe_allow_html=True)

    # Barra visual de color + etiquetas
    st.markdown("""
    <div style="margin: 10px 0 6px 0;">
        <div style="
            height: 16px;
            border-radius: 999px;
            background: linear-gradient(
                90deg,
                #b91c1c 0%,
                #ef4444 18%,
                #f59e0b 35%,
                #eab308 46%,
                #22c55e 50%,
                #eab308 54%,
                #f59e0b 65%,
                #ef4444 82%,
                #b91c1c 100%
            );
            border: 1px solid #334155;
        "></div>
        <div style="
            display:flex;
            justify-content:space-between;
            font-size:15px;
            font-weight:600;
            margin-top:8px;
            gap:8px;
        ">
            <span>Extremadamente importante</span>
            <span>Moderado</span>
            <span>Igual importancia</span>
            <span>Moderado</span>
            <span>Extremadamente importante</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.select_slider(
        label="",
        options=SCORE_OPTIONS,
        value=1,
        key="ui_score"
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown(f"<div style='text-align:left; font-size:22px; font-weight:700;'><b>-9 → {a}</b></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='text-align:center; font-size:22px; font-weight:700;'><b>1 = iguales</b></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div style='text-align:right; font-size:22px; font-weight:700;'><b>{b} ← 9</b></div>", unsafe_allow_html=True)

    st.selectbox("¿Qué tan seguro está de esta evaluación?", CONFIDENCE_OPTIONS, key="ui_conf")

    st.markdown(f"""
    <div style="
    background-color:#0b1220;
    border-left:8px solid #3b82f6;
    padding:14px;
    border-radius:10px;
    margin-top:10px;">
    <b>Interpretación actual:</b> {interpret_pair(int(st.session_state['ui_score']), a, b)}
    </div>
    """, unsafe_allow_html=True)

    st.header("Resultado de la comparación")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("λmax", f"{lam:.4f}")
    with m2:
        st.metric("CI", f"{CI:.4f}")
    with m3:
        st.metric("CR global", f"{CR:.4f}")

    st.success("✅ Con dos criterios, la comparación queda registrada directamente.")

st.header("Finalizar")

if st.button("Enviar respuestas", key="send_responses_button"):
    if st.session_state.current_step > 0 and "ui_score" in st.session_state and "ui_conf" in st.session_state:
        save_current_question_from_ui()
        rows, A_crisp, lam, CI, CR, w_crisp, L, M, U, w_fuzzy_tfn, w_fuzzy_def = collect_all_rows_and_results()

    if not email_enabled():
        st.error("No se puede enviar el archivo porque faltan los datos SMTP en st.secrets.")
    else:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        initial_rank = get_initial_ranking()
        ahp_rank = current_ahp_ranking(w_crisp)

        df_pairwise = pd.DataFrame(rows)
        df_pairwise.insert(0, "Respondent_Name", respondent_name.strip())
        df_pairwise.insert(1, "Profession", profession.strip())
        df_pairwise.insert(2, "Academic_Level", academic_level)
        df_pairwise.insert(3, "Timestamp", ts)

        df_initial_rank = pd.DataFrame({
            "Posición_inicial": list(range(1, len(initial_rank) + 1)),
            "Criterio": initial_rank
        })

        df_current_rank = pd.DataFrame({
            "Posición_AHP": list(range(1, len(ahp_rank) + 1)),
            "Criterio": ahp_rank
        })

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

        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df_participant.to_excel(writer, index=False, sheet_name="Participante")
            df_initial_rank.to_excel(writer, index=False, sheet_name="Orden_Inicial")
            df_current_rank.to_excel(writer, index=False, sheet_name="Orden_AHP")
            df_pairwise.to_excel(writer, index=False, sheet_name="Pairwise")
            df_cr.to_excel(writer, index=False, sheet_name="Consistencia")
            df_crisp_weights.to_excel(writer, index=False, sheet_name="Pesos_Crisp")
            df_fuzzy_weights.to_excel(writer, index=False, sheet_name="Pesos_Fuzzy")
            df_crisp_matrix.to_excel(writer, sheet_name="Matriz_Crisp")
            df_L.to_excel(writer, sheet_name="Fuzzy_L")
            df_M.to_excel(writer, sheet_name="Fuzzy_M")
            df_U.to_excel(writer, sheet_name="Fuzzy_U")

        excel_bytes = bio.getvalue()
        filename = f"AHP_Fuzzy_Muestreo_vs_Analitica_{respondent_name.strip().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        admin = SECRETS["ADMIN_EMAIL"]
        subject = f"Respuesta AHP/Fuzzy Muestreo vs Analítica: {respondent_name.strip()}"
        body = (
            f"Se recibió una nueva respuesta.\n\n"
            f"Participante: {respondent_name.strip()}\n"
            f"Profesión: {profession.strip()}\n"
            f"Nivel académico: {academic_level}\n"
            f"Timestamp: {ts}\n"
            f"CR: {CR:.4f}\n\n"
            f"Se adjunta el archivo Excel con la comparación entre Muestreo y Analítica."
        )

        try:
            send_email(admin, subject, body, excel_bytes, filename)
            st.success("✅ Respuestas enviadas correctamente al email configurado.")
        except Exception as e:
            st.error(f"No se pudo enviar el correo: {e}")
