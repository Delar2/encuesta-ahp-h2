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

# ============================================================
# CRITERIOS ACTUALIZADOS
# ============================================================
CRITERIA = [
    "Costo",
    "Rango de detección de H₂",
    "Detección multigas",
    "Portabilidad y autonomía energética"
]

DEFINICIONES = {
    "Costo": "el precio de adquisición del equipo, sus accesorios y los costos de implementación",
    "Rango de detección de H₂": "el intervalo de concentración de hidrógeno que el equipo puede medir, desde valores bajos hasta altos",
    "Detección multigas": "la capacidad del equipo para medir otros gases además del hidrógeno, como CH₄ y N₂",
    "Portabilidad y autonomía energética": "la batería, fuente de alimentación, duración de carga, peso y dimensiones del equipo"
}


# ============================================================
# TEXTO
# ============================================================
def interpret_pair(k: int, a: str, b: str) -> str:
    a_fmt = f"<b><u>{a}</u></b>"
    b_fmt = f"<b><u>{b}</u></b>"
    if k == 5:
        return f"{a_fmt} y {b_fmt} tienen importancia similar."
    elif k < 5:
        return f"{a_fmt} es más importante que {b_fmt}."
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
TOTAL_STEPS = TOTAL_QUESTIONS + 1


# ============================================================
# ESTADO
# ============================================================
def ensure_answer_state():
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0

    for _, _, qid in COMPARISONS:
        if f"answer_k_{qid}" not in st.session_state:
            st.session_state[f"answer_k_{qid}"] = 5
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
        st.session_state["ui_k"] = st.session_state[f"answer_k_{qid}"]
        st.session_state["ui_conf"] = st.session_state[f"answer_conf_{qid}"]
        st.session_state["ui_loaded_qid"] = qid


def save_current_question_from_ui():
    if st.session_state.current_step == 0:
        return
    if "ui_k" not in st.session_state or "ui_conf" not in st.session_state:
        return

    _, _, qid = COMPARISONS[st.session_state.current_step - 1]
    st.session_state[f"answer_k_{qid}"] = int(st.session_state["ui_k"])
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


def go_to_question(question_number: int):
    if st.session_state.current_step > 0:
        save_current_question_from_ui()
    request_question_load(question_number)


# ============================================================
# RESULTADOS
# ============================================================
def collect_all_rows_and_results():
    idx_map = {name: i for i, name in enumerate(CRITERIA)}
    crisp_answers = {}
    rows = []

    for q_num, (a, b, qid) in enumerate(COMPARISONS, start=1):
        k = int(st.session_state[f"answer_k_{qid}"])
        conf = st.session_state[f"answer_conf_{qid}"]

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


def pair_local_inconsistency(A: np.ndarray, pair_i: int, pair_j: int) -> float:
    n = A.shape[0]
    errs = []

    for k in range(n):
        if k == pair_i or k == pair_j:
            continue
        implied_ij = A[pair_i, k] / A[pair_j, k]
        err = abs(np.log(A[pair_i, pair_j]) - np.log(implied_ij))
        errs.append(err)

    if not errs:
        return 0.0
    return float(np.mean(errs))


def top_problematic_pairs(A: np.ndarray, labels: list, top_k: int = 5):
    pair_to_q = {}
    for q_num, (a, b, _) in enumerate(COMPARISONS, start=1):
        pair_to_q[(a, b)] = q_num
        pair_to_q[(b, a)] = q_num

    n = A.shape[0]
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            a = labels[i]
            b = labels[j]
            rows.append({
                "Pregunta": pair_to_q[(a, b)],
                "Comparación": f"{a} vs {b}",
                "Inconsistencia_local": pair_local_inconsistency(A, i, j)
            })

    rows.sort(key=lambda x: x["Inconsistencia_local"], reverse=True)
    return rows[:min(top_k, len(rows))]


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Encuesta AHP H₂", layout="centered")

ensure_answer_state()

if st.session_state.current_step > 0:
    if (
        "ui_k" not in st.session_state
        or "ui_conf" not in st.session_state
        or st.session_state.get("pending_question_load", False)
    ):
        load_current_question_into_ui(force=True)
        st.session_state["pending_question_load"] = False

st.title("Encuesta AHP — Selección de medidores para la detección de hidrógeno natural en campo")
st.caption("Encuesta realizada por Juan Pardo y Salim Shalom")

st.markdown("""
### Descripción de la encuesta

El objetivo de esta encuesta es determinar, de manera estructurada y transparente, cuáles son los criterios técnicos más relevantes para definir la **selección de medidores para la detección de hidrógeno natural en campo**.

### Enfoque metodológico

Para establecer la importancia relativa de los criterios se utilizará el método **AHP (Analytic Hierarchy Process)**. Este método permite comparar diferentes criterios de forma pareada con el fin de determinar sus pesos relativos dentro de la estructura de decisión.

Los pesos obtenidos mediante AHP representarán las prioridades técnicas del estudio.

Posteriormente, estos pesos serán ingresados en un modelo matemático de optimización cuyo objetivo será seleccionar la mejor combinación de equipos, considerando tanto el desempeño técnico como las restricciones económicas del proyecto.

### Consideraciones importantes

- No existe una respuesta correcta o incorrecta.
- Las comparaciones deben realizarse siempre pensando en:  
  **¿Qué criterio es más importante para garantizar la calidad técnica en la caracterización de filtraciones de hidrógeno?**
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
    and "ui_k" in st.session_state
    and "ui_conf" in st.session_state
    and not st.session_state.get("pending_question_load", False)
):
    save_current_question_from_ui()

rows, A_crisp, lam, CI, CR, w_crisp, L, M, U, w_fuzzy_tfn, w_fuzzy_def = collect_all_rows_and_results()
ok = CR <= CR_THRESHOLD

st.header("Progreso")
step_num = st.session_state.current_step + 1
st.progress(step_num / TOTAL_STEPS)
st.caption(f"Paso {step_num} de {TOTAL_STEPS}")

if st.session_state.current_step == 0:
    st.header("Paso 1 — Orden inicial de criterios")
    
    st.markdown("### Contexto de los criterios")

    st.image(
        "imagen_2026-04-08_142656956.png",
        use_container_width=True
    )
    
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
    Utilice la escala lineal para desplazar el marcador hacia el criterio que considere predominante.
    <br>
    Marque <b>5</b> si cree que ambos son igualmente necesarios.

    </div>
    """, unsafe_allow_html=True)

    st.slider("Escala de preferencia", 1, 9, key="ui_k")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div style='text-align:left; font-size:20px'><b>1 → {a}</b></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div style='text-align:right; font-size:20px'><b>{b} ← 9</b></div>", unsafe_allow_html=True)

    st.selectbox("¿Qué tan seguro está de esta evaluación?", CONFIDENCE_OPTIONS, key="ui_conf")

    st.markdown(f"""
    <div style="
    background-color:#0b1220;
    border-left:8px solid #3b82f6;
    padding:14px;
    border-radius:10px;
    margin-top:10px;">
    <b>Interpretación actual:</b> {interpret_pair(int(st.session_state['ui_k']), a, b)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navegador entre preguntas")

    row1_col1, row1_col2, row1_col3 = st.columns([1, 2, 1])
    with row1_col1:
        st.button("⬅️ Anterior", key="nav_prev_main", on_click=go_prev, use_container_width=True)

    with row1_col2:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px; font-weight:700;'>"
            f"Pregunta {st.session_state.current_step} / {TOTAL_QUESTIONS}"
            f"</div>",
            unsafe_allow_html=True
        )

    with row1_col3:
        st.button(
            "Siguiente ➡️",
            key="nav_next_main",
            on_click=go_next,
            disabled=(st.session_state.current_step == TOTAL_QUESTIONS),
            use_container_width=True
        )

    quick_options = list(range(1, TOTAL_QUESTIONS + 1))
    quick_current = st.session_state.current_step
    if quick_current not in quick_options:
        quick_current = 1

    row2_col1, row2_col2 = st.columns([1, 1])
    with row2_col1:
        st.selectbox(
            "Pregunta rápida",
            options=quick_options,
            index=quick_options.index(quick_current),
            key="quick_question_selector",
            format_func=lambda x: f"Pregunta {x}"
        )

    with row2_col2:
        if st.button("Abrir pregunta", key="open_quick_question_button", use_container_width=True):
            go_to_question(int(st.session_state["quick_question_selector"]))

    st.header("Organización de preferencias")

    initial_rank = get_initial_ranking()
    ahp_rank = current_ahp_ranking(w_crisp)

    rc1, rc2 = st.columns(2)
    with rc1:
        st.subheader("Orden inicial")
        for i_rank, c in enumerate(initial_rank, start=1):
            st.markdown(f"**{i_rank}.** {c}")

    with rc2:
        st.subheader("Orden actual según AHP")
        for i_rank, c in enumerate(ahp_rank, start=1):
            st.markdown(f"**{i_rank}.** {c}")

    st.header("Consistencia global")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("λmax", f"{lam:.4f}")
    with m2:
        st.metric("CI", f"{CI:.4f}")
    with m3:
        st.metric("CR global", f"{CR:.4f}")

    if ok:
        st.success("✅ Consistencia global aceptable.")
    else:
        st.warning("⚠️ La consistencia global aún es alta.")

    st.header("Comparaciones más sensibles")
    pairs = top_problematic_pairs(A_crisp, CRITERIA, top_k=6)

    for p in pairs:
        st.markdown(f"""
        <div style="
        background-color:#111827;
        border:1px solid #334155;
        border-left:8px solid #ef4444;
        padding:14px;
        border-radius:12px;
        margin:10px 0;">
        <b>Pregunta {p['Pregunta']}</b><br>
        {p['Comparación']}<br>
        <b>Inconsistencia local:</b> {p['Inconsistencia_local']:.4f}
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"Ir a P{p['Pregunta']}", key=f"go_sensitive_pair_{p['Pregunta']}", use_container_width=True):
            go_to_question(int(p["Pregunta"]))

st.header("Finalizar")

if not ok:
    st.error("❌ No puede finalizar mientras el CR sea mayor a 0.10.")
else:
    if st.button("Enviar respuestas", key="send_responses_button"):
        if st.session_state.current_step > 0 and "ui_k" in st.session_state and "ui_conf" in st.session_state:
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

            df_pairs = pd.DataFrame(top_problematic_pairs(A_crisp, CRITERIA, top_k=15))

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
                df_pairs.to_excel(writer, index=False, sheet_name="Pares_Sensibles")
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
                f"Se adjunta el archivo Excel con ranking inicial, comparaciones AHP, consistencia y comparaciones sensibles."
            )

            try:
                send_email(admin, subject, body, excel_bytes, filename)
                st.success("✅ Respuestas enviadas correctamente al email configurado.")
            except Exception as e:
                st.error(f"No se pudo enviar el correo: {e}")
