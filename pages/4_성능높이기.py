import inspect
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "성능 높이기"

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon="📈", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/dbswlgnqudtls/data/"
    "refs/heads/main/stroke.csv"
)
RANDOM_STATE = 42

# 입력으로 쓸 수 있는 속성 다섯 개 (열 이름 → 우리말 이름)
FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KOREAN = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}  # 0 또는 1만 가지는 속성
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

# 화면에 적는 모델 이름 (교과서의 이름을 앞에)
NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"

NOT_AVAILABLE = "계산할 수 없음"
BEFORE = "가중치 맞추기 전"
AFTER = "가중치 맞춘 뒤"

# 색
POINT_BLUE = "#3b78b8"     # 실제 뇌졸중 아님(점)
POINT_RED = "#d62728"      # 실제 뇌졸중(점)
BEFORE_COLOR = "#2ca02c"   # 가중치를 맞추기 전의 0.5 경계(초록색 점선 / 평면)
AFTER_COLOR = "#7b2cbf"    # 가중치를 맞춘 뒤의 0.5 경계(보라색 실선 / 평면)


# ------------------------------------------------------------------
# 도우미 함수
# ------------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def fmt_num(v) -> str:
    """숫자를 소수 둘째 자리까지, 끝의 0은 지워서 적는다."""
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def show_plot(fig):
    """Streamlit 버전에 따라 폭 옵션이 달라서, 있을 때만 넘긴다."""
    kwargs = {}
    if "use_container_width" in inspect.signature(st.plotly_chart).parameters:
        kwargs["use_container_width"] = True
    st.plotly_chart(fig, **kwargs)


def cm_cells(y_true, y_pred):
    """혼동행렬의 네 칸. 뇌졸중(1)이 양성."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp, tn = (int(v) for v in cm.ravel())
    return tp, fn, fp, tn


def metrics_from_cells(tp, fn, fp, tn) -> dict:
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 구할 수 없으면 글로 적는다."""
    total = tp + fn + fp + tn
    actual_pos = tp + fn      # 실제 뇌졸중인 사람 수 (재현율의 분모)
    pred_pos = tp + fp        # 뇌졸중이라 예측한 사람 수 (정밀도의 분모)

    accuracy = (tp + tn) / total if total else None
    recall = tp / actual_pos if actual_pos else None
    precision = tp / pred_pos if pred_pos else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0  # 맞힌 뇌졸중이 하나도 없으면 F1은 0으로 둔다
    else:
        f1 = 2 * precision * recall / (precision + recall)

    def show(v):
        return NOT_AVAILABLE if v is None else pct(v)

    return {
        "정확도": show(accuracy),
        "재현율": show(recall),
        "정밀도": show(precision),
        "F1": show(f1),
    }


def before_after_table(c_before, c_after) -> pd.DataFrame:
    """네 칸과 네 지표를 전·후로 나란히 놓은 표."""
    mb = metrics_from_cells(*c_before)
    ma = metrics_from_cells(*c_after)
    labels = [
        "찾아낸 환자 (TP)",
        "놓친 환자 (FN)",
        "헛짚은 사람 (FP)",
        "아님을 맞힌 사람 (TN)",
    ]
    rows = {}
    for i, label in enumerate(labels):
        rows[label] = (f"{c_before[i]:,}명", f"{c_after[i]:,}명")
    for name in ("정확도", "재현율", "정밀도", "F1"):
        rows[name] = (mb[name], ma[name])
    table = pd.DataFrame(rows, index=[BEFORE, AFTER]).T
    table.index.name = "항목"
    return table


def change_line(c_before, c_after) -> str:
    """찾아낸 환자·놓친 환자·헛짚은 사람이 몇 명에서 몇 명으로 바뀌었는지 한 줄."""
    return (
        f"찾아낸 환자는 {c_before[0]:,}명에서 {c_after[0]:,}명으로, "
        f"놓친 환자는 {c_before[1]:,}명에서 {c_after[1]:,}명으로, "
        f"헛짚은 사람은 {c_before[2]:,}명에서 {c_after[2]:,}명으로 바뀌었어요."
    )


def axis_range(feature: str, series: pd.Series):
    """그림의 축 범위. 0/1 속성은 -0.5 ~ 1.5."""
    if feature in BINARY:
        return -0.5, 1.5
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def jitter(ids: pd.Series, mult: int) -> pd.Series:
    """0/1 속성의 점이 겹치지 않게 살짝 흩는 값(번호로 정해서 늘 같다)."""
    return (((ids * mult) % 1000) / 1000 - 0.5) * 0.3


def line_segment(a, b, c, x0, x1, y0, y1):
    """a*x + b*y + c = 0 인 선이 사각형 [x0,x1]x[y0,y1]을 지나는 두 끝점. 안 지나면 None."""
    eps = 1e-12
    pts = []
    if abs(b) > eps:
        for x in (x0, x1):
            y = -(a * x + c) / b
            if y0 - 1e-9 <= y <= y1 + 1e-9:
                pts.append((x, y))
    if abs(a) > eps:
        for y in (y0, y1):
            x = -(b * y + c) / a
            if x0 - 1e-9 <= x <= x1 + 1e-9:
                pts.append((x, y))
    uniq = []
    for p in pts:
        if all(
            abs(p[0] - q[0]) > 1e-9 * (x1 - x0) or abs(p[1] - q[1]) > 1e-9 * (y1 - y0)
            for q in uniq
        ):
            uniq.append(p)
    return uniq[:2] if len(uniq) >= 2 else None


def _cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _dot(u, v):
    return sum(p * q for p, q in zip(u, v))


def plane_polygon(coefs, d, lows, highs):
    """a*x + b*y + c*z + d = 0 인 평면이 상자 [lows]~[highs]를 자른 단면의 꼭짓점들(순서대로).
    상자를 지나지 않으면 None."""
    if all(abs(c) < 1e-12 for c in coefs):
        return None
    corners = [
        (x, y, z)
        for x in (lows[0], highs[0])
        for y in (lows[1], highs[1])
        for z in (lows[2], highs[2])
    ]
    scale = sum(abs(c) * max(abs(lo), abs(hi)) for c, lo, hi in zip(coefs, lows, highs)) + abs(d)
    eps = 1e-9 * max(scale, 1.0)

    def f(p):
        return sum(c * v for c, v in zip(coefs, p)) + d

    pts = [p for p in corners if abs(f(p)) <= eps]  # 평면 위에 있는 꼭짓점
    for i in range(8):
        for j in range(i + 1, 8):
            p, q = corners[i], corners[j]
            if sum(1 for u, v in zip(p, q) if u != v) != 1:  # 모서리만
                continue
            fp, fq = f(p), f(q)
            if fp * fq < 0:  # 모서리의 두 끝이 평면의 반대쪽
                t = fp / (fp - fq)
                pts.append(tuple(u + t * (v - u) for u, v in zip(p, q)))

    spans = [hi - lo for lo, hi in zip(lows, highs)]
    uniq = []
    for p in pts:
        if all(max(abs(p[i] - q[i]) / spans[i] for i in range(3)) > 1e-9 for q in uniq):
            uniq.append(p)
    if len(uniq) < 3:
        return None

    # 단면 위에서 꼭짓점을 돌아가는 순서로 세운다
    cen = tuple(sum(p[i] for p in uniq) / len(uniq) for i in range(3))
    axis = min(range(3), key=lambda i: abs(coefs[i]))
    e = [0.0, 0.0, 0.0]
    e[axis] = 1.0
    u = _cross(coefs, e)
    v = _cross(coefs, u)

    def angle(p):
        w = tuple(p[i] - cen[i] for i in range(3))
        return math.atan2(_dot(w, v), _dot(w, u))

    return sorted(uniq, key=angle)


def side_word(center_value: float) -> str:
    """그림 전체에서 뇌졸중 확률이 0.5보다 낮은지 높은지."""
    return "낮아요" if center_value < 0 else "높아요"


# ------------------------------------------------------------------
# 화면 맨 위: 제목과 입력 속성 고르기
# ------------------------------------------------------------------
st.title(f"📈 {PAGE_TITLE}")
st.caption(
    f"{APP_TITLE} · 뇌졸중인 사람이 아주 드물 때 가중치를 맞추면 결과가 어떻게 달라지는지 살펴봐요. "
    "뇌졸중(stroke = 1)을 양성으로 둬요."
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

st.header("입력으로 사용할 속성 고르기")
chosen = st.multiselect(
    "입력으로 사용할 속성을 골라 주세요. (두 개 이상)",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda k: KOREAN[k],
)
if len(chosen) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개 이상이어야 모델을 만들고 그림을 그릴 수 있어요.")
    st.stop()
selected = [f for f in FEATURES if f in chosen]  # 늘 같은 순서로 쓴다

# ------------------------------------------------------------------
# 데이터 나누기 ('채점표' 페이지와 같은 방법)
# ------------------------------------------------------------------
data = df.sort_values("id").reset_index(drop=True)
is_test = (data.index % 10) < 3  # 열 명씩 묶어 앞 세 명은 테스트용

X_train = data.loc[~is_test, selected].copy()
X_test = data.loc[is_test, selected].copy()
y_train = data.loc[~is_test, "stroke"]
y_test = data.loc[is_test, "stroke"]

notes = [
    "‘채점표’ 페이지와 같은 방법으로 나눴어요. "
    f"훈련용은 {len(X_train):,}명, 테스트용은 {len(X_test):,}명이에요.",
]
if "bmi" in selected:
    bmi_fill = X_train["bmi"].median()  # 훈련용의 중앙값
    X_train["bmi"] = X_train["bmi"].fillna(bmi_fill)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_fill)
    notes.append(
        f"체질량지수는 빈 값이 있어서, 훈련용의 중앙값({fmt_num(bmi_fill)})으로 채웠어요."
    )
for n in notes:
    st.caption(n)

# ------------------------------------------------------------------
# 모델 학습: 가중치를 맞추기 전(기본) / 맞춘 뒤(balanced)
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)  # 크기 맞추기는 훈련용으로만
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

logit_models = {
    "before": LogisticRegression(random_state=RANDOM_STATE).fit(X_train_s, y_train),
    "after": LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE).fit(
        X_train_s, y_train
    ),
}
tree_models = {
    "before": DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
    ).fit(X_train, y_train),
    "after": DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE
    ).fit(X_train, y_train),
}

cells = {
    NAME_LOGIT: {k: cm_cells(y_test, m.predict(X_test_s)) for k, m in logit_models.items()},
    NAME_TREE: {k: cm_cells(y_test, m.predict(X_test)) for k, m in tree_models.items()},
}

# ------------------------------------------------------------------
# 1. 가중치를 맞추기 전과 뒤
# ------------------------------------------------------------------
st.header("1. 가중치를 맞추기 전과 뒤")
st.caption(
    "뇌졸중인 사람이 아주 드물어서, 뇌졸중인 사람과 아닌 사람의 가중치를 훈련용 사람 수에 반비례하게 맞춘 뒤 "
    "다시 학습했어요. 드문 쪽의 틀린 답을 더 무겁게 치는 거예요. "
    "‘맞추기 전’은 ‘분류 모델’ 페이지와 같은 설정이에요."
)
n_test = len(y_test)
n_stroke = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 가운데 실제 뇌졸중인 사람은 {n_stroke:,}명이에요.")
st.caption(
    "재현율의 분모는 실제 뇌졸중인 사람 수, 정밀도의 분모는 뇌졸중이라 예측한 사람 수예요. "
    f"뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 ‘{NOT_AVAILABLE}’으로 적어요."
)

left, right = st.columns(2)
for col, name in ((left, NAME_LOGIT), (right, NAME_TREE)):
    with col:
        cb, ca = cells[name]["before"], cells[name]["after"]
        st.markdown(f"**{name}**")
        st.table(before_after_table(cb, ca))
        st.write(change_line(cb, ca))

# ------------------------------------------------------------------
# 그림에 쓸 준비: 그리지 않은 속성은 테스트 데이터의 중앙값에 세운다
# ------------------------------------------------------------------
med = X_test.median()
ids = data.loc[X_test.index, "id"]
actual_label = y_test.map({1: "뇌졸중", 0: "아님"})


def plane_coefs(model, axes):
    """로지스틱 회귀의 뇌졸중 점수가 0이 되는 선(면):  sum(coef*축) + d = 0.
    축으로 고르지 않은 속성은 중앙값에 세운다. 점수가 0이면 확률이 0.5다."""

    def score(values):
        row = {f: med[f] for f in selected}
        for f, v in zip(axes, values):
            row[f] = v
        frame = pd.DataFrame([row], columns=selected)
        return float(model.decision_function(scaler.transform(frame))[0])

    zero = [0.0] * len(axes)
    d = score(zero)
    coefs = []
    for i in range(len(axes)):
        unit = list(zero)
        unit[i] = 1.0
        coefs.append(score(unit) - d)
    return coefs, d


def fixed_text(axes) -> str:
    others = [f for f in selected if f not in axes]
    if not others:
        return "고른 속성이 모두 축이라, 따로 중앙값에 세워 둔 속성은 없어요."
    fixed = ", ".join(f"{KOREAN[f]} = {fmt_num(med[f])}" for f in others)
    return f"그리지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed}."


# ------------------------------------------------------------------
# 2. 두 속성으로 본 0.5 경계선 (2차원)
# ------------------------------------------------------------------
st.header("2. 두 속성으로 본 0.5 경계선")
sa, sb = st.columns(2)
fx = sa.selectbox("가로축", selected, format_func=lambda k: KOREAN[k])
fy = sb.selectbox(
    "세로축",
    [f for f in selected if f != fx],
    format_func=lambda k: KOREAN[k],
)

x0, x1 = axis_range(fx, X_test[fx])
y0, y1 = axis_range(fy, X_test[fy])

x_plot = X_test[fx] + (jitter(ids, 7919) if fx in BINARY else 0)
y_plot = X_test[fy] + (jitter(ids, 104729) if fy in BINARY else 0)

fig2 = go.Figure()
for label, color, value in (("아님", POINT_BLUE, 0), ("뇌졸중", POINT_RED, 1)):
    mask = (y_test == value).to_numpy()  # 뇌졸중이 위에 그려지도록 나중에
    fig2.add_trace(
        go.Scatter(
            x=x_plot[mask],
            y=y_plot[mask],
            mode="markers",
            name=f"실제: {label} (점)",
            marker=dict(color=color, size=7, opacity=0.75, line=dict(width=0.5, color="white")),
            customdata=X_test.loc[mask, [fx, fy]].to_numpy(),
            hovertemplate=(
                f"{KOREAN[fx]}: %{{customdata[0]}}<br>"
                f"{KOREAN[fy]}: %{{customdata[1]}}<extra>실제: {label}</extra>"
            ),
        )
    )

line_notes = []
for key, label, color, dash, color_name in (
    ("before", BEFORE, BEFORE_COLOR, "dash", "초록색 점선"),
    ("after", AFTER, AFTER_COLOR, "solid", "보라색 실선"),
):
    (ca, cb), c0 = plane_coefs(logit_models[key], [fx, fy])
    segment = line_segment(ca, cb, c0, x0, x1, y0, y1)
    if segment is not None:
        fig2.add_trace(
            go.Scatter(
                x=[segment[0][0], segment[1][0]],
                y=[segment[0][1], segment[1][1]],
                mode="lines",
                line=dict(color=color, width=3, dash=dash),
                name=f"{label}의 0.5 경계선",
                hoverinfo="skip",
            )
        )
        line_notes.append(f"{color_name}은 {label}의 0.5 경계선이에요.")
    else:
        center = ca * (x0 + x1) / 2 + cb * (y0 + y1) / 2 + c0
        line_notes.append(
            f"{label}의 0.5 경계선({color_name})은 이 그림의 범위 밖에 있어요. "
            f"그림 안에서는 어디서나 뇌졸중 확률이 0.5보다 {side_word(center)}."
        )

fig2.update_layout(
    height=620,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
)
for axis_fn, feature, lo, hi in ((fig2.update_xaxes, fx, x0, x1), (fig2.update_yaxes, fy, y0, y1)):
    axis_fn(title_text=KOREAN[feature], range=[lo, hi])
    if feature in BINARY:
        axis_fn(tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])
show_plot(fig2)

st.write(f"{NAME_LOGIT}이 뇌졸중 확률을 0.5로 가르는 선이에요. " + " ".join(line_notes))
st.write(fixed_text([fx, fy]))
if fx in BINARY or fy in BINARY:
    st.caption("0 또는 1만 가지는 속성은 점이 겹치지 않게 살짝 흩어서 찍었어요. 실제 값은 0 또는 1이에요.")

# ------------------------------------------------------------------
# 3. 세 속성으로 본 0.5 평면 (3차원)
# ------------------------------------------------------------------
if len(selected) >= 3:
    st.header("3. 세 속성으로 본 0.5 평면")
    ta, tb, tc = st.columns(3)
    ax_x = ta.selectbox("x축", selected, format_func=lambda k: KOREAN[k])
    ax_y = tb.selectbox(
        "y축",
        [f for f in selected if f != ax_x],
        format_func=lambda k: KOREAN[k],
    )
    ax_z = tc.selectbox(
        "z축",
        [f for f in selected if f not in (ax_x, ax_y)],
        format_func=lambda k: KOREAN[k],
    )
    axes3 = [ax_x, ax_y, ax_z]
    ranges3 = [axis_range(f, X_test[f]) for f in axes3]
    lows = [r[0] for r in ranges3]
    highs = [r[1] for r in ranges3]

    plots3 = [
        X_test[f] + (jitter(ids, mult) if f in BINARY else 0)
        for f, mult in zip(axes3, (7919, 104729, 1299709))
    ]

    fig3 = go.Figure()
    for label, color, value, size in (("아님", POINT_BLUE, 0, 3), ("뇌졸중", POINT_RED, 1, 4)):
        mask = (y_test == value).to_numpy()
        fig3.add_trace(
            go.Scatter3d(
                x=plots3[0][mask],
                y=plots3[1][mask],
                z=plots3[2][mask],
                mode="markers",
                name=f"실제: {label} (점)",
                marker=dict(color=color, size=size, opacity=0.8),
                customdata=X_test.loc[mask, axes3].to_numpy(),
                hovertemplate=(
                    f"{KOREAN[ax_x]}: %{{customdata[0]}}<br>"
                    f"{KOREAN[ax_y]}: %{{customdata[1]}}<br>"
                    f"{KOREAN[ax_z]}: %{{customdata[2]}}<extra>실제: {label}</extra>"
                ),
            )
        )

    plane_notes = []
    for key, label, color, color_name in (
        ("before", BEFORE, BEFORE_COLOR, "초록색 평면"),
        ("after", AFTER, AFTER_COLOR, "보라색 평면"),
    ):
        coefs, d = plane_coefs(logit_models[key], axes3)
        poly = plane_polygon(coefs, d, lows, highs)
        if poly is not None:
            m = len(poly)
            fig3.add_trace(
                go.Mesh3d(
                    x=[p[0] for p in poly],
                    y=[p[1] for p in poly],
                    z=[p[2] for p in poly],
                    i=[0] * (m - 2),
                    j=list(range(1, m - 1)),
                    k=list(range(2, m)),
                    color=color,
                    opacity=0.45,
                    flatshading=True,
                    showlegend=True,
                    name=f"{label}의 0.5 평면",
                    hoverinfo="skip",
                )
            )
            plane_notes.append(f"{color_name}은 {label}의 0.5 평면이에요.")
        else:
            center = sum(c * (lo + hi) / 2 for c, lo, hi in zip(coefs, lows, highs)) + d
            plane_notes.append(
                f"{label}의 0.5 평면({color_name})은 이 상자 밖에 있어요. "
                f"상자 안에서는 어디서나 뇌졸중 확률이 0.5보다 {side_word(center)}."
            )

    scene = {}
    for name, f, lo, hi in zip(("xaxis", "yaxis", "zaxis"), axes3, lows, highs):
        axis_cfg = dict(title=dict(text=KOREAN[f]), range=[lo, hi])
        if f in BINARY:
            axis_cfg.update(tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])
        scene[name] = axis_cfg
    fig3.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(aspectmode="cube", **scene),
        legend=dict(orientation="h", yanchor="top", y=0.0, xanchor="left", x=0),
    )
    show_plot(fig3)

    st.write(
        f"{NAME_LOGIT}이 뇌졸중 확률을 0.5로 가르는 면이에요. " + " ".join(plane_notes)
    )
    st.write(fixed_text(axes3))
    st.caption("마우스로 끌면 그림을 돌려 볼 수 있고, 휠로 크기를 바꿀 수 있어요.")
    if any(f in BINARY for f in axes3):
        st.caption("0 또는 1만 가지는 속성은 점이 겹치지 않게 살짝 흩어서 찍었어요. 실제 값은 0 또는 1이에요.")
else:
    st.caption("속성을 셋 이상 고르면 세 속성으로 보는 3차원 그림도 나와요.")
