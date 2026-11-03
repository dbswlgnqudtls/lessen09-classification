import inspect

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "분류 모델"

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon="🤖", layout="wide")

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
DEFAULT_FEATURES = [f for f in FEATURES if f != "bmi"]

# 화면에 적는 모델 이름 (교과서의 이름을 앞에)
NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델"

# 색
POINT_BLUE = "#3b78b8"      # 실제 뇌졸중 아님(점)
POINT_RED = "#d62728"       # 실제 뇌졸중(점)
REGION_BLUE = "#dfeaf6"     # 트리가 '아님'이라고 답하는 칸(옅은 색)
REGION_RED = "#fbe0dd"      # 트리가 '뇌졸중'이라고 답하는 칸(옅은 색)
LEAF_BLUE = "#b9d3ec"       # 가지 그림: '아님'이라고 답하는 마디
LEAF_RED = "#f4b6b0"        # 가지 그림: '뇌졸중'이라고 답하는 마디
LINE_COLOR = "#7b2cbf"      # 로지스틱 회귀의 0.5 선

st.markdown(
    """
    <style>
    .acc-card {
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 16px;
        padding: 18px 12px 16px 12px;
        text-align: center;
    }
    .acc-card .name { font-size: 0.95rem; font-weight: 600; min-height: 3.2em; }
    .acc-card .cap { font-size: 0.85rem; opacity: 0.7; margin-top: 4px; }
    .acc-card .big { font-size: 2.8rem; font-weight: 700; line-height: 1.2; }
    .acc-card .small-row {
        display: flex; justify-content: space-around;
        font-size: 0.8rem; opacity: 0.75; margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def acc_card(col, name: str, train_acc: float, test_acc: float):
    html = (
        '<div class="acc-card">'
        f'<div class="name">{name}</div>'
        '<div class="cap">테스트 데이터 정확도</div>'
        f'<div class="big">{pct(test_acc)}</div>'
        '<div class="small-row">'
        f"<span>훈련 데이터 {pct(train_acc)}</span>"
        f"<span>테스트 데이터 {pct(test_acc)}</span>"
        "</div></div>"
    )
    col.markdown(html, unsafe_allow_html=True)


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


def build_tree_dot(model, selected, X, y):
    """의사결정트리를 DOT 문자열로 바꾼다.
    마디마다 훈련용 사람 수, 그중 뇌졸중 사람 수, 비율을 적는다."""
    t = model.tree_
    left, right, feat, thr = t.children_left, t.children_right, t.feature, t.threshold

    # 잎에 떨어진 훈련용 사람 수를 세고, 위쪽 마디는 아래 마디를 더해서 구한다
    leaf_of = pd.Series(model.apply(X), index=X.index)
    n_by_leaf = leaf_of.value_counts()
    k_by_leaf = leaf_of[y == 1].value_counts()
    memo = {}

    def stats(node):
        if node in memo:
            return memo[node]
        if left[node] == -1:
            res = (int(n_by_leaf.get(node, 0)), int(k_by_leaf.get(node, 0)))
        else:
            a, b = stats(int(left[node])), stats(int(right[node]))
            res = (a[0] + b[0], a[1] + b[1])
        memo[node] = res
        return res

    def question(node):
        f = selected[int(feat[node])]
        if f in BINARY:  # 값이 0이면 '예'
            return f"{KOREAN[f]}이 없나요?"
        return f"{KOREAN[f]} ≤ {fmt_num(thr[node])} ?"

    lines = [
        "digraph Tree {",
        'graph [bgcolor="white", rankdir=TB, nodesep=0.3, ranksep=0.5];',
        'node [shape=box, fontname="sans-serif", fontcolor="black", color="#555555"];',
        'edge [fontname="sans-serif", fontcolor="black", color="#555555"];',
    ]
    leaf_answers = []
    used = set()
    for node in range(t.node_count):
        n, k = stats(node)
        rate = k / n * 100 if n else 0.0
        counts = f"훈련용 {n:,}명\\n뇌졸중 {k:,}명 ({rate:.1f}%)"
        if left[node] == -1:  # 더 묻지 않고 답을 내는 마디
            ans = int(model.classes_[t.value[node][0].argmax()])
            leaf_answers.append(ans)
            head = "답: 뇌졸중" if ans == 1 else "답: 아님"
            fill = LEAF_RED if ans == 1 else LEAF_BLUE
            style = "rounded,filled"
        else:
            used.add(int(feat[node]))
            head = question(node)
            fill = "#f2f2f2"
            style = "filled"
        lines.append(
            f'n{node} [label="{head}\\n{counts}", style="{style}", fillcolor="{fill}"];'
        )
    for node in range(t.node_count):
        if left[node] != -1:
            lines.append(f'n{node} -> n{int(left[node])} [label="예"];')
            lines.append(f'n{node} -> n{int(right[node])} [label="아니요"];')
    lines.append("}")
    used_features = [selected[i] for i in sorted(used)]
    return "\n".join(lines), leaf_answers, used_features


# ------------------------------------------------------------------
# 화면 맨 위
# ------------------------------------------------------------------
st.title(f"🤖 {PAGE_TITLE}")
st.caption(
    f"{APP_TITLE} · stroke가 1이면 뇌졸중, 0이면 아님이에요. 뇌졸중을 양성으로 둬요."
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ------------------------------------------------------------------
# 1. 입력 속성 고르기
# ------------------------------------------------------------------
st.header("1. 입력으로 사용할 속성 고르기")
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
# 2. 데이터 나누기 (번호 순 → 열 명씩 → 앞 세 명은 테스트용)
# ------------------------------------------------------------------
data = df.sort_values("id").reset_index(drop=True)
is_test = (data.index % 10) < 3

X_train = data.loc[~is_test, selected].copy()
X_test = data.loc[is_test, selected].copy()
y_train = data.loc[~is_test, "stroke"]
y_test = data.loc[is_test, "stroke"]

notes = [
    "번호(id) 순으로 정렬한 뒤 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"훈련용은 {len(X_train):,}명, 테스트용은 {len(X_test):,}명이에요.",
    "크기 맞추기(표준화)는 훈련용 사람들로만 계산해서 로지스틱 회귀에 써요. "
    "의사결정트리는 원래 값을 그대로 써요.",
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
# 모델 만들기
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

logit = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

# ------------------------------------------------------------------
# 3. 정확도 카드
# ------------------------------------------------------------------
st.header("2. 정확도 비교")
c1, c2, c3 = st.columns(3)
acc_card(
    c1,
    NAME_LOGIT,
    accuracy_score(y_train, logit.predict(X_train_s)),
    accuracy_score(y_test, logit.predict(X_test_s)),
)
acc_card(
    c2,
    NAME_TREE,
    accuracy_score(y_train, tree.predict(X_train)),
    accuracy_score(y_test, tree.predict(X_test)),
)
acc_card(
    c3,
    NAME_BASE,
    accuracy_score(y_train, base.predict(X_train)),
    accuracy_score(y_test, base.predict(X_test)),
)

# ------------------------------------------------------------------
# 4. 두 속성으로 그린 산점도
# ------------------------------------------------------------------
st.header("3. 두 속성으로 본 테스트 데이터")
sa, sb = st.columns(2)
fx = sa.selectbox("가로축", selected, format_func=lambda k: KOREAN[k])
fy = sb.selectbox(
    "세로축",
    [f for f in selected if f != fx],
    format_func=lambda k: KOREAN[k],
)

x0, x1 = axis_range(fx, X_test[fx])
y0, y1 = axis_range(fy, X_test[fy])
med = X_test.median()  # 테스트 데이터의 중앙값
others = [f for f in selected if f not in (fx, fy)]


def make_rows(xv, yv):
    """그리지 않은 속성은 중앙값에 세운 입력표를 만든다."""
    cols = {}
    for f in selected:
        if f == fx:
            cols[f] = xv
        elif f == fy:
            cols[f] = yv
        else:
            cols[f] = med[f]
    return pd.DataFrame(cols, columns=selected)


# (1) 의사결정트리가 나눈 칸 (옅은 색)
N = 200
gx = [x0 + (x1 - x0) * i / (N - 1) for i in range(N)]
gy = [y0 + (y1 - y0) * i / (N - 1) for i in range(N)]
grid = pd.MultiIndex.from_product([gy, gx]).to_frame(index=False, name=["gy", "gx"])
tree_pred = tree.predict(make_rows(grid["gx"], grid["gy"]))
z = [tree_pred[i * N : (i + 1) * N].tolist() for i in range(N)]

# (2) 로지스틱 회귀가 0.5로 가르는 선:  a*x + b*y + c = 0
def z_logit(xv, yv):
    row = make_rows([xv], [yv])
    return float(logit.decision_function(scaler.transform(row))[0])


c0 = z_logit(0.0, 0.0)
ca = z_logit(1.0, 0.0) - c0
cb = z_logit(0.0, 1.0) - c0
segment = line_segment(ca, cb, c0, x0, x1, y0, y1)

# (3) 테스트 데이터 점
ids = data.loc[X_test.index, "id"]
x_plot = X_test[fx] + (jitter(ids, 7919) if fx in BINARY else 0)
y_plot = X_test[fy] + (jitter(ids, 104729) if fy in BINARY else 0)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=gx,
        y=gy,
        z=z,
        zmin=0,
        zmax=1,
        colorscale=[
            [0.0, REGION_BLUE],
            [0.5, REGION_BLUE],
            [0.5, REGION_RED],
            [1.0, REGION_RED],
        ],
        showscale=False,
        hoverinfo="skip",
    )
)
for label, color, value in (("아님", POINT_BLUE, 0), ("뇌졸중", POINT_RED, 1)):
    mask = (y_test == value).to_numpy()  # 뇌졸중이 위에 그려지도록 나중에
    fig.add_trace(
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
if segment is not None:
    fig.add_trace(
        go.Scatter(
            x=[segment[0][0], segment[1][0]],
            y=[segment[0][1], segment[1][1]],
            mode="lines",
            line=dict(color=LINE_COLOR, width=3, dash="dash"),
            name=f"{NAME_LOGIT}이 0.5로 가르는 선",
            hoverinfo="skip",
        )
    )
for label, color in (("뇌졸중", REGION_RED), ("아님", REGION_BLUE)):
    fig.add_trace(  # 칸 색을 범례에 보여 주기 위한 빈 점
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="square", size=13, color=color, line=dict(width=1, color="#999999")),
            name=f"{NAME_TREE}의 칸: {label}이라고 답함",
        )
    )

fig.update_layout(
    height=640,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
)
for axis_fn, feature, lo, hi in ((fig.update_xaxes, fx, x0, x1), (fig.update_yaxes, fy, y0, y1)):
    axis_fn(title_text=KOREAN[feature], range=[lo, hi])
    if feature in BINARY:
        axis_fn(tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])
show_plot(fig)

# 그림 아래 설명 글
if others:
    fixed = ", ".join(f"{KOREAN[f]} = {fmt_num(med[f])}" for f in others)
    st.write(f"그리지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed}.")
else:
    st.write("고른 속성이 두 개뿐이라, 따로 중앙값에 세워 둔 속성은 없어요.")

if segment is not None:
    st.write(f"보라색 점선은 {NAME_LOGIT}이 뇌졸중 확률을 0.5로 가르는 선이에요.")
else:
    center = z_logit((x0 + x1) / 2, (y0 + y1) / 2)
    side = "낮아요" if center < 0 else "높아요"
    st.write(
        f"{NAME_LOGIT}이 0.5로 가르는 선은 이 그림의 범위 밖에 있어요. "
        f"그림 안에서는 어디서나 뇌졸중 확률이 0.5보다 {side}."
    )

st.write(f"옅은 색 칸은 {NAME_TREE}이 그림을 나눈 모습이에요.")
if fx in BINARY or fy in BINARY:
    st.caption("0 또는 1만 가지는 속성은 점이 겹치지 않게 살짝 흩어서 찍었어요. 실제 값은 0 또는 1이에요.")

# ------------------------------------------------------------------
# 5. 의사결정트리가 던진 질문
# ------------------------------------------------------------------
st.header("4. 의사결정트리가 던진 질문")
st.caption(
    "‘예’는 왼쪽, ‘아니요’는 오른쪽 가지예요. 빨간 마디는 ‘뇌졸중’, 파란 마디는 ‘아님’이라고 답해요. "
    "마디 안의 사람 수는 훈련용 사람이에요."
)
dot, leaf_answers, used_features = build_tree_dot(tree, selected, X_train, y_train)
st.graphviz_chart(dot)

n_leaf = len(leaf_answers)
n_no = sum(1 for a in leaf_answers if a == 0)
st.write(f"답을 내는 마디는 모두 {n_leaf}칸이고, 그중 {n_no}칸이 ‘아님’이라고 답해요.")
if used_features:
    asked = ", ".join(KOREAN[f] for f in used_features)
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 속성: {asked}.")
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 속성: 없어요. (질문을 하나도 던지지 않았어요.)")
