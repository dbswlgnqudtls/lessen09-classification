import inspect

import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "채점표"

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon="📝", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/dbswlgnqudtls/data/"
    "refs/heads/main/stroke.csv"
)
RANDOM_STATE = 42

# '분류 모델' 페이지에서 처음 골라져 있던 속성(체질량지수를 뺀 넷)
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
KOREAN = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

# 화면에 적는 모델 이름 (교과서의 이름을 앞에)
NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "훈련용에서 많은 쪽으로만 답하는 모델"

NOT_AVAILABLE = "계산할 수 없음"
MAX_ROWS = 20

st.markdown(
    """
    <style>
    .cm-table { border-collapse: collapse; width: 100%; text-align: center; }
    .cm-table th, .cm-table td {
        border: 1px solid rgba(128, 128, 128, 0.35);
        padding: 10px 8px;
    }
    .cm-table th { background: rgba(128, 128, 128, 0.12); font-weight: 600; }
    .cm-table .ok  { background: rgba(46, 160, 67, 0.16); }
    .cm-table .bad { background: rgba(214, 39, 40, 0.14); }
    .cm-table .tot { background: rgba(128, 128, 128, 0.08); }
    .cm-table .abbr { font-size: 0.8rem; font-weight: 700; opacity: 0.8; }
    .cm-table .num { font-size: 1.5rem; font-weight: 700; }
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


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def stretch_kwargs() -> dict:
    """Streamlit 버전에 따라 폭 옵션이 달라서, 있을 때만 넘긴다."""
    if "use_container_width" in inspect.signature(st.dataframe).parameters:
        return {"use_container_width": True}
    return {}


def cm_cells(y_true, y_pred):
    """혼동행렬의 네 칸. 뇌졸중(1)이 양성."""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp, tn = (int(v) for v in cm.ravel())
    return tp, fn, fp, tn


def cm_html(tp, fn, fp, tn) -> str:
    def cell(abbr, n):
        return f'<div class="abbr">{abbr}</div><div class="num">{n:,}명</div>'

    total = tp + fn + fp + tn
    return (
        '<table class="cm-table">'
        "<tr><th></th><th>예측: 뇌졸중</th><th>예측: 아님</th><th>합계</th></tr>"
        f'<tr><th>실제: 뇌졸중</th><td class="ok">{cell("TP", tp)}</td>'
        f'<td class="bad">{cell("FN", fn)}</td><td class="tot">{tp + fn:,}명</td></tr>'
        f'<tr><th>실제: 아님</th><td class="bad">{cell("FP", fp)}</td>'
        f'<td class="ok">{cell("TN", tn)}</td><td class="tot">{fp + tn:,}명</td></tr>'
        f'<tr><th>합계</th><td class="tot">{tp + fp:,}명</td>'
        f'<td class="tot">{fn + tn:,}명</td><td class="tot">{total:,}명</td></tr>'
        "</table>"
    )


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


def person_list(sub: pd.DataFrame) -> pd.DataFrame:
    """번호·나이·평균 혈당·고혈압 여부·심장병 여부 목록. 나이가 많은 순."""
    sub = sub.sort_values(["age", "id"], ascending=[False, True])
    out = pd.DataFrame(
        {
            "번호": sub["id"].to_numpy(),
            "나이": sub["age"].to_numpy(),
            "평균 혈당": sub["avg_glucose_level"].to_numpy(),
            "고혈압 여부": sub["hypertension"].map({0: "없음", 1: "있음"}).to_numpy(),
            "심장병 여부": sub["heart_disease"].map({0: "없음", 1: "있음"}).to_numpy(),
        }
    )
    return out


def show_list(title: str, sub: pd.DataFrame):
    total = len(sub)
    lst = person_list(sub)
    if total > MAX_ROWS:
        st.markdown(f"##### {title} · 전체 {total:,}명 (나이가 많은 앞 {MAX_ROWS}명만 보여 줘요)")
        lst = lst.head(MAX_ROWS)
    else:
        st.markdown(f"##### {title} · 전체 {total:,}명")
    if total == 0:
        st.caption("해당하는 사람이 없어요.")
    else:
        st.dataframe(lst, hide_index=True, **stretch_kwargs())


# ------------------------------------------------------------------
# 화면 맨 위
# ------------------------------------------------------------------
st.title(f"📝 {PAGE_TITLE}")
st.caption(
    f"{APP_TITLE} · 테스트 데이터로 모델의 답을 채점해요. 뇌졸중(stroke = 1)을 양성으로 둬요."
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ------------------------------------------------------------------
# 데이터 나누기와 모델 학습 ('분류 모델' 페이지와 같은 방법)
# ------------------------------------------------------------------
data = df.sort_values("id").reset_index(drop=True)
is_test = (data.index % 10) < 3  # 열 명씩 묶어 앞 세 명은 테스트용

X_train = data.loc[~is_test, FEATURES].copy()
X_test = data.loc[is_test, FEATURES].copy()
y_train = data.loc[~is_test, "stroke"]
y_test = data.loc[is_test, "stroke"]

st.caption(
    "‘분류 모델’ 페이지와 같은 방법으로 나누고 학습했어요. "
    f"입력 속성은 {', '.join(KOREAN[f] for f in FEATURES)}이고, "
    f"훈련용은 {len(X_train):,}명, 테스트용은 {len(X_test):,}명이에요."
)

scaler = StandardScaler().fit(X_train)  # 크기 맞추기는 훈련용으로만
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

logit = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

# 테스트 데이터에 대한 예측 (뇌졸중 = 1, 아님 = 0)
preds = {
    NAME_LOGIT: pd.Series(logit.predict(X_test_s), index=X_test.index),
    NAME_TREE: pd.Series(tree.predict(X_test), index=X_test.index),
    NAME_BASE: pd.Series(base.predict(X_test), index=X_test.index),
}
cells = {name: cm_cells(y_test, p) for name, p in preds.items()}

# ------------------------------------------------------------------
# 1. 혼동행렬
# ------------------------------------------------------------------
st.header("1. 혼동행렬")
st.caption("행은 실제, 열은 예측이에요. 숫자는 테스트 데이터의 사람 수예요.")
left, right = st.columns(2)
with left:
    st.markdown(f"**{NAME_LOGIT}**")
    st.markdown(cm_html(*cells[NAME_LOGIT]), unsafe_allow_html=True)
with right:
    st.markdown(f"**{NAME_TREE}**")
    st.markdown(cm_html(*cells[NAME_TREE]), unsafe_allow_html=True)
st.caption(
    "TP: 뇌졸중인데 뇌졸중이라 예측 · FN: 뇌졸중인데 아니라고 예측 · "
    "FP: 아닌데 뇌졸중이라 예측 · TN: 아닌데 아니라고 예측"
)

# ------------------------------------------------------------------
# 2. 네 가지 점수
# ------------------------------------------------------------------
st.header("2. 정확도·재현율·정밀도·F1")
n_test = len(y_test)
n_stroke = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 가운데 실제 뇌졸중인 사람은 {n_stroke:,}명이에요.")

score_table = pd.DataFrame(
    [metrics_from_cells(*cells[name]) for name in (NAME_LOGIT, NAME_TREE, NAME_BASE)],
    index=pd.Index([NAME_LOGIT, NAME_TREE, NAME_BASE], name="모델"),
)
st.table(score_table)
st.caption(
    "정확도 = (TP + TN) ÷ 전체 · 재현율 = TP ÷ (TP + FN), 분모는 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP ÷ (TP + FP), 분모는 뇌졸중이라 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    f"뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 ‘{NOT_AVAILABLE}’으로 적어요."
)

# ------------------------------------------------------------------
# 3. 놓친 환자와 헛짚은 사람
# ------------------------------------------------------------------
st.header("3. 놓친 환자와 헛짚은 사람")
choice = st.radio(
    "어느 모델의 목록을 볼까요?",
    [NAME_LOGIT, NAME_TREE],
    horizontal=True,
)
pred = preds[choice]
tp, fn, fp, tn = cells[choice]

test_rows = data.loc[X_test.index, ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]]
missed = test_rows[((y_test == 1) & (pred == 0)).to_numpy()]  # 놓친 환자
wrong = test_rows[((y_test == 0) & (pred == 1)).to_numpy()]   # 헛짚은 사람

left, right = st.columns(2)
with left:
    show_list("놓친 환자 (실제 뇌졸중인데 아니라고 예측한 사람)", missed)
with right:
    show_list("헛짚은 사람 (아닌데 뇌졸중이라고 예측한 사람)", wrong)

if len(missed) == fn and len(wrong) == fp:
    st.write(
        f"놓친 환자 {len(missed):,}명은 혼동행렬의 FN 칸({fn:,}명)과, "
        f"헛짚은 사람 {len(wrong):,}명은 FP 칸({fp:,}명)과 같아요."
    )
else:
    st.warning(
        f"목록의 사람 수(놓친 환자 {len(missed):,}명, 헛짚은 사람 {len(wrong):,}명)가 "
        f"혼동행렬의 FN 칸({fn:,}명), FP 칸({fp:,}명)과 달라요."
    )
