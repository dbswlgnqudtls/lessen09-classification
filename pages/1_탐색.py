import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "탐색"

st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon="🔍", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/dbswlgnqudtls/data/"
    "refs/heads/main/stroke.csv"
)

STROKE_LABEL = {0: "뇌졸중 없음", 1: "뇌졸중 있음"}
YES_NO_LABEL = {0: "없음", 1: "있음"}


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate_by(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    """col의 값(0/1)마다 뇌졸중 비율(%)을 계산한다."""
    out = (
        df.groupby(col)["stroke"]
        .agg(사람_수="size", 뇌졸중_사람_수="sum", 비율="mean")
        .reset_index()
    )
    out["비율"] = (out["비율"] * 100).round(2)
    out[name] = out[col].map(YES_NO_LABEL)
    return out


def rate_bar(df: pd.DataFrame, col: str, name: str):
    data = stroke_rate_by(df, col, name)
    fig = px.bar(
        data,
        x=name,
        y="비율",
        text="비율",
        title=f"{name} 여부에 따른 뇌졸중 비율",
        labels={"비율": "뇌졸중 비율(%)"},
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_yaxes(range=[0, max(data["비율"].max() * 1.25, 1)])
    return fig


# ------------------------------------------------------------------
# 데이터 준비
# ------------------------------------------------------------------
st.title(f"🔍 {PAGE_TITLE}")
st.caption(f"{APP_TITLE} · 데이터를 그림과 표로 살펴보는 화면이에요.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map(STROKE_LABEL)

# ------------------------------------------------------------------
# 1. 나이와 평균 혈당의 분포
# ------------------------------------------------------------------
st.header("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)
with left:
    fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=40,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 2. 뇌졸중 여부에 따른 비교
# ------------------------------------------------------------------
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
order = {"뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]}
left, right = st.columns(2)
with left:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders=order,
        title="나이 비교",
        labels={"age": "나이"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders=order,
        title="평균 혈당 비교",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

mean_table = (
    df.groupby("뇌졸중 여부")
    .agg(
        사람_수=("stroke", "size"),
        나이_평균=("age", "mean"),
        평균혈당_평균=("avg_glucose_level", "mean"),
    )
    .reindex(order["뇌졸중 여부"])
    .round(2)
    .reset_index()
)
mean_table.columns = ["그룹", "사람 수", "나이 평균", "평균 혈당 평균"]
st.markdown("**두 그룹의 평균값**")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# 3. 고혈압·심장병과 뇌졸중 비율
# ------------------------------------------------------------------
st.header("3. 고혈압·심장병에 따른 뇌졸중 비율")
left, right = st.columns(2)
with left:
    st.plotly_chart(rate_bar(df, "hypertension", "고혈압"), use_container_width=True)
with right:
    st.plotly_chart(rate_bar(df, "heart_disease", "심장병"), use_container_width=True)

# ------------------------------------------------------------------
# 4. bmi가 빈 사람들
# ------------------------------------------------------------------
st.header("4. 체질량지수(bmi)가 빈 사람들")
bmi_missing = df[df["bmi"].isna()]


def summarize(name: str, part: pd.DataFrame) -> dict:
    n = len(part)
    s = int((part["stroke"] == 1).sum())
    return {
        "구분": name,
        "사람 수": n,
        "뇌졸중 사람 수": s,
        "뇌졸중 비율(%)": round(s / n * 100, 2) if n else None,
    }


compare = pd.DataFrame(
    [summarize("bmi가 빈 사람", bmi_missing), summarize("전체", df)]
)
st.dataframe(compare, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# 5. 흡연 상태별 사람 수
# ------------------------------------------------------------------
st.header("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking["흡연 상태"] = smoking["흡연 상태"].fillna("(빈 값)")
st.dataframe(smoking, use_container_width=True, hide_index=True)
