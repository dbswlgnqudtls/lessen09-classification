import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정 (브라우저 탭 제목과 아이콘)
# ------------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/dbswlgnqudtls/data/"
    "refs/heads/main/stroke.csv"
)
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"


# ------------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(s: pd.Series) -> str:
    """열의 값의 종류를 짧은 글로 설명한다."""
    uniq = s.dropna().unique()
    if len(uniq) <= 10:
        try:
            uniq = sorted(uniq)
        except TypeError:
            pass
        return ", ".join(str(v) for v in uniq)
    if pd.api.types.is_numeric_dtype(s):
        return f"숫자 ({s.min():g} ~ {s.max():g}), {len(uniq):,}가지"
    return f"{len(uniq):,}가지 값"


# ------------------------------------------------------------------
# 기록 저장/불러오기 (내_기록 폴더의 json 파일)
# ------------------------------------------------------------------
def load_record() -> dict:
    record = {"meanings": {}, "source": ""}
    try:
        if SAVE_FILE.exists():
            saved = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            record["meanings"] = dict(saved.get("meanings", {}))
            record["source"] = str(saved.get("source", ""))
    except (OSError, ValueError):
        pass
    return record


def save_record(record: dict) -> bool:
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        SAVE_FILE.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except OSError:
        return False


# ------------------------------------------------------------------
# 화면 꾸미기 (큰 숫자 카드)
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .big-card {
        background: rgba(128, 128, 128, 0.12);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 16px;
        padding: 22px 12px;
        text-align: center;
    }
    .big-card .label { font-size: 1rem; opacity: 0.75; margin-bottom: 6px; }
    .big-card .value { font-size: 2.8rem; font-weight: 700; line-height: 1.2; }
    </style>
    """,
    unsafe_allow_html=True,
)


def card(col, label: str, value: str):
    col.markdown(
        f'<div class="big-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# 데이터 준비
# ------------------------------------------------------------------
try:
    df = load_data()
except Exception as e:  # 인터넷 문제 등
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 새로고침 후 처음 한 번만 파일에서 불러온다
if "last_saved" not in st.session_state:
    loaded = load_record()
    st.session_state["last_saved"] = loaded
    st.session_state["meanings_init"] = loaded["meanings"]
    st.session_state["source_text"] = loaded["source"]

# ------------------------------------------------------------------
# 맨 위: 제목
# ------------------------------------------------------------------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("이 데이터가 무엇인지 알아보는 첫 화면이에요.")

# ------------------------------------------------------------------
# 큰 숫자 카드 네 개
# ------------------------------------------------------------------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
card(c1, "전체 사람 수", f"{total:,}명")
card(c2, "열 개수", f"{n_cols}개")
card(c3, "stroke가 1인 사람 수", f"{n_stroke:,}명")
card(c4, "stroke가 1인 비율", f"{ratio:.2f}%")

# ------------------------------------------------------------------
# 열 설명 표 (우리말 뜻은 직접 입력)
# ------------------------------------------------------------------
st.subheader("📋 열 설명 표")
st.caption("‘우리말 뜻’ 칸을 더블클릭해서 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장돼요.")

table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [
            st.session_state["meanings_init"].get(c, "") for c in df.columns
        ],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

edited = st.data_editor(
    table,
    key="column_table",
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻", help="교재를 보고 직접 적어 주세요."
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
)

# ------------------------------------------------------------------
# 데이터 처음 다섯 줄
# ------------------------------------------------------------------
st.subheader("👀 데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# 데이터 출처
# ------------------------------------------------------------------
st.subheader("📚 데이터 출처")
source_text = st.text_area(
    "교재에 있는 출처 글을 여기에 적어 주세요.",
    key="source_text",
    height=120,
)

# ------------------------------------------------------------------
# 자동 저장 (내_기록/기록.json)
# ------------------------------------------------------------------
current = {
    "meanings": {
        str(row["열 이름"]): (row["우리말 뜻"] or "").strip()
        for _, row in edited.iterrows()
        if (row["우리말 뜻"] or "").strip()
    },
    "source": source_text,
}

if current != st.session_state["last_saved"]:
    if save_record(current):
        st.session_state["last_saved"] = current
        st.caption("💾 저장했어요: 내_기록/기록.json")
    else:
        st.warning("저장하지 못했어요. 폴더 쓰기 권한을 확인해 주세요.")
else:
    st.caption("💾 적은 내용은 내_기록/기록.json에 자동 저장돼요.")
