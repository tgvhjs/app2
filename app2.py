import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="월별 매출 대시보드", layout="wide")

# 🎨 팔레트 (Toss 톤 라이트)
COLOR_PRIMARY = "#035FFC"   # 핵심 블루
COLOR_ACCENT  = "#91C3F9"   # 보조 라이트 블루
COLOR_TEXT    = "#2E2F2F"   # 본문/축 텍스트(다크 그레이)
COLOR_BG      = "#FFFFFF"   # 카드/페이지 배경 (라이트)
COLOR_GRID    = "rgba(46,47,47,0.18)"  # 그리드(2E2F2F의 투명도 적용)

def parse_month_to_period(s):
    try:
        return pd.Period(str(s).strip(), freq="M")
    except Exception:
        return pd.NaT

def moving_avg(series, k=3):
    return series.rolling(window=k, min_periods=k).mean()

def krw(n):
    try:
        return f"{int(n):,} 원"
    except Exception:
        return "-"

# ─────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────
st.sidebar.title("설정")
uploaded = st.sidebar.file_uploader("CSV 업로드 (월, 매출액, 전년동월, 증감률)", type=["csv"])
target = st.sidebar.number_input("연간 매출 목표(원)", min_value=0, value=0, step=1)
st.sidebar.caption("※ 헤더명은 반드시 `월, 매출액, 전년동월, 증감률`을 사용하세요.")

st.title("📊 월별 매출 대시보드")

if uploaded is None:
    st.info("좌측에서 CSV 파일을 업로드해주세요.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 데이터 처리
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(uploaded)

required_cols = ["월", "매출액", "전년동월", "증감률"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"다음 컬럼이 없습니다: {missing}")
    st.stop()

for c in ["매출액", "전년동월"]:
    df[c] = (
        df[c].astype(str).str.replace(",", "", regex=False).str.strip()
        .replace({"": np.nan}).astype(float)
    )
df["증감률"] = (
    df["증감률"].astype(str).str.replace("%", "", regex=False).str.strip()
    .replace({"": np.nan}).astype(float)
)

df["월_period"] = df["월"].apply(parse_month_to_period)
df = df.sort_values("월_period").reset_index(drop=True)

df["MA3"] = moving_avg(df["매출액"], 3)
df["MoM"] = df["매출액"].diff()
df["누적매출"] = df["매출액"].cumsum()

# ─────────────────────────────────────────────────────────────
# KPI
# ─────────────────────────────────────────────────────────────
ytd_sales = df["누적매출"].iloc[-1] if len(df) else 0
ytd_prev = df["전년동월"].sum(skipna=True)
yoy = ((ytd_sales - ytd_prev) / ytd_prev * 100) if ytd_prev else np.nan

max_idx = df["매출액"].idxmax()
min_idx = df["매출액"].idxmin()

k1, k2, k3, k4 = st.columns(4)
k1.metric("YTD 누적 매출", krw(ytd_sales))
k2.metric("YTD 전년대비 증감률", f"{yoy:.1f} %" if pd.notna(yoy) else "-")
k3.metric("최고 매출 월", f"{df.loc[max_idx,'월']} · {krw(df.loc[max_idx,'매출액'])}")
k4.metric("최저 매출 월", f"{df.loc[min_idx,'월']} · {krw(df.loc[min_idx,'매출액'])}")

st.divider()

# 공통 레이아웃(라이트 테마용)
common_layout = dict(
    height=360,
    margin=dict(t=20, r=20, b=40, l=50),
    paper_bgcolor=COLOR_BG,
    plot_bgcolor=COLOR_BG,
    font=dict(color=COLOR_TEXT),
    xaxis=dict(title="월", tickangle=-30, gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
)

# ─────────────────────────────────────────────────────────────
# ① 매출 추세 & 3M 이동평균
# ─────────────────────────────────────────────────────────────
fig_trend = go.Figure()
fig_trend.add_trace(
    go.Scatter(
        x=df["월"], y=df["매출액"],
        mode="lines+markers", name="매출액",
        line=dict(width=3, color=COLOR_PRIMARY),
        marker=dict(color=COLOR_PRIMARY)
    )
)
fig_trend.add_trace(
    go.Scatter(
        x=df["월"], y=df["MA3"],
        mode="lines", name="3M 이동평균",
        line=dict(dash="dash", color=COLOR_ACCENT)
    )
)
fig_trend.update_layout(**common_layout, yaxis=dict(title="원", gridcolor=COLOR_GRID))
st.subheader("① 매출 추세 & 3M 이동평균")
st.plotly_chart(fig_trend, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ② 전년동월 대비 (이중축)
# ─────────────────────────────────────────────────────────────
fig_yoy = make_subplots(specs=[[{"secondary_y": True}]])
fig_yoy.add_trace(
    go.Scatter(
        x=df["월"], y=df["매출액"],
        name="매출액", mode="lines+markers",
        line=dict(width=3, color=COLOR_PRIMARY),
        marker=dict(color=COLOR_PRIMARY)
    ),
    secondary_y=False
)
fig_yoy.add_trace(
    go.Scatter(
        x=df["월"], y=df["전년동월"],
        name="전년동월", mode="lines+markers",
        line=dict(dash="dot", color=COLOR_ACCENT),
        marker=dict(color=COLOR_ACCENT)
    ),
    secondary_y=True
)
fig_yoy.update_yaxes(title_text="매출액(원)", secondary_y=False, gridcolor=COLOR_GRID)
fig_yoy.update_yaxes(title_text="전년동월(원)", secondary_y=True, showgrid=False)
fig_yoy.update_layout(**common_layout)
st.subheader("② 전년동월 대비")
st.plotly_chart(fig_yoy, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ③ 증감률(%)
# ─────────────────────────────────────────────────────────────
bar_colors = np.where(df["증감률"] >= 0, COLOR_PRIMARY, COLOR_TEXT)  # 하락은 다크그레이
fig_rate = go.Figure(
    data=[
        go.Bar(
            x=df["월"], y=df["증감률"],
            marker=dict(color=bar_colors),
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
            name="증감률(%)",
        )
    ]
)
fig_rate.update_layout(**common_layout, yaxis=dict(title="증감률(%)", gridcolor=COLOR_GRID))
st.subheader("③ 전년동월 증감률(%)")
st.plotly_chart(fig_rate, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ④ 목표 달성도(게이지)
# ─────────────────────────────────────────────────────────────
cum_last = df["누적매출"].iloc[-1] if len(df) else 0
pct = min(100, cum_last / target * 100) if target and target > 0 else None

fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=pct if pct is not None else 0,
        number={"suffix": "%", "font": {"color": COLOR_TEXT}},
        title={
            "text": f"누적/목표 ({krw(cum_last)} / {krw(target)})" if target else "목표 입력 필요",
            "font": {"color": COLOR_TEXT}
        },
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLOR_TEXT},
            "bar": {"color": COLOR_PRIMARY},
            "bgcolor": COLOR_BG,
            "bordercolor": COLOR_GRID,
            "steps": [
                {"range": [0, 50], "color": "rgba(3,95,252,0.08)"},
                {"range": [50, 80], "color": "rgba(3,95,252,0.16)"},
                {"range": [80, 100], "color": "rgba(3,95,252,0.24)"},
            ],
        },
    )
)
fig_gauge.update_layout(height=360, paper_bgcolor=COLOR_BG, margin=dict(t=40, r=20, b=20, l=20))
st.subheader("④ 목표 달성도(누적)")
st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ⑤ 월별 기여도(워터폴)
# ─────────────────────────────────────────────────────────────
if len(df) > 0:
    base = df["매출액"].iloc[0]
    measures = ["absolute"] + ["relative"] * (len(df) - 1)
    values = [base] + df["MoM"].iloc[1:].tolist()

    fig_wf = go.Figure(
        go.Waterfall(
            x=[f"{df['월'].iloc[0]} (기준)"] + df["월"].iloc[1:].tolist(),
            measure=measures,
            y=values,
            decreasing={"marker": {"color": COLOR_TEXT}},   # 감소: 다크 그레이
            increasing={"marker": {"color": COLOR_PRIMARY}},# 증가: 블루
            totals={"marker": {"color": COLOR_ACCENT}},     # 합계: 라이트 블루
            connector={"line": {"color": COLOR_ACCENT}},
        )
    )
    fig_wf.update_layout(**common_layout, yaxis=dict(title="원", gridcolor=COLOR_GRID))
else:
    fig_wf = go.Figure()

st.subheader("⑤ 월별 기여도(워터폴)")
st.plotly_chart(fig_wf, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 원본 데이터 보기
# ─────────────────────────────────────────────────────────────
with st.expander("원본 데이터 보기"):
    st.dataframe(df[["월", "매출액", "전년동월", "증감률", "MA3", "MoM", "누적매출"]])
