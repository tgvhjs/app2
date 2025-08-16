# ================================================
# app.py — 월별 매출 대시보드 (개선 통합판, 슬라이더 타입 오류 픽스)
# ================================================
# 배포 체크리스트
# - requirements.txt 권장:
#   streamlit>=1.36
#   pandas>=2.2
#   numpy>=1.26
#   plotly>=5.22
#   scipy>=1.11    # (선택) 상관 p-value 계산 시 필요. 미설치 시 p-value는 생략됩니다.
#   kaleido>=0.2.1 # (선택) 차트 PNG 저장 기능 추가 시 필요
# - 기본 CSV 파일을 앱과 같은 폴더에 배치: "파트5_월별_매출.csv"
# - 공개 저장소라면 실제 데이터 대신 더미 CSV 사용 권장

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional, Dict

st.set_page_config(page_title="월별 매출 대시보드", layout="wide")

# 🎨 팔레트 (요청 반영, 글자색 유지)
COLOR_1 = "#F4F6F5"   # 배경
COLOR_2 = "#57C2F3"   # ACCENT
COLOR_3 = "#033BF5"   # PRIMARY
COLOR_4 = "#0564FB"
COLOR_5 = "#5B8FF9"
COLOR_TEXT = "#2E2F2F"            # 글자색 (유지)
COLOR_BG   = COLOR_1
COLOR_GRID = "rgba(91,143,249,0.20)"  # COLOR_5 기반 투명도

COMMON_LAYOUT = dict(
    height=360,
    margin=dict(t=20, r=20, b=40, l=50),
    paper_bgcolor=COLOR_BG,
    plot_bgcolor=COLOR_BG,
    font=dict(color=COLOR_TEXT),
    xaxis=dict(title="월", tickangle=-30, gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
)

DEFAULT_FILE = "파트5_월별_매출.csv"
REQUIRED_COLS = ["월", "매출액", "전년동월", "증감률"]

# ─────────────────────────────────────────────────────────────
# 유틸 함수 (포맷/검증/가공)
# ─────────────────────────────────────────────────────────────
def format_krw_short(n: Optional[float]) -> str:
    """한국 관용 단위 변환 (만/억/조). NaN/None 안전."""
    try:
        if n is None or pd.isna(n):
            return "-"
        n = float(n)
        abs_n = abs(n)
        sign = "-" if n < 0 else ""
        if abs_n >= 1_0000_0000_0000:  # 조
            return f"{sign}{abs_n/1_0000_0000_0000:.2f}조"
        if abs_n >= 1_0000_0000:       # 억
            return f"{sign}{abs_n/1_0000_0000:.2f}억"
        if abs_n >= 1_0000:            # 만
            return f"{sign}{abs_n/1_0000:.2f}만"
        return f"{int(abs_n):,}원" if sign == "" else f"-{int(abs_n):,}원"
    except Exception:
        return "-"

def format_percent(x: Optional[float], digits: int = 1) -> str:
    try:
        if x is None or pd.isna(x):
            return "-"
        return f"{x:.{digits}f}%"
    except Exception:
        return "-"

def parse_month_to_period(s):
    try:
        return pd.Period(str(s).strip(), freq="M")
    except Exception:
        return pd.NaT

def moving_avg(series: pd.Series, k: int = 3) -> pd.Series:
    return series.rolling(window=k, min_periods=k).mean()

def read_csv_safely(file_like_or_path) -> pd.DataFrame:
    """CSV 인코딩 이슈 대비: utf-8-sig → cp949 → 기본."""
    tried = []
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(file_like_or_path, encoding=enc)
        except Exception as e:
            tried.append((enc, str(e)))
    try:
        return pd.read_csv(file_like_or_path)
    except Exception as e:
        errors = "\n".join([f"- {enc}: {msg}" for enc, msg in tried])
        raise RuntimeError(f"CSV 읽기 실패:\n{errors}\n- default: {e}")

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """헤더 정규화: 공백/특수문자 제거 및 유사명 매핑."""
    mapping = {}
    for c in df.columns:
        norm = (
            str(c)
            .strip()
            .replace(" ", "")
            .replace("\u00A0", "")
            .replace("\t", "")
            .replace("(", "")
            .replace(")", "")
            .replace("_", "")
            .replace("-", "")
        )
        # 유사명 매핑
        if norm.lower() in ["month", "yyyymm", "date", "월"]:
            norm = "월"
        elif norm in ["전년동월", "전년도동월", "작년동월", "전년같은월", "전년월"]:
            norm = "전년동월"
        elif norm in ["매출", "매출액", "총매출", "판매액"]:
            norm = "매출액"
        elif norm in ["증감률", "전년동월대비", "전년대비", "yoy", "yoy증감률"]:
            norm = "증감률"
        mapping[c] = norm
    return df.rename(columns=mapping)

def validate_schema(df: pd.DataFrame):
    """필수 컬럼 존재/결측률 경고 메시지 표시."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"다음 컬럼이 없습니다: {missing}\n헤더를 확인하거나 업로드 파일을 점검하세요.")
        st.stop()
    # 결측률 요약
    lines = []
    for c in REQUIRED_COLS:
        na_ratio = df[c].isna().mean()
        if na_ratio > 0:
            lines.append(f"- '{c}' 결측률: {na_ratio*100:.1f}% (총 {df[c].isna().sum()}건)")
    if lines:
        st.warning("데이터 품질 경고:\n" + "\n".join(lines))

def detect_outliers(series: pd.Series, z: float = 3.0) -> pd.Series:
    """평균±z*표준편차 기준 이상치 bool mask."""
    s = series.dropna()
    if len(s) < 3:
        return pd.Series([False]*len(series), index=series.index)
    m = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return pd.Series([False]*len(series), index=series.index)
    mask = (series - m).abs() > (z * sd)
    mask = mask.fillna(False)
    return mask

def compute_correlations(x: pd.Series, y: pd.Series) -> Dict[str, Optional[float]]:
    """피어슨/스피어만 상관 및 p-value (scipy 있으면 p-value 계산)."""
    dfc = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(dfc) < 3:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    try:
        from scipy import stats
        pr = stats.pearsonr(dfc['x'], dfc['y'])
        sr = stats.spearmanr(dfc['x'], dfc['y'])
        return {
            "pearson_r": float(getattr(pr, "statistic", pr[0])),
            "pearson_p": float(getattr(pr, "pvalue", pr[1])),
            "spearman_r": float(getattr(sr, "statistic", sr.correlation)),
            "spearman_p": float(sr.pvalue),
        }
    except Exception:
        pr = np.corrcoef(dfc['x'], dfc['y'])[0,1]
        rx = dfc['x'].rank(method='average')
        ry = dfc['y'].rank(method='average')
        sr = np.corrcoef(rx, ry)[0,1]
        return {"pearson_r": float(pr), "pearson_p": None, "spearman_r": float(sr), "spearman_p": None}

# ─────────────────────────────────────────────────────────────
# 데이터 로드 / 전처리
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return read_csv_safely(path)

@st.cache_data(show_spinner=False)
def load_sample_csv(rows: int = 12) -> bytes:
    """샘플 CSV (현재 스키마로 12개월 더미)."""
    months = pd.period_range("2024-01", periods=rows, freq="M").strftime("%Y-%m")
    rng = np.random.default_rng(42)
    sales = rng.integers(80_000_000, 200_000_000, size=rows)  # 0.8~2.0억
    prev = (sales * rng.uniform(0.85, 1.1, size=rows)).astype(int)
    rate = (sales - prev) / np.where(prev==0, np.nan, prev) * 100
    df = pd.DataFrame({"월": months, "매출액": sales, "전년동월": prev, "증감률": rate})
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_headers(df_raw.copy())
    validate_schema(df)

    # 수치형 전처리
    for c in ["매출액", "전년동월"]:
        df[c] = (
            df[c].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip()
            .replace({"": np.nan, "-": np.nan})
            .astype(float)
        )
    df["증감률"] = (
        df["증감률"].astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "-": np.nan})
        .astype(float)
    )

    # 월 정렬 및 파생 지표
    df["월_period"] = df["월"].apply(parse_month_to_period)
    df["월_dt"] = df["월_period"].dt.to_timestamp()
    df = df.sort_values("월_period").reset_index(drop=True)

    # 파생 (전체 기준, 이후 필터 후 재계산)
    df["MA3"] = moving_avg(df["매출액"], 3)
    df["MoM"] = df["매출액"].diff()
    df["누적매출"] = df["매출액"].cumsum()
    return df

def apply_filters(df: pd.DataFrame, date_range: Tuple[pd.Timestamp, pd.Timestamp], cat_col: Optional[str], cat_values: Optional[list]) -> pd.DataFrame:
    # pandas.Timestamp로 통일해서 비교 (Slider는 datetime 반환)
    msk = (
        df["월_dt"] >= pd.Timestamp(date_range[0])
    ) & (
        df["월_dt"] <= pd.Timestamp(date_range[1])
    )
    dff = df.loc[msk].copy()
    if cat_col and cat_values:
        dff = dff[dff[cat_col].isin(cat_values)].copy()
    # 필터 후 파생 재계산
    dff["MA3"] = moving_avg(dff["매출액"], 3)
    dff["MoM"] = dff["매출액"].diff()
    dff["누적매출"] = dff["매출액"].cumsum()
    return dff

# ─────────────────────────────────────────────────────────────
# KPI & 예상치
# ─────────────────────────────────────────────────────────────
def linear_year_end_forecast(ytd: float, n_months: int) -> Optional[float]:
    if n_months <= 0:
        return None
    avg = ytd / n_months
    return ytd + avg * (12 - n_months)

def build_kpis(df: pd.DataFrame, target: Optional[float]):
    if df.empty:
        st.info("선택된 기간에 데이터가 없습니다.")
        return

    last_idx = df.index[-1]
    cur_month = df.loc[last_idx, "월"]
    cur_sales = df.loc[last_idx, "매출액"]
    prev_sales = df.loc[last_idx-1, "매출액"] if len(df) >= 2 else np.nan
    yoy_base = df.loc[last_idx, "전년동월"]

    ytd_sales = df["누적매출"].iloc[-1]
    ytd_prev = df["전년동월"].sum(skipna=True)
    yoy_ytd = ((ytd_sales - ytd_prev) / ytd_prev * 100) if ytd_prev else np.nan

    delta_mom = cur_sales - prev_sales if pd.notna(prev_sales) else np.nan
    delta_mom_pct = (cur_sales / prev_sales - 1) * 100 if pd.notna(prev_sales) and prev_sales != 0 else np.nan
    delta_yoy_pct = ((cur_sales - yoy_base) / yoy_base * 100) if pd.notna(yoy_base) and yoy_base != 0 else np.nan

    forecast = linear_year_end_forecast(ytd_sales, df.shape[0])
    achieve_now = (ytd_sales / target * 100) if target else np.nan
    achieve_forecast = (forecast / target * 100) if (target and forecast) else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label=f"이번 달 매출 ({cur_month})",
            value=format_krw_short(cur_sales),
            delta=f"{format_krw_short(delta_mom)} ({format_percent(delta_mom_pct)})" if pd.notna(delta_mom_pct) else "-"
        )
    with c2:
        st.metric(
            label="이번 달 전년동월 대비",
            value=format_percent(delta_yoy_pct) if pd.notna(delta_yoy_pct) else "-",
            delta=None
        )
    with c3:
        st.metric(
            label="YTD 누적 매출",
            value=format_krw_short(ytd_sales),
            delta=f"전년동월 합 대비 {format_percent(yoy_ytd)}" if pd.notna(yoy_ytd) else "-"
        )
    with c4:
        if target and target > 0:
            sub = f"현재 {format_percent(achieve_now)} / 연말 예상 {format_percent(achieve_forecast)}"
            st.metric("목표 달성도(예상)", value=f"{format_percent(achieve_forecast)}", delta=sub)
        else:
            st.metric("목표 달성도(예상)", value="목표 입력 필요", delta=None)

# ─────────────────────────────────────────────────────────────
# 차트 빌더
# ─────────────────────────────────────────────────────────────
def build_fig_trend(df: pd.DataFrame, show_outliers: bool, show_target_line: bool, target: Optional[float]) -> go.Figure:
    fig = go.Figure()
    hover = [
        f"{m}<br>매출액: {format_krw_short(a)}<br>3M MA: {format_krw_short(ma)}<br>MoM: {format_krw_short(mom)}"
        for m, a, ma, mom in zip(df["월"], df["매출액"], df["MA3"], df["MoM"])
    ]
    fig.add_trace(
        go.Scatter(
            x=df["월"], y=df["매출액"],
            mode="lines+markers", name="매출액",
            line=dict(width=3, color=COLOR_3), marker=dict(color=COLOR_3),
            hovertemplate="%{text}<extra></extra>", text=hover
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["월"], y=df["MA3"],
            mode="lines", name="3M 이동평균",
            line=dict(dash="dash", color=COLOR_2),
            hovertemplate="%{x}<br>3M MA: %{y:,}<extra></extra>"
        )
    )
    if show_target_line and target and target > 0:
        monthly_target = target / 12.0
        fig.add_hline(y=monthly_target, line=dict(color=COLOR_5, width=2, dash="dot"),
                      annotation_text="월평균 목표", annotation_position="top left")

    if show_outliers:
        mask = detect_outliers(df["매출액"])
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "월"], y=df.loc[mask, "매출액"],
                    mode="markers+text", name="이상치(3σ)",
                    marker=dict(size=12, color="rgba(255,0,0,0.85)", line=dict(width=1, color=COLOR_TEXT)),
                    text=[format_krw_short(v) for v in df.loc[mask, "매출액"]],
                    textposition="top center",
                    hovertemplate="%{x}<br><b>이상치</b> 매출액: %{y:,}<extra></extra>"
                )
            )
    fig.update_layout(**COMMON_LAYOUT, yaxis=dict(title="원", gridcolor=COLOR_GRID))
    return fig

def build_fig_yoy(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    hover_a = [f"{m}<br>매출액: {format_krw_short(a)}" for m, a in zip(df["월"], df["매출액"])]
    hover_b = [f"{m}<br>전년동월: {format_krw_short(a)}" for m, a in zip(df["월"], df["전년동월"])]

    fig.add_trace(
        go.Scatter(
            x=df["월"], y=df["매출액"],
            name="매출액", mode="lines+markers",
            line=dict(width=3, color=COLOR_3), marker=dict(color=COLOR_3),
            hovertemplate="%{text}<extra></extra>", text=hover_a
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df["월"], y=df["전년동월"],
            name="전년동월", mode="lines+markers",
            line=dict(dash="dot", color=COLOR_2), marker=dict(color=COLOR_2),
            hovertemplate="%{text}<extra></extra>", text=hover_b
        ),
        secondary_y=True
    )
    fig.update_yaxes(title_text="매출액(원)", secondary_y=False, gridcolor=COLOR_GRID)
    fig.update_yaxes(title_text="전년동월(원)", secondary_y=True, showgrid=False)
    fig.update_layout(**COMMON_LAYOUT)

    stats = compute_correlations(df["매출액"], df["전년동월"])
    f3 = lambda v: "N/A" if v is None else f"{v:.3f}"
    caption = (
        f"피어슨 r={f3(stats['pearson_r'])}"
        + (f", p={f3(stats['pearson_p'])}" if stats["pearson_p"] is not None else ", p=N/A")
        + f" | 스피어만 r={f3(stats['spearman_r'])}"
        + (f", p={f3(stats['spearman_p'])}" if stats["spearman_p"] is not None else ", p=N/A")
    )
    return fig, caption

def build_fig_rate(df: pd.DataFrame) -> go.Figure:
    bar_colors = np.where(df["증감률"] >= 0, COLOR_3, COLOR_TEXT)
    hover = [f"{m}<br>증감률: {format_percent(r)}" for m, r in zip(df["월"], df["증감률"])]
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["월"], y=df["증감률"],
                marker=dict(color=bar_colors),
                hovertemplate="%{text}<extra></extra>", text=hover,
                name="증감률(%)",
            )
        ]
    )
    fig.add_hline(y=0, line=dict(color=COLOR_TEXT, width=1))  # 0 기준선 강조
    fig.update_layout(**COMMON_LAYOUT, yaxis=dict(title="증감률(%)", gridcolor=COLOR_GRID))
    return fig

def build_fig_gauge(df: pd.DataFrame, target: Optional[float]) -> go.Figure:
    cum_last = df["누적매출"].iloc[-1] if len(df) else 0.0
    pct = min(100.0, cum_last / target * 100.0) if (target and target > 0) else None
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct if pct is not None else 0,
            number={"suffix": "%", "font": {"color": COLOR_TEXT}},
            title={
                "text": f"누적/목표 ({format_krw_short(cum_last)} / {format_krw_short(target)})" if target else "목표 입력 필요",
                "font": {"color": COLOR_TEXT}
            },
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLOR_TEXT},
                "bar": {"color": COLOR_3},
                "bgcolor": COLOR_BG,
                "bordercolor": COLOR_GRID,
                "steps": [
                    {"range": [0, 50], "color": "rgba(91,143,249,0.08)"},
                    {"range": [50, 80], "color": "rgba(91,143,249,0.16)"},
                    {"range": [80, 100], "color": "rgba(91,143,249,0.24)"},
                ],
            },
        )
    )
    fig.update_layout(height=360, paper_bgcolor=COLOR_BG, margin=dict(t=40, r=20, b=20, l=20))
    return fig

def build_fig_waterfall(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    base = df["매출액"].iloc[0]
    measures = ["absolute"] + ["relative"] * (len(df) - 1)
    values = [base] + df["MoM"].iloc[1:].tolist()
    fig = go.Figure(
        go.Waterfall(
            x=[f"{df['월'].iloc[0]} (기준)"] + df["월"].iloc[1:].tolist(),
            measure=measures,
            y=values,
            decreasing={"marker": {"color": COLOR_TEXT}},     # 감소: 다크 그레이
            increasing={"marker": {"color": COLOR_3}},        # 증가: 블루
            totals={"marker": {"color": COLOR_2}},            # 합계: 라이트 블루
            connector={"line": {"color": COLOR_2}},
        )
    )
    fig.update_layout(**COMMON_LAYOUT, yaxis=dict(title="원", gridcolor=COLOR_GRID))
    return fig

# ─────────────────────────────────────────────────────────────
# 사이드바 (정보 구조 & 내비)
# ─────────────────────────────────────────────────────────────
st.sidebar.title("설정")

# 데이터 섹션
st.sidebar.subheader("데이터")
uploaded = st.sidebar.file_uploader("CSV 업로드 (월, 매출액, 전년동월, 증감률)", type=["csv"])
st.sidebar.download_button(
    label="샘플 CSV 저장 (UTF-8-SIG)",
    data=load_sample_csv(),
    file_name="sample_월별매출.csv",
    mime="text/csv",
    help="현재 스키마(월, 매출액, 전년동월, 증감률)로 12개월 더미 데이터"
)

st.sidebar.markdown("---")

# 지표 섹션
st.sidebar.subheader("지표")
target = st.sidebar.number_input("연간 매출 목표(원)", min_value=0, value=0, step=1, help="게이지/목표선에서 사용됩니다.")

st.sidebar.markdown("---")

# 시각화 섹션
st.sidebar.subheader("시각화 옵션")
opt_outliers = st.sidebar.checkbox("이상치(3σ) 마커 표시", value=True)
opt_targetline = st.sidebar.checkbox("추세 그래프에 월평균 목표선 표시", value=False)

st.sidebar.markdown("---")

# 도움말 섹션
st.sidebar.subheader("도움말")
st.sidebar.caption(
    "• 헤더는 반드시 `월, 매출액, 전년동월, 증감률`\n"
    "• 인코딩은 UTF-8-SIG 권장\n"
    "• 공개 저장소에는 더미 CSV 사용 권장"
)

# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
if uploaded is not None:
    df_raw = read_csv_safely(uploaded)
else:
    try:
        df_raw = load_data_from_path(DEFAULT_FILE)
        st.sidebar.info(f"기본 데이터({DEFAULT_FILE})가 로드되었습니다.")
    except Exception as e:
        st.title("📊 월별 매출 대시보드")
        st.info("좌측에서 CSV 파일을 업로드하거나, 샘플 CSV를 내려받아 형식을 확인하세요.")
        st.error(f"기본 데이터 로드 실패: {e}")
        st.stop()

df = preprocess(df_raw)

# 선택적 카테고리 컬럼 탐색
cat_candidates = ["카테고리", "제품군", "제품", "분류"]
cat_col = next((c for c in cat_candidates if c in df.columns), None)
cat_values = None
if cat_col:
    unique_vals = [v for v in df[cat_col].dropna().unique().tolist()]
    if unique_vals:
        cat_values = st.sidebar.multiselect(f"{cat_col} 필터", options=unique_vals, default=unique_vals)

# 기간 필터 (슬라이더 타입 오류 픽스: datetime으로 받고, 비교는 pd.Timestamp로)
min_dt_pd, max_dt_pd = df["월_dt"].min(), df["월_dt"].max()
min_dt = pd.to_datetime(min_dt_pd).to_pydatetime()
max_dt = pd.to_datetime(max_dt_pd).to_pydatetime()
date_range = st.sidebar.slider(
    "분석 기간",
    min_value=min_dt,
    max_value=max_dt,
    value=(min_dt, max_dt),
)

# 필터 적용
dff = apply_filters(df, date_range, cat_col, cat_values)

# ─────────────────────────────────────────────────────────────
# 메인 레이아웃
# ─────────────────────────────────────────────────────────────
st.title("📊 월별 매출 대시보드")

# KPI
build_kpis(dff, target if target > 0 else None)
st.divider()

# 탭: 추세 / YoY / 증감률 / 게이지 / 워터폴
tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 추세", "② 전년동월 대비", "③ 증감률(%)", "④ 목표 달성도", "⑤ 월별 기여도"])

with tab1:
    fig_trend = build_fig_trend(dff, show_outliers=opt_outliers, show_target_line=opt_targetline, target=target if target > 0 else None)
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    fig_yoy, cap = build_fig_yoy(dff)
    st.plotly_chart(fig_yoy, use_container_width=True)
    st.caption(f"상관 요약: {cap}")

with tab3:
    st.plotly_chart(build_fig_rate(dff), use_container_width=True)

with tab4:
    st.plotly_chart(build_fig_gauge(dff, target if target > 0 else None), use_container_width=True)

with tab5:
    st.plotly_chart(build_fig_waterfall(dff), use_container_width=True)

st.markdown("### 원본 데이터")
st.dataframe(dff[["월", "매출액", "전년동월", "증감률", "MA3", "MoM", "누적매출"]], use_container_width=True)

# 다운로드
csv_bytes = dff[["월", "매출액", "전년동월", "증감률"]].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("현재 필터 데이터 CSV 다운로드", data=csv_bytes, file_name="filtered_월별매출.csv", mime="text/csv")

# 푸터
st.caption("ⓒ 월별 매출 대시보드 · 버전 1.0 · 공개 저장소 사용 시 민감 데이터 주의")
