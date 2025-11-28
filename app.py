# app.py  外城约车助手 V0.3.5
# SSOT workarea + 缓存 + 未来增量(干线确定量+清关行预估车量 vs 产能扣未集包) 
# + 站点比例(固定/当天) + 路区比例分摊 + 围板箱优先估托
# + OCF/JAX/MCO 城市维度估托 + SRQ/TPA 串联建议 + MCO.HUB 提示
# + MCO.HUB 站点视图（OCF+JAX+MCO 合并）
# + 约车建议“依据说明” + 多站点联运试算

import streamlit as st
import pandas as pd
import math
from datetime import datetime, time
from typing import Dict, List

st.set_page_config(page_title="外城约车助手版本", layout="wide")
st.title("外城约车助手 V0.3.5（上传明细→未来增量→站点/路区分摊→估托→最少约车）")

# =========================
# 固定外城列表（按你们业务）
# =========================
OUTCITY_LIST = ["TPA", "WPB", "JAX", "OCF", "FTM", "SRQ", "MCO"]

# ✅ 这些站在 MIA 不按路区分拣，只按城市维度
CITY_ONLY_STATIONS = {"OCF", "JAX", "MCO", "MCO.HUB"}
MCO_HUB_GROUP = ["OCF", "JAX", "MCO"]

def std_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.upper()
         .str.replace(r"\s+", "", regex=True)
    )

# =========================
# 默认站点比例（你给的）
# =========================
DEFAULT_STATION_RATIOS = {
    "FTM": 0.073758,
    "JAX": 0.02381,
    "MCO": 0.072766,
    "MIA": 0.584295,  # center 也会有，但外城只取 OUTCITY_LIST
    "OCF": 0.024513,
    "SRQ": 0.046426,
    "TPA": 0.0475,
    "WPB": 0.12566
}

# =========================
# Google Sheet workarea（SSOT）
# =========================
WORKAREA_SHEET_CSV = (
    "https://docs.google.com/spreadsheets/d/"
    "17lYLDZR_oDl1okvzlxb_Z6coiLuxsaCC55QZiYtsZ4w"
    "/export?format=csv&gid=0"
)

@st.cache_data(show_spinner=False)
def load_workarea_master(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = df.columns.astype(str).str.strip()

    if "station3" not in df.columns:
        raise ValueError("workarea master 缺少 station3 列，请检查 Google Sheet")

    df["station3"] = df["station3"].astype(str).str.upper().str.strip()
    if "邮编" in df.columns:
        df["邮编"] = df["邮编"].astype(str).str.strip().str.zfill(5)
    return df

try:
    wa_master = load_workarea_master(WORKAREA_SHEET_CSV)
except Exception as e:
    st.error(f"❌ workarea Google Sheet 读取失败：{e}")
    st.stop()

# =========================
# Sidebar: 上传明细文件
# =========================
st.sidebar.header("每日明细上传（可随时更新）")
report_file = st.sidebar.file_uploader("上传明细报表（含集包时间）", type=["xlsx"])

if report_file is None:
    st.info("请先在左侧上传明细报表")
    st.stop()

@st.cache_data(show_spinner=False)
def load_report_excel(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip()
    if "邮编" in df.columns:
        df["邮编"] = df["邮编"].astype(str).str.strip().str.zfill(5)
    return df

report_bytes = report_file.getvalue()
report_df = load_report_excel(report_bytes)

required_cols = {"目的中心", "目的站点", "邮编", "运单号"}
missing = required_cols - set(report_df.columns)
if missing:
    st.error(f"明细表缺少必要列：{missing}")
    st.stop()

report_df["目的中心_std"] = std_text(report_df["目的中心"])
report_df["目的站点_std"] = std_text(report_df["目的站点"])
report_df["station3"] = report_df["目的站点_std"].str[:3]

# ===== 站点下拉：OCF/JAX/MCO 合并为 MCO.HUB =====
raw_stations = sorted(set(report_df["station3"]) & set(OUTCITY_LIST))
if not raw_stations:
    st.error("明细里没有识别到外城站点（前三位），请确认目的站点字段")
    st.stop()

has_hub = any(s in raw_stations for s in MCO_HUB_GROUP)
display_stations: List[str] = [s for s in raw_stations if s not in MCO_HUB_GROUP]
if has_hub:
    display_stations.append("MCO.HUB")

st.sidebar.markdown("---")
target_station3 = st.sidebar.selectbox("选择外城站点", display_stations)

snapshot_time = st.sidebar.selectbox(
    "本次快照时间（可选）",
    ["17:00", "18:00", "19:00", "20:00", "21:00", "自定义"]
)
if snapshot_time == "自定义":
    snapshot_time = st.sidebar.text_input("输入时间，如 19:30", value="19:30")

def parse_snapshot_to_time(s: str):
    try:
        return datetime.strptime(s, "%H:%M").time()
    except Exception:
        return None

# =========================
# ✅ 先算全站未集包（用于产能扣减）
# =========================
bag_time_col_all = "集包时间"
if bag_time_col_all in report_df.columns:
    unbagged_all_cnt = int(report_df[bag_time_col_all].isna().sum())
else:
    unbagged_all_cnt = 0

# =========================
# cached：计算站点内路区货量 + 已/未集包（单站）
# =========================
@st.cache_data(show_spinner=False)
def calc_route_pkg_cached(report_df: pd.DataFrame,
                          station3: str,
                          wa_master: pd.DataFrame):
    report_s = report_df[report_df["station3"].eq(station3)].copy()

    pkg_total_now = len(report_s)

    bag_time_col = "集包时间"
    if bag_time_col in report_s.columns:
        bagged_cnt   = int(report_s[bag_time_col].notna().sum())
        unbagged_cnt = int(report_s[bag_time_col].isna().sum())
    else:
        bagged_cnt = None
        unbagged_cnt = None

    wa_s = wa_master[wa_master["station3"].eq(station3)].copy()
    if wa_s.empty:
        return report_s, None, 0, 0, pkg_total_now, bagged_cnt, unbagged_cnt

    wa_cols = wa_s.columns.astype(str).tolist()
    if "分拣码" in wa_cols:
        route_col = "分拣码"
    elif "route_id" in wa_cols:
        route_col = "route_id"
    elif "快递员工作区域名称" in wa_cols:
        route_col = "快递员工作区域名称"
    elif "路区" in wa_cols:
        route_col = "路区"
    else:
        route_col = None

    if route_col is None or "邮编" not in wa_cols:
        return report_s, None, 0, 0, pkg_total_now, bagged_cnt, unbagged_cnt

    wa = (
        wa_s[["邮编", route_col]]
            .rename(columns={route_col: "route_id"})
            .drop_duplicates("邮编")
    )

    zip_counts = (
        report_s.groupby("邮编")["运单号"]
                .count()
                .reset_index(name="pkg_cnt")
    )

    zip_route = zip_counts.merge(wa, on="邮编", how="left")
    unmapped_zips = int(zip_route[zip_route["route_id"].isna()]["邮编"].nunique())

    route_pkg = (
        zip_route.dropna(subset=["route_id"])
                 .groupby("route_id")["pkg_cnt"].sum()
                 .reset_index()
    )

    active_routes = int((route_pkg["pkg_cnt"] > 0).sum() + unmapped_zips)

    return report_s, route_pkg, active_routes, unmapped_zips, pkg_total_now, bagged_cnt, unbagged_cnt

# ===== 针对 MCO.HUB 做汇总，其它站按单站算 =====
if target_station3 == "MCO.HUB":
    report_s = report_df[report_df["station3"].isin(MCO_HUB_GROUP)].copy()
    pkg_total_now = len(report_s)

    bag_time_col = "集包时间"
    if bag_time_col in report_s.columns:
        bagged_cnt   = int(report_s[bag_time_col].notna().sum())
        unbagged_cnt = int(report_s[bag_time_col].isna().sum())
    else:
        bagged_cnt = None
        unbagged_cnt = None

    route_pkg = None          # 城市维度，不做路区
    active_routes = 0
    unmapped_zips = 0
else:
    report_s, route_pkg, active_routes, unmapped_zips, pkg_total_now, bagged_cnt, unbagged_cnt = \
        calc_route_pkg_cached(report_df, target_station3, wa_master)

# =========================
# ✅ Sidebar: 未来总增量估算
# =========================
st.sidebar.markdown("---")
st.sidebar.header("后续增量估算（未来总增量）")

st.sidebar.subheader("① 后面可能要做的货（干线 + 清关行）")

use_linehaul = st.sidebar.checkbox("干线确定会来多少件", value=True)
linehaul_pkgs = 0
if use_linehaul:
    linehaul_pkgs = st.sidebar.number_input(
        "干线确定来货量（件）",
        min_value=0, value=0, step=500
    )

use_broker = st.sidebar.checkbox("清关行预估还会来多少车", value=True)
broker_trucks = 0
broker_pkgs_per_truck = 10000
broker_pkgs = 0
if use_broker:
    broker_trucks = st.sidebar.number_input(
        "清关行预计还会来几车货",
        min_value=0, value=0, step=1
    )
    broker_pkgs_per_truck = st.sidebar.number_input(
        "清关行平均每车货量（默认10000）",
        min_value=0, value=10000, step=500
    )
    broker_pkgs = int(broker_trucks * broker_pkgs_per_truck)

arrival_forecast = int(linehaul_pkgs + broker_pkgs)

st.sidebar.caption(
    f"来货预测合计 = 干线 {linehaul_pkgs:,.0f} 件 + "
    f"清关行 {broker_trucks} 车×{broker_pkgs_per_truck:,.0f}≈{broker_pkgs:,.0f} 件 "
    f"= {arrival_forecast:,.0f} 件"
)

st.sidebar.subheader("② 剩余产能（先扣全站未集包）")

cutoff_t = time(22, 0)
snap_t = parse_snapshot_to_time(snapshot_time)

if snap_t is None:
    remaining_hours_auto = 0.0
else:
    snap_dt = datetime.combine(datetime.today(), snap_t)
    cutoff_dt = datetime.combine(datetime.today(), cutoff_t)
    remaining_hours_auto = max((cutoff_dt - snap_dt).total_seconds() / 3600, 0)

override_hours = st.sidebar.checkbox("手动覆盖剩余小时（可选）", value=False)
if override_hours:
    remaining_hours = st.sidebar.number_input(
        "覆盖后的剩余小时",
        min_value=0.0, value=remaining_hours_auto, step=0.5
    )
else:
    remaining_hours = remaining_hours_auto

st.sidebar.caption(f"离22:00还剩 {remaining_hours:.1f} 小时")

sort_rate = st.sidebar.number_input(
    "分拣产能/人效（件/小时，默认12000）",
    min_value=0, value=12000, step=500
)
capacity_total = remaining_hours * sort_rate

capacity_left_for_new = max(capacity_total - unbagged_all_cnt, 0)

future_total_increase = int(min(arrival_forecast, capacity_left_for_new))

slack = capacity_left_for_new - arrival_forecast

st.sidebar.info(
    f"来货预测≈ {arrival_forecast:,.0f} 件；"
    f"剩余产能≈ {capacity_total:,.0f} 件；\n"
    f"全站未集包≈ {unbagged_all_cnt:,} 件（一定要先做）；\n"
    f"可用于新来货的产能≈ {capacity_left_for_new:,.0f} 件；\n"
    f"未来总增量=min(来货,可用产能)= {future_total_increase:,.0f} 件；\n"
    f"{'✅ 产能足够，能做完所有后续来货' if slack >= 0 else '⚠️ 产能不足，部分后续来货做不完'}"
)

# =========================
# Sidebar: 未来总增量 -> 站点比例
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("③ 未来总增量按比例分摊到站点")

ratio_mode = st.sidebar.radio(
    "站点比例来源：",
    ["固定比例（默认）", "按当天货量占比"],
    index=0
)

@st.cache_data(show_spinner=False)
def calc_today_station_ratios(report_df: pd.DataFrame,
                              station_keys: List[str]) -> Dict[str, float]:
    s = report_df["station3"].astype(str).str.upper().str.strip()
    cnt = s.value_counts()
    cnt = cnt[cnt.index.isin(station_keys)]
    total = cnt.sum()
    if total == 0:
        return DEFAULT_STATION_RATIOS.copy()
    return {k: float(cnt.get(k, 0)) / float(total) for k in station_keys}

def get_station_forecast(total_inc: int, station3: str) -> int:
    if total_inc <= 0:
        return 0
    if ratio_mode == "按当天货量占比":
        today_ratios = calc_today_station_ratios(
            report_df, list(DEFAULT_STATION_RATIOS.keys())
        )
        if station3 == "MCO.HUB":
            ratio = sum(today_ratios.get(s, 0.0) for s in MCO_HUB_GROUP)
        else:
            ratio = today_ratios.get(station3, 0.0)
    else:
        if station3 == "MCO.HUB":
            ratio = sum(DEFAULT_STATION_RATIOS.get(s, 0.0) for s in MCO_HUB_GROUP)
        else:
            ratio = DEFAULT_STATION_RATIOS.get(station3, 0.0)
    return int(round(total_inc * ratio))

forecast_in_station = get_station_forecast(future_total_increase, target_station3)
st.sidebar.caption(f"本站点未来增量 ≈ {forecast_in_station:,} 件")

# =========================
# Sidebar: 车型选择
# =========================
st.sidebar.markdown("---")
truck_mode = st.sidebar.radio(
    "车型选择",
    ["混用（先53后26）", "只用53尺", "只用26尺"],
    index=0
)
mode_map = {"混用（先53后26）": "mix", "只用53尺": "53_only", "只用26尺": "26_only"}
truck_mode_key = mode_map[truck_mode]

# =========================
# Sidebar: 容器估算规则（围板箱优先）
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("④ 容器估算规则（默认围板箱优先）")

prefer_board_only = st.sidebar.checkbox("默认按围板箱估托", value=True)
st.sidebar.caption("围板箱不足/现场明确用GL时，取消勾选再按混装比例估托")

use_container = st.sidebar.checkbox("我知道容器数量（围板箱 + gaylord）", value=False)

board_cap = st.sidebar.number_input("围板箱计划容量", min_value=150, value=250, step=10)
gay_cap   = st.sidebar.number_input("Gaylord计划容量", min_value=300, value=450, step=10)

board_cnt = gay_cnt = None
r_gay = 0.6
if use_container:
    board_cnt = st.sidebar.number_input("围板箱数量", min_value=0, value=0, step=1)
    gay_cnt   = st.sidebar.number_input("Gaylord数量", min_value=0, value=0, step=1)
else:
    if not prefer_board_only:
        r_gay = st.sidebar.slider("Gaylord占比（特殊情况才用）", 0.0, 1.0, 0.6, 0.05)
    else:
        r_gay = 0.0

# =========================
# 车数函数
# =========================
def calc_trucks_by_type(pallets_final, mode="mix", cap_53=30, cap_26=12):
    if pallets_final <= 0:
        return {"trucks_53": 0, "trucks_26": 0,
                "total_trucks": 0, "buffer_pallets": 0}

    if mode == "53_only":
        t53 = math.ceil(pallets_final / cap_53)
        return {"trucks_53": t53, "trucks_26": 0, "total_trucks": t53,
                "buffer_pallets": t53 * cap_53 - pallets_final}

    if mode == "26_only":
        t26 = math.ceil(pallets_final / cap_26)
        return {"trucks_53": 0, "trucks_26": t26, "total_trucks": t26,
                "buffer_pallets": t26 * cap_26 - pallets_final}

    t53 = pallets_final // cap_53
    rem = pallets_final - t53 * cap_53
    t26 = math.ceil(rem / cap_26) if rem > 0 else 0
    return {"trucks_53": int(t53), "trucks_26": int(t26),
            "total_trucks": int(t53 + t26),
            "buffer_pallets": int(t53 * cap_53 + t26 * cap_26 - pallets_final)}

# =========================
# 工具函数：估任意站点当前托数（给串联&hub提示用）
# =========================
def pallets_for_pkgs(pkg_r: int, board_cap=250, gay_cap=450,
                     r_gay=0.6, prefer_board_only=True):
    board_p = math.ceil(pkg_r / board_cap) / 2
    if prefer_board_only:
        return math.ceil(board_p)
    gay_p = math.ceil(pkg_r / gay_cap)
    return math.ceil((1 - r_gay) * board_p + r_gay * gay_p)

def estimate_pallets_for_station(report_df: pd.DataFrame,
                                 station3: str,
                                 wa_master: pd.DataFrame,
                                 board_cap=250,
                                 gay_cap=450,
                                 r_gay=0.6,
                                 prefer_board_only=True) -> int:
    rep_s, route_pkg_s, active_routes_s, _, pkg_total_now_s, _, _ = \
        calc_route_pkg_cached(report_df, station3, wa_master)

    if pkg_total_now_s == 0:
        return 0

    if station3 in CITY_ONLY_STATIONS or route_pkg_s is None or route_pkg_s.empty:
        board_p_total = math.ceil(pkg_total_now_s / board_cap) / 2
        if prefer_board_only:
            return math.ceil(board_p_total)
        else:
            gay_p_total = math.ceil(pkg_total_now_s / gay_cap)
            return math.ceil(r_gay * gay_p_total + (1 - r_gay) * board_p_total)

    pallets_est_s = int(route_pkg_s["pkg_cnt"].apply(
        lambda x: pallets_for_pkgs(x, board_cap, gay_cap, r_gay, prefer_board_only)
    ).sum())
    return max(pallets_est_s, active_routes_s)

def estimate_pallets_for_mcohub(report_df: pd.DataFrame,
                                wa_master: pd.DataFrame,
                                board_cap=250,
                                gay_cap=450,
                                r_gay=0.6,
                                prefer_board_only=True) -> int:
    total_pallets = 0
    for st3 in MCO_HUB_GROUP:
        total_pallets += estimate_pallets_for_station(
            report_df, st3, wa_master,
            board_cap=board_cap, gay_cap=gay_cap,
            r_gay=r_gay, prefer_board_only=prefer_board_only
        )
    return total_pallets

# =========================
# 本站点未来增量 -> 路区当天占比再分摊
# =========================
route_pkg_fc = None
if route_pkg is not None and not route_pkg.empty and pkg_total_now > 0:
    route_pkg_fc = route_pkg.copy()
    share = route_pkg_fc["pkg_cnt"] / pkg_total_now
    route_pkg_fc["future_add"] = (share * forecast_in_station).round().astype(int)
    route_pkg_fc["pkg_cnt_fc"] = route_pkg_fc["pkg_cnt"] + route_pkg_fc["future_add"]
else:
    route_pkg_fc = route_pkg

# =========================
# 托数估算（含城市维度站点分支）
# =========================
def calc_pallets_with_route(pkg_total_now, active_routes, forecast_in_station,
                            board_cnt=None, gay_cnt=None,
                            board_cap=250, gay_cap=450, r_gay=0.6,
                            route_pkg_fc: pd.DataFrame = None,
                            prefer_board_only=True):

    final_cnt = pkg_total_now + forecast_in_station

    # 1) 用户输入容器数量（最准确）
    if board_cnt is not None and gay_cnt is not None:
        pallets_est = math.ceil(board_cnt / 2) + gay_cnt
        cap_container = board_cnt * board_cap + gay_cnt * gay_cap
        pallets_final = max(pallets_est, active_routes)
        return "container_count", final_cnt, pallets_est, pallets_final, cap_container

    # 2) 有路区/城市维度信息
    if route_pkg_fc is not None and not route_pkg_fc.empty:
        if target_station3 in CITY_ONLY_STATIONS:
            # 城市维度站点：按总量估托，不套路区最小托数
            board_p_total = math.ceil(final_cnt / board_cap) / 2
            if prefer_board_only:
                pallets_est = math.ceil(board_p_total)
            else:
                gay_p_total = math.ceil(final_cnt / gay_cap)
                pallets_est = math.ceil(
                    r_gay * gay_p_total + (1 - r_gay) * board_p_total
                )
            pallets_final = pallets_est
            mode = "city_only"
            return mode, final_cnt, pallets_est, pallets_final, None
        else:
            # 正常站点：按路区估托 + 最小托数约束
            pallets_est = int(route_pkg_fc["pkg_cnt_fc"].apply(
                lambda x: pallets_for_pkgs(x, board_cap, gay_cap, r_gay, prefer_board_only)
            ).sum())
            pallets_final = max(pallets_est, active_routes)
            mode = "route_board_only" if prefer_board_only else "route_mix_est"
            return mode, final_cnt, pallets_est, pallets_final, None

    # 3) fallback：没有路区信息时，按站点总量估托
    board_p_total = math.ceil(final_cnt / board_cap) / 2
    if prefer_board_only:
        pallets_est = math.ceil(board_p_total)
        pallets_final = max(pallets_est, active_routes)
        return "ratio_board_only", final_cnt, pallets_est, pallets_final, None

    gay_p_total = math.ceil(final_cnt / gay_cap)
    pallets_est = math.ceil(r_gay * gay_p_total + (1 - r_gay) * board_p_total)
    pallets_final = max(pallets_est, active_routes)
    return "ratio_mix_est", final_cnt, pallets_est, pallets_final, None

mode, final_cnt, pallets_est, pallets_final, cap_container = calc_pallets_with_route(
    pkg_total_now, active_routes, forecast_in_station,
    board_cnt, gay_cnt, board_cap, gay_cap, r_gay,
    route_pkg_fc=route_pkg_fc,
    prefer_board_only=prefer_board_only
)

truck_plan = calc_trucks_by_type(pallets_final, mode=truck_mode_key)

# =========================
# 展示区
# =========================
st.caption(f"当前结果基于 {snapshot_time} 上传的明细快照")

c1, c2, c3, c4 = st.columns(4)
c1.metric("当前包裹总量", f"{pkg_total_now:,}")
c2.metric("已集包量", f"{bagged_cnt:,}" if bagged_cnt is not None else "N/A")
c3.metric("未集包量(本站点)", f"{unbagged_cnt:,}" if unbagged_cnt is not None else "N/A")
c4.metric("活跃路区数（最小托数）", f"{active_routes:,}")

st.caption(f"全站未集包合计（用于扣产能）：{unbagged_all_cnt:,} 件")

if unmapped_zips > 0:
    st.warning(
        f"⚠️ 有 {unmapped_zips} 个邮编未映射到路区，已按“每邮编=1托”计入最小托数。"
        "建议更新该站点 workarea master。"
    )

c5, c6, c7 = st.columns(3)
c5.metric("预计截单前总包裹(本站点)", f"{final_cnt:,}")
c6.metric("估算托数（未加最小托数）", f"{pallets_est}")
c7.metric("最少托数（加路区约束）", f"{pallets_final}")

# =========================
# 最少约车建议 + 依据说明
# =========================
st.markdown("### 最少约车建议")

st.success(f"✅ 建议最少约 **{truck_plan['total_trucks']}** 车")

# 解释约车依据
explain_lines = []

# 1) 货量构成
explain_lines.append(
    f"- 预计截单前本站点总包裹：**{final_cnt:,} 件** "
    f"（当前 {pkg_total_now:,} 件 + 本站未来增量 {forecast_in_station:,} 件）"
)

# 2) 托数逻辑
if mode == "container_count":
    explain_lines.append(
        f"- 托数按你输入的容器数量估算：围板箱 {board_cnt or 0} 个、"
        f"Gaylord {gay_cnt or 0} 个 → 估算托数 **{pallets_est} 托**；"
        f"考虑最少托数约束后为 **{pallets_final} 托**。"
    )
elif mode == "city_only":
    explain_lines.append(
        f"- 当前站点按**城市维度**估托（不看路区），"
        f"估算托数 **{pallets_est} 托**；"
        f"城市站点不套“每路区≥1托”的约束，最终托数 = **{pallets_final} 托**。"
    )
else:
    explain_lines.append(
        f"- 按路区/比例估托得到 **{pallets_est} 托**，"
        f"再叠加“活跃路区至少一托”的约束后，为 **{pallets_final} 托**。"
    )

# 3) 车型对应关系
if truck_mode_key == "53_only":
    explain_lines.append(
        f"- 车型：**只用 53 尺**，按 30 托/车 计算："
        f"{pallets_final} ÷ 30 → 需 **{truck_plan['trucks_53']} 辆 53 尺车**，"
        f"多出的容量折算为缓冲托数 **{truck_plan['buffer_pallets']} 托**。"
    )
elif truck_mode_key == "26_only":
    explain_lines.append(
        f"- 车型：**只用 26 尺**，按 12 托/车 计算："
        f"{pallets_final} ÷ 12 → 需 **{truck_plan['trucks_26']} 辆 26 尺车**，"
        f"多出的容量折算为缓冲托数 **{truck_plan['buffer_pallets']} 托**。"
    )
else:  # mix
    explain_lines.append(
        "- 车型：**混用（优先 53 尺，再用 26 尺补足）**："
        f"在总托数 {pallets_final} 托的前提下，"
        f"算法为：先尽量用 53 尺装满 → {truck_plan['trucks_53']} 辆 53 尺；"
        f"剩余托数再按 12 托/车安排 {truck_plan['trucks_26']} 辆 26 尺；"
        f"合计缓冲托数 **{truck_plan['buffer_pallets']} 托**。"
    )

st.markdown("\n".join(explain_lines))

# 原来的简要展示保留
st.write(
    f"53尺车：{truck_plan['trucks_53']} 车（30托/车）  |  "
    f"26尺车：{truck_plan['trucks_26']} 车（12托/车）"
)
st.write(f"剩余缓冲托数：{truck_plan['buffer_pallets']} 托")

if mode == "container_count":
    st.info(f"你输入的容器计划可装约 {cap_container:,} 件包裹（按计划容量）")
else:
    st.caption("未输入容器数量，按规则估算托数。")

# ===== MCO.HUB 内部城市拆分（当前货量估托）=====
if target_station3 == "MCO.HUB":
    city_rows = []
    for st3 in MCO_HUB_GROUP:
        # 先拿到这个城市当前的包裹数
        _, _, _, _, pkg_total_now_s, _, _ = calc_route_pkg_cached(
            report_df, st3, wa_master
        )
        # 再用统一逻辑估托
        pallets_city = estimate_pallets_for_station(
            report_df, st3, wa_master,
            board_cap=board_cap, gay_cap=gay_cap,
            r_gay=r_gay, prefer_board_only=prefer_board_only
        )
        city_rows.append({
            "站点": st3,
            "当前包裹数": pkg_total_now_s,
            "当前估算托数": pallets_city
        })

    if city_rows:
        st.markdown("#### MCO.HUB 内部拆分（按城市维度估托，仅基于当前已签入+未集包货量）")
        st.table(pd.DataFrame(city_rows))


# =========================
# 多站点联运试算（比如 SRQ+TPA 一起看几车）
# =========================
st.markdown("---")
st.markdown("### 多站点联运试算（实验功能）")

multi_candidates = display_stations

multi_selected = st.multiselect(
    "选择想一起发车的站点（例如同时勾选 SRQ 和 TPA）：",
    options=multi_candidates,
    default=[target_station3] if target_station3 in multi_candidates else []
)

if len(multi_selected) >= 2:
    total_pallets_multi = 0
    detail_lines = []

    def estimate_pallets_with_current_logic(st3: str) -> int:
        # 联运试算用：主站用当前页面托数，其它站按当前货量估托
        if st3 == target_station3:
            return pallets_final
        if st3 == "MCO.HUB":
            return estimate_pallets_for_mcohub(
                report_df, wa_master,
                board_cap=board_cap, gay_cap=gay_cap,
                r_gay=r_gay, prefer_board_only=prefer_board_only
            )
        return estimate_pallets_for_station(
            report_df, st3, wa_master,
            board_cap=board_cap, gay_cap=gay_cap,
            r_gay=r_gay, prefer_board_only=prefer_board_only
        )

    for st3 in multi_selected:
        pallets_st = estimate_pallets_with_current_logic(st3)
        total_pallets_multi += pallets_st
        detail_lines.append(f"- {st3}：约 {pallets_st} 托")

    trial_truck_plan = calc_trucks_by_type(total_pallets_multi, mode=truck_mode_key)

    st.info(
        "你勾选的站点联运试算结果：\n"
        + "\n".join(detail_lines)
        + f"\n\n→ 合计约 **{total_pallets_multi} 托**，"
        f"按当前车型策略预计需要 **{trial_truck_plan['total_trucks']} 车**"
        f"（53 尺 {trial_truck_plan['trucks_53']} 辆，"
        f"26 尺 {trial_truck_plan['trucks_26']} 辆，"
        f"缓冲托数约 {trial_truck_plan['buffer_pallets']} 托）。\n"
        "⚠️ 说明：联运试算基于**当前已签入+未集包货量**估托，"
        "不额外分摊你在侧边栏输入的未来来货，只用于粗略判断“几站放一起大概要几车”。"
    )
else:
    st.caption("如需查看联运效果，请至少选择两个站点。")

# ===== SRQ & TPA 串联建议（不影响主逻辑，只做提示） =====
if target_station3 in {"SRQ", "TPA"}:
    other = "TPA" if target_station3 == "SRQ" else "SRQ"
    pallets_this = pallets_final
    pallets_other = estimate_pallets_for_station(
        report_df, other, wa_master,
        board_cap=board_cap, gay_cap=gay_cap,
        r_gay=r_gay, prefer_board_only=prefer_board_only
    )
    total_pallets_st = pallets_this + pallets_other
    if 0 < total_pallets_st <= 30:
        st.info(
            f"📌 串点建议：当前 {target_station3} 约 {pallets_this} 托，"
            f"{other} 约 {pallets_other} 托，总计约 {total_pallets_st} 托，"
            f"可考虑 {target_station3}+{other} 串联一辆 53 尺车。"
        )

# ===== MCO.HUB (OCF+JAX+MCO) 第二车装载率 + WPB 串联提示 =====
if target_station3 in CITY_ONLY_STATIONS:
    pallets_mcohub = estimate_pallets_for_mcohub(
        report_df, wa_master,
        board_cap=board_cap, gay_cap=gay_cap,
        r_gay=r_gay, prefer_board_only=prefer_board_only
    )
    if pallets_mcohub > 0:
        cap_53 = 30
        trucks_full = pallets_mcohub // cap_53
        last_truck_pallets = pallets_mcohub % cap_53
        if trucks_full >= 1 and last_truck_pallets > 0:
            load_ratio_last = last_truck_pallets / cap_53
            if load_ratio_last < 0.6:
                st.warning(
                    f"📌 MCO.HUB 提示：OCF+JAX+MCO 合计约 {pallets_mcohub} 托，"
                    f"第 {trucks_full + 1} 辆 53 尺车预计仅装 {last_truck_pallets} 托"
                    f"（装载率约 {load_ratio_last:.0%}）。"
                    "可考虑：① 若 MCO.HUB 有晚班，可将部分货安排到 HUB 晚班/次日早班；"
                    "② 若当晚 WPB 也有一辆车装载率较低，可考虑 MCO.HUB 与 WPB 串联发一车。"
                )

# =========================
# 路区货量 + 围板箱换算表（MCO.HUB 没路区就不会展示）
# =========================
if route_pkg_fc is not None and not route_pkg_fc.empty:
    show_df = route_pkg_fc.copy()
    show_df["围板箱估算(个)"] = show_df["pkg_cnt_fc"].apply(
        lambda x: int(math.ceil(x / board_cap))
    )
    show_df["围板箱托数(托)"] = show_df["围板箱估算(个)"].apply(
        lambda x: int(math.ceil(x / 2))
    )
    show_df["路区估算托数(托)"] = show_df["pkg_cnt_fc"].apply(
        lambda x: pallets_for_pkgs(x, board_cap, gay_cap, r_gay, prefer_board_only)
    )

    st.markdown("---")
    st.markdown("### 路区货量分布（含未来增量按路区比例分摊）")
    st.dataframe(show_df.sort_values("pkg_cnt_fc", ascending=False), use_container_width=True)

    st.download_button(
        "下载路区分布+围板箱换算表",
        data=show_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{target_station3}_route_pkg_with_boardbox.csv",
        mime="text/csv"
    )

# =========================
# 缓存控制
# =========================
st.sidebar.markdown("---")
if st.sidebar.button("🔄 清空缓存并重算（比如workarea更新后）"):
    st.cache_data.clear()
    st.rerun()
