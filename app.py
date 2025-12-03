# app.py 外城约车助手 V0.3.7 (最终版：主力 53尺 + 尾部 26尺策略 + 产能班次预设)
# SSOT workarea + 缓存 + 未来增量(干线确定量+清关行预估车量 vs 产能扣未集包)
# + 站点比例(固定/当天) + 路区比例分摊 + 围板箱优先估托
# + OCF/JAX/MCO 城市维度估托 + SRQ/TPA 串联建议 + MCO.HUB 提示
# + MCO.HUB 站点视图（OCF+JAX+MCO 合并）
# + 最小托数逻辑：每个路区至少一个围板箱
# + 新增 V0.3.7:
#   1. ✅ 约车逻辑 V4：混用模式采用“主力 53 尺 + 尾部按容器数量决定 53/26 尺”策略。
#   2. ✅ 产能设置：提供中班、大班、小班的产能预设，并支持自定义。
#   3. ✅ 线路提醒：MIA→SRQ→TPA、MIA→WPB→MCO 串点提示（不改主逻辑，仅做提醒）。

import streamlit as st
import pandas as pd
import math
from datetime import datetime, time
from typing import Dict, List, Tuple

st.set_page_config(page_title="外城约车助手版本", layout="wide")
st.title("外城约车助手 V0.3.7（多站点串联估算 + 最终约车策略）")

# =========================
# 固定外城列表（按你们业务）
# =========================
OUTCITY_LIST = ["TPA", "WPB", "JAX", "OCF", "FTM", "SRQ", "MCO", "MIA"]  # ✅ 加上 MIA

# 这些站在 MIA 不按路区分拣，只按城市维度
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

# ===== 站点下拉：多选支持串联（OCF/JAX/MCO 合并为 MCO.HUB） =====
raw_stations = sorted(set(report_df["station3"]) & set(OUTCITY_LIST))
if not raw_stations:
    st.error("明细里没有识别到外城站点（前三位），请确认目的站点字段")
    st.stop()

has_hub = any(s in raw_stations for s in MCO_HUB_GROUP)
display_stations: List[str] = [s for s in raw_stations if s not in MCO_HUB_GROUP]
if has_hub:
    display_stations.append("MCO.HUB")

st.sidebar.markdown("---")
# 🔧 多选，允许用户自定义串点
selected_station3_list: List[str] = st.sidebar.multiselect(
    "✅ 选择本次约车站点（可多选，Ctrl/Cmd 多选）",
    display_stations,
    default=[display_stations[0]] if display_stations else None  # 默认选中第一个
)

if not selected_station3_list:
    st.warning("请至少选择一个站点进行估算")
    st.stop()

# 将 MCO.HUB 展开为实际的站点列表，以便后续数据筛选
actual_station3_list: List[str] = []
for s in selected_station3_list:
    if s == "MCO.HUB":
        actual_station3_list.extend(MCO_HUB_GROUP)
    else:
        actual_station3_list.append(s)
actual_station3_list = list(set(actual_station3_list))

# 用于展示的名称（多站点用 “ / ” 拼接）
target_station3 = " / ".join(selected_station3_list)

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
# 先算全站未集包（用于产能扣减）
# =========================
bag_time_col_all = "集包时间"
if bag_time_col_all in report_df.columns:
    unbagged_all_cnt = int(report_df[bag_time_col_all].isna().sum())
else:
    unbagged_all_cnt = 0

# =========================
# cached：计算单站路区货量 + 已/未集包 (保留单站计算，以供单站或 MCO.HUB 拆分使用)
# =========================
@st.cache_data(show_spinner=False)
def calc_route_pkg_cached(
    report_df: pd.DataFrame, station3: str, wa_master: pd.DataFrame
):
    report_s = report_df[report_df["station3"].eq(station3)].copy()
    pkg_total_now = len(report_s)

    bag_time_col = "集包时间"
    if bag_time_col in report_s.columns:
        bagged_cnt = int(report_s[bag_time_col].notna().sum())
        unbagged_cnt = int(report_s[bag_time_col].isna().sum())
    else:
        bagged_cnt = None
        unbagged_cnt = None

    wa_s = wa_master[wa_master["station3"].eq(station3)].copy()
    if wa_s.empty:
        return report_s, None, 0, 0, pkg_total_now, bagged_cnt, unbagged_cnt

    wa_cols = wa_s.columns.astype(str).tolist()
    route_col = next(
        (c for c in ["分拣码", "route_id", "快递员工作区域名称", "路区"] if c in wa_cols), None
    )

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
    return (
        report_s,
        route_pkg,
        active_routes,
        unmapped_zips,
        pkg_total_now,
        bagged_cnt,
        unbagged_cnt,
    )


# 计算选定站点集合的货量（串点时按集合计算）
@st.cache_data(show_spinner=False)
def calc_multiple_stations(
    report_df: pd.DataFrame, station_list: List[str]
) -> Tuple[pd.DataFrame, int, int | None, int | None, int, int]:
    # 筛选报告数据
    report_s_combined = report_df[report_df["station3"].isin(station_list)].copy()
    pkg_total_now = len(report_s_combined)

    bag_time_col = "集包时间"
    if bag_time_col in report_s_combined.columns:
        bagged_cnt = int(report_s_combined[bag_time_col].notna().sum())
        unbagged_cnt = int(report_s_combined[bag_time_col].isna().sum())
    else:
        bagged_cnt = None
        unbagged_cnt = None

    # 串点计算时不进行路区聚合
    active_routes = 0
    unmapped_zips = 0

    return (
        report_s_combined,
        pkg_total_now,
        bagged_cnt,
        unbagged_cnt,
        active_routes,
        unmapped_zips,
    )


# 仅单站且非 Hub 时才计算路区
is_single_station = (
    len(selected_station3_list) == 1
    and selected_station3_list[0] not in CITY_ONLY_STATIONS
)

if is_single_station:
    single_station_key = actual_station3_list[0]
    (
        report_s,
        route_pkg,
        active_routes,
        unmapped_zips,
        pkg_total_now,
        bagged_cnt,
        unbagged_cnt,
    ) = calc_route_pkg_cached(report_df, single_station_key, wa_master)
else:
    # 串点计算或 MCO.HUB
    (
        report_s,
        pkg_total_now,
        bagged_cnt,
        unbagged_cnt,
        active_routes,
        unmapped_zips,
    ) = calc_multiple_stations(report_df, actual_station3_list)
    route_pkg = None  # 串点时不展示路区

# =========================
# Sidebar: 未来总增量估算
# =========================
st.sidebar.markdown("---")
st.sidebar.header("后续增量估算（未来总增量）")

# ✅ 用 session_state 记住干线/清关行输入，避免每次勾选站点都要重输
st.sidebar.subheader("① 后面可能要做的货（干线 + 清关行）")

for key, default in [
    ("use_linehaul", True),
    ("linehaul_pkgs", 0),
    ("use_broker", True),
    ("broker_trucks", 0),
    ("broker_pkgs_per_truck", 10000),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# 干线：是否勾选 + 车货量
use_linehaul = st.sidebar.checkbox("干线确定会来多少件", key="use_linehaul")
linehaul_pkgs = 0
if use_linehaul:
    linehaul_pkgs = st.sidebar.number_input(
        "干线确定来货量（件）",
        min_value=0,
        step=500,
        key="linehaul_pkgs",
    )

# 清关行：车数 + 每车件数
use_broker = st.sidebar.checkbox("清关行预估还会来多少车", key="use_broker")
broker_trucks = 0
broker_pkgs_per_truck = 10000
broker_pkgs = 0
if use_broker:
    broker_trucks = st.sidebar.number_input(
        "清关行预计还会来几车货",
        min_value=0,
        step=1,
        key="broker_trucks",
    )
    broker_pkgs_per_truck = st.sidebar.number_input(
        "清关行平均每车货量（默认10000）",
        min_value=0,
        step=500,
        key="broker_pkgs_per_truck",
    )
    broker_pkgs = int(broker_trucks * broker_pkgs_per_truck)

arrival_forecast = int(linehaul_pkgs + broker_pkgs)

# ----------------------------------------------------
# 🔧 分拣产能预设
# ----------------------------------------------------
st.sidebar.subheader("② 剩余产能（先扣全站未集包）")

# 剩余时间计算
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
        "覆盖后的剩余小时", min_value=0.0, value=remaining_hours_auto, step=0.5
    )
else:
    remaining_hours = remaining_hours_auto
st.sidebar.caption(f"离22:00还剩 {remaining_hours:.1f} 小时")

# 分拣人效预设
st.sidebar.markdown("---")
st.sidebar.subheader("分拣人效/产能设置")
shift_options = {
    "中班（默认）：12,000 件/小时": 12000,
    "大班：16,000 件/小时": 16000,
    "小班：8,000 件/小时": 8000,
    "自定义": "custom",
}

shift_selection = st.sidebar.selectbox(
    "选择班次或人效预设：",
    options=list(shift_options.keys()),
    index=0,  # 默认选中中班
)

sort_rate = 0
if shift_options[shift_selection] == "custom":
    sort_rate = st.sidebar.number_input(
        "自定义分拣产能（件/小时）", min_value=0, value=12000, step=500
    )
else:
    sort_rate = shift_options[shift_selection]

if shift_options[shift_selection] != "custom":
    st.sidebar.caption(f"当前人效：{sort_rate:,} 件/小时")

# 产能计算
capacity_total = remaining_hours * sort_rate
capacity_left_for_new = max(capacity_total - unbagged_all_cnt, 0)
future_total_increase = int(min(arrival_forecast, capacity_left_for_new))
slack = capacity_left_for_new - arrival_forecast

st.sidebar.caption(
    f"来货预测合计 = 干线 {linehaul_pkgs:,.0f} 件 + "
    f"清关行 {broker_trucks} 车×{broker_pkgs_per_truck:,.0f}≈{broker_pkgs:,.0f} 件 "
    f"= {arrival_forecast:,.0f} 件"
)
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
st.sidebar.subheader("③ 未来总增量按比例分摊到站点集合")


@st.cache_data(show_spinner=False)
def calc_today_station_ratios(
    report_df: pd.DataFrame, station_keys: List[str]
) -> Dict[str, float]:
    s = report_df["station3"].astype(str).str.upper().str.strip()
    cnt = s.value_counts()
    cnt = cnt[cnt.index.isin(station_keys)]
    total = cnt.sum()
    if total == 0:
        return DEFAULT_STATION_RATIOS.copy()
    return {k: float(cnt.get(k, 0)) / float(total) for k in station_keys}


ratio_mode = st.sidebar.radio(
    "站点集合比例来源：", ["固定比例（默认）", "按当天货量占比"], index=0
)


# 调整后的分摊函数，适用于站点集合
def get_station_group_forecast(total_inc: int, station_list: List[str]) -> int:
    if total_inc <= 0:
        return 0

    if ratio_mode == "按当天货量占比":
        ratios_dict = calc_today_station_ratios(
            report_df, list(DEFAULT_STATION_RATIOS.keys())
        )
    else:
        ratios_dict = DEFAULT_STATION_RATIOS

    # 计算所选站点集合的总比例（按实际站点）
    base_station_list = [s for s in station_list if s in DEFAULT_STATION_RATIOS]
    ratio = sum(ratios_dict.get(s, 0.0) for s in base_station_list)

    return int(round(total_inc * ratio))


forecast_in_station_group = get_station_group_forecast(
    future_total_increase, actual_station3_list
)
st.sidebar.caption(
    f"本次约车站点集合未来增量 ≈ **{forecast_in_station_group:,}** 件"
)
forecast_in_station = forecast_in_station_group

# =========================
# Sidebar: 车型选择
# =========================
st.sidebar.markdown("---")
truck_mode = st.sidebar.radio(
    "车型选择", ["混用（主力 53尺 + 尾部 26尺）", "只用53尺", "只用26尺"], index=0
)
mode_map = {
    "混用（主力 53尺 + 尾部 26尺）": "mix",
    "只用53尺": "53_only",
    "只用26尺": "26_only",
}
truck_mode_key = mode_map[truck_mode]

# =========================
# Sidebar: 容器估算规则
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("④ 容器估算规则（默认围板箱优先）")
prefer_board_only = st.sidebar.checkbox(
    "默认按围板箱估托（未知容器数量时）", value=True
)
st.sidebar.caption(
    "⚠️ 若已知实际围板箱/Gaylord 数，请勾选下方“我知道容器数量”并直接填写。"
)

use_container = st.sidebar.checkbox("我知道当前容器数量（围板箱 + Gaylord）", value=False)
board_cap = st.sidebar.number_input(
    "围板箱计划容量（件/箱）", min_value=150, value=250, step=10
)
gay_cap = st.sidebar.number_input(
    "Gaylord 计划容量（件/个）", min_value=300, value=450, step=10
)

board_cnt = gay_cnt = None
future_container_mode = None
r_gay = 0.6  # 仅用于估算 fallback

if use_container:
    board_cnt = st.sidebar.number_input("当前围板箱数量", min_value=0, value=0, step=1)
    gay_cnt = st.sidebar.number_input("当前 Gaylord 数量", min_value=0, value=0, step=1)

    future_choice = st.sidebar.radio(
        "未来新增货物预计主要使用的容器类型",
        ["默认按围板箱为主", "未来新增全部用 Gaylord"],
        index=0,
    )
    future_container_mode = (
        "gay" if future_choice == "未来新增全部用 Gaylord" else "board"
    )
else:
    if not prefer_board_only:
        r_gay = st.sidebar.slider(
            "Gaylord 占比（仅用于估算容器数量）", 0.0, 1.0, 0.6, 0.05
        )
    else:
        r_gay = 0.0


# =========================
# 车型计算函数（V0.3.7：主力 53 尺 + 尾部 26 尺策略）
# =========================
def calc_trucks_by_type(
    pallets_final: int,
    mode: str = "mix",
    cap_53_pallets: int = 30,
    cap_26_pallets: int = 12,
    cap_26_containers: int = 12,  # 26 尺车按容器数算
    est_board_boxes: int | None = None,
    est_gaylords: int | None = None,
):
    if pallets_final <= 0:
        return {
            "trucks_53": 0,
            "trucks_26": 0,
            "total_trucks": 0,
            "buffer_pallets": 0,
            "suggestion_reason": "无货物",
        }

    # 计算总容器数（用于 mix 模式的逻辑判断）
    total_containers = None
    if est_board_boxes is not None and est_gaylords is not None:
        total_containers = est_board_boxes + est_gaylords

    # 53 尺车最大容器容量（近似）：30 托 * 2 箱/托 = 60 容器
    cap_53_containers = cap_53_pallets * 2

    # --- 1. 只用 26 尺模式 (26_only) ---
    if mode == "26_only":
        if total_containers is not None and total_containers > 0:
            t26 = math.ceil(total_containers / cap_26_containers)
            buffer_containers = t26 * cap_26_containers - total_containers
            buffer_pallets_est = math.ceil(buffer_containers / 2)
            reason = f"只用 26 尺车，按总容器数 {total_containers} / 12 计算。"
        else:
            t26 = math.ceil(pallets_final / cap_26_pallets)
            buffer_pallets_est = t26 * cap_26_pallets - pallets_final
            reason = "只用 26 尺车，容器信息缺失，按总托数/12 兜底计算。"

        return {
            "trucks_53": 0,
            "trucks_26": t26,
            "total_trucks": t26,
            "buffer_pallets": buffer_pallets_est,
            "suggestion_reason": reason,
        }

    # --- 2. 只用 53 尺模式 (53_only) ---
    if mode == "53_only":
        t53 = math.ceil(pallets_final / cap_53_pallets)
        buffer = t53 * cap_53_pallets - pallets_final
        return {
            "trucks_53": t53,
            "trucks_26": 0,
            "total_trucks": t53,
            "buffer_pallets": buffer,
            "suggestion_reason": "只用 53 尺车，按总托数/30 计算。",
        }

    # --- 3. 混用模式 (mix)：主力 53 尺 + 尾部 26 尺 ---
    if mode == "mix":
        # 容器信息缺失的兜底逻辑
        if total_containers is None:
            t53 = math.ceil(pallets_final / cap_53_pallets)
            buffer = t53 * cap_53_pallets - pallets_final
            return {
                "trucks_53": t53,
                "trucks_26": 0,
                "total_trucks": t53,
                "buffer_pallets": buffer,
                "suggestion_reason": "容器估算缺失，按总托数/30 建议 53 尺车。",
            }

        # --- 按容器数执行“主力 53 尺 + 尾部 26 尺”逻辑 ---
        # 1. 计算所需 53 尺车数量 (全装满的整数车)
        t53_full = math.floor(total_containers / cap_53_containers)

        # 剩余容器数量
        remaining_containers = total_containers % cap_53_containers

        t53 = t53_full
        t26 = 0
        reason = ""

        # 2. 处理剩余货量
        if remaining_containers == 0:
            # 刚好装满整数车 53 尺车
            t53 = max(1, t53_full)
            reason = f"总容器 {total_containers} 个，刚好装满 {t53} 辆 53 尺车。"

        elif remaining_containers <= cap_26_containers:
            # 剩余 1–12 个容器：用 1 辆 26 尺车
            t26 = 1
            t53 = t53_full
            reason = (
                f"总容器 {total_containers} 个，主力使用 {t53} 辆 53 尺车，"
                f"剩余 {remaining_containers} 个容器，建议用 1 辆 26 尺车装载。"
            )

        else:
            # 剩余容器在 13–59 个之间：多加一辆 53 尺
            t53 = t53_full + 1
            reason = (
                f"总容器 {total_containers} 个，主力使用 {t53_full} 辆 53 尺车后，"
                f"剩余 {remaining_containers} 个容器 (>12)，建议多加 1 辆 53 尺车装载。"
            )

        # 3. 重新计算缓冲托数 (按最终车型组合的总容量 - 实际托数)
        pallets_cap_53 = t53 * cap_53_pallets
        pallets_cap_26 = t26 * cap_26_pallets

        buffer = (pallets_cap_53 + pallets_cap_26) - pallets_final

        return {
            "trucks_53": t53,
            "trucks_26": t26,
            "total_trucks": t53 + t26,
            "buffer_pallets": buffer,
            "suggestion_reason": reason,
        }

    # 理论上不会走到这里
    return {
        "trucks_53": 0,
        "trucks_26": 0,
        "total_trucks": 0,
        "buffer_pallets": 0,
        "suggestion_reason": "未知车型模式",
    }


# =========================
# 工具函数：估任意站点当前托数 (用于 MCO.HUB 拆分展示 & 线路提示)
# =========================
def estimate_pallets_for_station(
    report_df: pd.DataFrame,
    station3: str,
    wa_master: pd.DataFrame,
    board_cap=250,
    gay_cap=450,
) -> int:
    """
    ⚠️ 仅用于：
      1）MCO.HUB 组成拆分展示；
      2）线路提醒（SRQ/TPA、WPB/MCO）；
    不影响主托数主逻辑。
    """
    (
        rep_s,
        route_pkg_s,
        active_routes_s,
        _,
        pkg_total_now_s,
        _,
        _,
    ) = calc_route_pkg_cached(report_df, station3, wa_master)

    if pkg_total_now_s == 0:
        return 0

    # 城市维度站点 / 无路区映射：按总量估托
    if station3 in CITY_ONLY_STATIONS or route_pkg_s is None or route_pkg_s.empty:
        board_boxes = math.ceil(pkg_total_now_s / board_cap)
        pallets_est_s = math.ceil(board_boxes / 2)
        return pallets_est_s

    # 有路区：每个路区至少一个围板箱，再按容量修正，然后全站汇总 /2 得托数
    route_boxes = route_pkg_s["pkg_cnt"].apply(
        lambda x: max(1, math.ceil(x / board_cap))
    )
    total_board_boxes = int(route_boxes.sum())
    pallets_est_s = math.ceil(total_board_boxes / 2)
    return pallets_est_s


def estimate_pallets_for_mcohub(
    report_df: pd.DataFrame,
    wa_master: pd.DataFrame,
    board_cap=250,
    gay_cap=450,
) -> int:
    total_pallets = 0
    for st3 in MCO_HUB_GROUP:
        total_pallets += estimate_pallets_for_station(
            report_df, st3, wa_master, board_cap=board_cap, gay_cap=gay_cap
        )
    return total_pallets


# =========================
# 本站点未来增量 -> 路区当天占比再分摊（仅单站 + 有路区）
# =========================
route_pkg_fc = None
if is_single_station and route_pkg is not None and not route_pkg.empty and pkg_total_now > 0:
    route_pkg_fc = route_pkg.copy()
    share = route_pkg_fc["pkg_cnt"] / pkg_total_now
    route_pkg_fc["future_add"] = (share * forecast_in_station).round().astype(int)
    route_pkg_fc["pkg_cnt_fc"] = route_pkg_fc["pkg_cnt"] + route_pkg_fc["future_add"]
# 串点或 MCO.HUB 模式下 route_pkg_fc 保持 None

# =========================
# 托数估算（核心逻辑）
# =========================
def calc_pallets_with_route(
    pkg_total_now,
    active_routes,
    forecast_in_station,
    board_cnt=None,
    gay_cnt=None,
    board_cap=250,
    gay_cap=450,
    r_gay=0.6,
    route_pkg_fc: pd.DataFrame = None,
    prefer_board_only=True,
    future_container_mode=None,
    target_station3: str = "",
):
    """
    返回：mode, final_cnt, pallets_est, pallets_final, cap_container, est_board_boxes, est_gaylords
    """
    final_cnt = pkg_total_now + forecast_in_station

    # ===== 情况1：已知当前围板箱 + Gaylord 数 =====
    if board_cnt is not None and gay_cnt is not None:
        board_now = int(board_cnt)
        gay_now = int(gay_cnt)

        board_add = gay_add = 0
        if forecast_in_station > 0:
            if future_container_mode == "gay":
                gay_add = math.ceil(forecast_in_station / gay_cap)
            else:
                board_add = math.ceil(forecast_in_station / board_cap)

        board_total = board_now + board_add
        gay_total = gay_now + gay_add

        pallets_est = math.ceil(board_total / 2) + gay_total
        pallets_final = pallets_est
        cap_container = board_total * board_cap + gay_total * gay_cap

        return (
            "container_known",
            final_cnt,
            pallets_est,
            pallets_final,
            cap_container,
            board_total,
            gay_total,
        )

    # ===== 情况2：不知道容器数量，用“估算逻辑”（兜底） =====
    # 2-1 有路区、非城市维度站点（仅单站模式才进入此逻辑）
    if (
        route_pkg_fc is not None
        and not route_pkg_fc.empty
        and target_station3 not in CITY_ONLY_STATIONS
    ):
        # 每路区至少一个围板箱，然后用容量修正
        route_boxes = route_pkg_fc["pkg_cnt_fc"].apply(
            lambda x: max(1, math.ceil(x / board_cap))
        )
        total_board_boxes = int(route_boxes.sum())
        est_board_boxes = total_board_boxes
        est_gaylords = 0

        pallets_est = math.ceil(total_board_boxes / 2)
        pallets_final = pallets_est
        cap_container = est_board_boxes * board_cap

        return (
            "route_board_only",
            final_cnt,
            pallets_est,
            pallets_final,
            cap_container,
            est_board_boxes,
            est_gaylords,
        )

    # 2-2 城市维度站点 或 无路区映射 或 多站点串联：按总量估容器
    if final_cnt <= 0:
        return "no_data", final_cnt, 0, 0, 0, 0, 0

    est_board_boxes = math.ceil(final_cnt / board_cap)
    est_gaylords = 0

    pallets_est = math.ceil(est_board_boxes / 2)
    pallets_final = pallets_est
    cap_container = est_board_boxes * board_cap
    mode_name = "ratio_board_only" if not is_single_station else "city_only"

    return (
        mode_name,
        final_cnt,
        pallets_est,
        pallets_final,
        cap_container,
        est_board_boxes,
        est_gaylords,
    )


# 实际调用：算托 + 容器估算
(
    mode,
    final_cnt,
    pallets_est,
    pallets_final,
    cap_container,
    est_board_boxes,
    est_gaylords,
) = calc_pallets_with_route(
    pkg_total_now,
    active_routes,
    forecast_in_station,
    board_cnt,
    gay_cnt,
    board_cap,
    gay_cap,
    r_gay,
    route_pkg_fc=route_pkg_fc,
    prefer_board_only=prefer_board_only,
    future_container_mode=future_container_mode,
    target_station3=target_station3,
)

# 约车建议（主力 53 + 尾部 26）
truck_plan = calc_trucks_by_type(
    pallets_final,
    mode=truck_mode_key,
    est_board_boxes=est_board_boxes,
    est_gaylords=est_gaylords,
)

# =========================
# 展示区
# =========================
st.caption(
    f"当前结果基于 {snapshot_time} 上传的明细快照，针对站点集合：**{target_station3}**"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("当前包裹总量(本集合)", f"{pkg_total_now:,}")
c2.metric("已集包量", f"{bagged_cnt:,}" if bagged_cnt is not None else "N/A")
c3.metric("未集包量(本集合)", f"{unbagged_cnt:,}" if unbagged_cnt is not None else "N/A")
c4.metric("活跃路区数（单站时）", f"{active_routes:,}" if is_single_station else "N/A (串点模式)")

st.caption(f"全站未集包合计（用于扣产能）：{unbagged_all_cnt:,} 件")

if is_single_station and unmapped_zips > 0:
    st.warning(
        f"⚠️ 有 {unmapped_zips} 个邮编未映射到路区，已按“每邮编=1个虚拟路区”计入最小容器需求。"
        "建议更新该站点 workarea master。"
    )

c5, c6, c7 = st.columns(3)
c5.metric("预计截单前总包裹(本集合)", f"{final_cnt:,}")
c6.metric("估算托数（容量换算）", f"{pallets_est}")
c7.metric("最少托数（当前规则）", f"{pallets_final}")

# 额外展示估算容器数
if (est_board_boxes or est_gaylords):
    c8, c9, _ = st.columns(3)
    c8.metric("估算围板箱数（含未来）", f"{est_board_boxes:,}")
    c9.metric("估算 Gaylord 数（含未来）", f"{est_gaylords:,}")

st.markdown("### 最少约车建议")
reason = truck_plan.get("suggestion_reason", "")
st.success(
    f"✅ 建议最少约 **{truck_plan['total_trucks']}** 车 "
    f"（**{reason}**）"
)
st.write(
    f"53尺车：{truck_plan['trucks_53']} 车（30托/车） | "
    f"26尺车：{truck_plan['trucks_26']} 车（12容器/车）"
)
st.write(f"剩余缓冲托数：{truck_plan['buffer_pallets']} 托（近似折算）")

if mode == "container_known":
    st.info(
        f"当前 + 未来预计约 {est_board_boxes:,} 个围板箱、{est_gaylords:,} 个 Gaylord，"
        f"按 2箱/托 + 1GL/托 估算出 {pallets_final} 托。"
    )
elif mode in {"route_board_only", "ratio_board_only", "city_only"}:
    st.caption(
        f"容器数量为估算值：围板箱≈{est_board_boxes:,} 个、Gaylord≈{est_gaylords:,} 个（默认用围板箱）。"
    )

# ===== MCO.HUB 城市维度估算拆分展示 =====
if "MCO.HUB" in selected_station3_list:
    pallets_mcohub = estimate_pallets_for_mcohub(
        report_df, wa_master, board_cap=board_cap, gay_cap=gay_cap
    )

    st.markdown("### MCO.HUB 组成站点托数估算（当前货量）")
    parts = {}
    for st3 in MCO_HUB_GROUP:
        parts[st3] = estimate_pallets_for_station(
            report_df, st3, wa_master, board_cap=board_cap, gay_cap=gay_cap
        )
    st.write(
        f"OCF ≈ {parts.get('OCF',0)} 托，"
        f"JAX ≈ {parts.get('JAX',0)} 托，"
        f"MCO ≈ {parts.get('MCO',0)} 托，"
        f"合计约 {pallets_mcohub} 托（仅当前货量，未含未来增量分摊）。"
    )

# ===== 路区货量 + 围板箱换算表（仅单站且有路区时展示） =====
if is_single_station and route_pkg_fc is not None and not route_pkg_fc.empty:
    show_df = route_pkg_fc.copy()
    # 用“每路区至少1个围板箱 + 容量”估当前+未来的围板箱数
    show_df["估算围板箱数(个)"] = show_df["pkg_cnt_fc"].apply(
        lambda x: max(1, math.ceil(x / board_cap))
    )
    show_df["估算托数(托，2箱/托)"] = show_df["估算围板箱数(个)"].apply(
        lambda x: int(math.ceil(x / 2))
    )

    st.markdown("---")
    st.markdown("### 路区货量分布（含未来增量按路区比例分摊）")
    st.dataframe(show_df.sort_values("pkg_cnt_fc", ascending=False), use_container_width=True)

    st.download_button(
        "下载路区分布+围板箱换算表",
        data=show_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{target_station3}_route_pkg_with_boardbox.csv",
        mime="text/csv",
    )

# ===== 线路提醒：MIA → SRQ → TPA、MIA → WPB → MCO （只做提示，不改主逻辑） =====
st.markdown("---")
st.markdown("### 线路提醒（不影响托数与约车，仅作运营参考）")

selected_set = set(selected_station3_list)

# 1）MIA → SRQ → TPA 串点建议（当前货量）
if "SRQ" in raw_stations and "TPA" in raw_stations and selected_set & {"SRQ", "TPA"}:
    pallets_srq = estimate_pallets_for_station(
        report_df, "SRQ", wa_master, board_cap=board_cap, gay_cap=gay_cap
    )
    pallets_tpa = estimate_pallets_for_station(
        report_df, "TPA", wa_master, board_cap=board_cap, gay_cap=gay_cap
    )
    total_st = pallets_srq + pallets_tpa
    if 0 < total_st <= 30:
        st.info(
            f"📌 线路提醒（SRQ/TPA）：当前 SRQ 约 {pallets_srq} 托，TPA 约 {pallets_tpa} 托，"
            f"合计约 {total_st} 托（基于当前货量估算）。\n\n"
            f"可考虑采用 **“MIA → SRQ → TPA” 一车串点线路**，两站点共用一辆 53 尺车，"
            f"减少单独干线车辆需求。"
        )

# 2）MIA → WPB → MCO 串点建议（当前货量）
has_mco_substation_today = any(s in raw_stations for s in MCO_HUB_GROUP)

# ✅ 放宽条件：只要勾了 WPB 或 MCO.HUB 其中一个，就提醒
if "WPB" in raw_stations and has_mco_substation_today and ("WPB" in selected_set or "MCO.HUB" in selected_set):
    pallets_wpb = estimate_pallets_for_station(
        report_df, "WPB", wa_master, board_cap=board_cap, gay_cap=gay_cap
    )
    pallets_mcohub_now = estimate_pallets_for_mcohub(
        report_df, wa_master, board_cap=board_cap, gay_cap=gay_cap
    )

    cap_53 = 30

    if pallets_wpb > 0 and pallets_mcohub_now > 0:
        full_trucks_wpb = pallets_wpb // cap_53
        last_pallets_wpb = pallets_wpb % cap_53
        total_wpbhub = pallets_wpb + pallets_mcohub_now

        # 场景 A：两边货量都不满一车，合起来 ≤1 车
        if total_wpbhub <= cap_53:
            st.info(
                f"📌 线路提醒（WPB/MCO，一车合并）：当前 WPB 约 {pallets_wpb} 托，"
                f"MCO.HUB 合计约 {pallets_mcohub_now} 托，总计约 {total_wpbhub} 托。\n\n"
                f"可考虑采用 **“MIA → WPB → MCO” 一车串点线路**，WPB 与 MCO.HUB 共用一辆 53 尺车，"
                f"两边都无需再额外增加干线车。"
            )

        # 场景 B：WPB 有至少 1 辆满载车 + 最后一车较空，可以和 MCO.HUB 拼一车
        elif full_trucks_wpb >= 1 and last_pallets_wpb > 0 and (last_pallets_wpb + pallets_mcohub_now) <= cap_53:
            last_truck_index = full_trucks_wpb + 1
            combined = last_pallets_wpb + pallets_mcohub_now
            st.info(
                f"📌 线路提醒（WPB/MCO，最后一车拼载）：当前 WPB 共约 {pallets_wpb} 托，"
                f"约 {full_trucks_wpb} 辆 53 尺车满载 + 第 {last_truck_index} 辆约 {last_pallets_wpb} 托；"
                f"MCO.HUB 合计约 {pallets_mcohub_now} 托。\n\n"
                f"WPB 第 {last_truck_index} 辆车剩余 {last_pallets_wpb} 托 + MCO.HUB {pallets_mcohub_now} 托 "
                f"≈ {combined} 托，可考虑合并装成 1 辆 53 尺车，线路 **“MIA → WPB → MCO”**，"
                f"有助于提升最后一车的装载率。"
            )

# =========================
# 缓存控制
# =========================
st.sidebar.markdown("---")
if st.sidebar.button("🔄 清空缓存并重算（比如workarea更新后）"):
    st.cache_data.clear()
    st.rerun()
