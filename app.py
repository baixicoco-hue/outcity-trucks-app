def calc_trucks_by_type(
    pallets_final: int,
    mode: str = "mix",
    cap_53_pallets: int = 30,
    cap_26_pallets: int = 12,
    cap_26_containers: int = 12,  # 先保留参数，后面如果要用容器约束还可以继续扩展
    est_board_boxes: int | None = None,
    est_gaylords: int | None = None,
):
    """
    按托数“全局找最优组合”：
    1) 车数最少
    2) 缓冲托数最少
    3) 同样的情况下优先多用 53 尺车
    """
    if pallets_final <= 0:
        return {
            "trucks_53": 0,
            "trucks_26": 0,
            "total_trucks": 0,
            "buffer_pallets": 0,
            "suggestion_reason": "无货物",
        }

    # 估算总容器数（只是用于文案说明，不再作为硬约束）
    total_containers = None
    if est_board_boxes is not None and est_gaylords is not None:
        total_containers = est_board_boxes + est_gaylords

    # 粗略认为 53 尺车最多装 30 托 * 2 箱/托 ≈ 60 个容器
    cap_53_containers = cap_53_pallets * 2

    # --- 1. 只用 26 尺 ---
    if mode == "26_only":
        t26 = math.ceil(pallets_final / cap_26_pallets)
        buffer_pallets_est = t26 * cap_26_pallets - pallets_final
        reason = "只用 26 尺车，按总托数/12 兜底计算。"
        return {
            "trucks_53": 0,
            "trucks_26": t26,
            "total_trucks": t26,
            "buffer_pallets": buffer_pallets_est,
            "suggestion_reason": reason,
        }

    # --- 2. 只用 53 尺 ---
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

    # --- 3. 混用模式：枚举组合，选“最少车 + 缓冲最小”的方案 ---
    if mode == "mix":
        # 给个安全上界，多加 2 作为缓冲
        max_53 = math.ceil(pallets_final / cap_53_pallets) + 2
        max_26 = math.ceil(pallets_final / cap_26_pallets) + 2

        best = None
        best_key = None

        for t53 in range(0, max_53 + 1):
            for t26 in range(0, max_26 + 1):
                if t53 == 0 and t26 == 0:
                    continue

                pallets_cap = t53 * cap_53_pallets + t26 * cap_26_pallets
                if pallets_cap < pallets_final:
                    # 托数不够，直接跳过
                    continue

                trucks = t53 + t26
                buffer = pallets_cap - pallets_final

                # 关键排序：
                #   1) 车数少优先
                #   2) 缓冲托数少优先
                #   3) 同样情况下，53 多的优先（-t53）
                key = (trucks, buffer, -t53)

                if best is None or key < best_key:
                    best = (t53, t26, pallets_cap, buffer)
                    best_key = key

        # 理论上不会 None，这里兜底一下
        if best is None:
            t53 = math.ceil(pallets_final / cap_53_pallets)
            buffer = t53 * cap_53_pallets - pallets_final
            return {
                "trucks_53": t53,
                "trucks_26": 0,
                "total_trucks": t53,
                "buffer_pallets": buffer,
                "suggestion_reason": "兜底：按总托数/30 建议 53 尺车。",
            }

        t53, t26, pallets_cap, buffer = best

        if total_containers is not None:
            reason = (
                f"总容器约 {total_containers} 个，按托数最优组合为 "
                f"{t53} 辆 53 尺车 + {t26} 辆 26 尺车 "
                f"(容量约 {pallets_cap} 托，比需求多 {buffer} 托)。"
            )
        else:
            reason = (
                f"按托数最优组合为 {t53} 辆 53 尺车 + {t26} 辆 26 尺车 "
                f"(容量约 {pallets_cap} 托，比需求多 {buffer} 托)。"
            )

        return {
            "trucks_53": t53,
            "trucks_26": t26,
            "total_trucks": t53 + t26,
            "buffer_pallets": buffer,
            "suggestion_reason": reason,
        }
