#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
赛道A 数据处理 v3 —— 对齐原论文 (Schreurs et al., Nat Commun 2024, 15:2368) 的方法

论文 Methods 原文要点（第 12 页）：
  - "Chemical properties with log-normal distributions (Shapiro-Wilk test, p<0.05)
     were log-transformed."
  - "Missing chemical measurements (0.1% of all data) were replaced with
     mean values per attribute."
  - "training set (70%, 175 beers) / test set (30%, 75 beers), stratified per beer style"
  - "Chemical measurements (p=231) were normalized based on the training set
     average and standard deviation."

v3 相对 v2 的增量：
  1. NA 填补由「中位数」改为「均值」——与论文一致（保留中位数版作敏感性对照）
  2. 新增逐列「地板值 / 左删失」检测（不只认 -1.8178，覆盖 CS2 / X4VG 等）
  3. 新增数据字典 data_dictionary.csv
  4. 新增感官字段名称映射 sensory_name_mapping.csv（4.csv ↔ 7.csv ↔ 8.xlsx）
  5. 新增 70/30 分层切分 split_train_test.csv（按 tasting_category_fine）
  6. 新增按训练集 mean/std 标准化的特征矩阵 1_X_log_normalized.csv
  7. 新增孤立极端异常值（录入/单位错误）识别与处理（v3-final）：
     规则——「极端」(MAX/中位数>100) 为前提，且满足以下任一形态即判定：
     A「孤立」MAX/次大值>20；B「超分位」MAX/P99>15（覆盖存在两个相邻高值、
     不满足孤立性、但整体分布仍被单点严重拉偏的录入/单位错误，如
     X3.methylbutanal）。真实高浓度（重尾分布）因不满足极端性而保留，
     避免误删真实风味差异。命中点按缺失值处理（列均值填补）。
     实测共命中 3 个：X2.methyltetrahydrofuran.3.one、X.E..beta.damascenone、
     X3.methylbutanal（单点超同列中位数最高达上千倍）。
输出目录： processed/
"""
import csv, os, math
from collections import Counter, OrderedDict
import openpyxl

SRC = r"C:\Users\Olivia\Desktop\A\附件"
OUT = r"C:\Users\Olivia\Desktop\A\processed"
os.makedirs(OUT, exist_ok=True)

BDL = -1.8178
TOL = 1e-4
RAW_HINT = ["ethanol", "color", "bitterness", "ph", "co2", "sugars",
            "protein", "caloric", "kcal", "iron", "betaglucan", "ammonia", "glycerol"]
REPORT = []


def log(m=""):
    print(m)
    REPORT.append(str(m))


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
        return list(csv.reader(fh))


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    log("   -> %-42s %5d x %4d" % (name, len(rows), len(header)))


def fnum(s):
    if s is None:
        return None
    if isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        v = float(s)
        return None if math.isnan(v) or math.isinf(v) else v
    s = str(s).strip()
    if s == "" or s.upper() in ("NA", "NAN", "NULL", "NONE"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def mean(v):
    return sum(v) / len(v) if v else None


def median(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def percentile(v, q):
    """线性插值分位数（与 numpy 默认 'linear' 口径一致）。"""
    s = sorted(v)
    n = len(s)
    if n == 0:
        return None
    h = (n - 1) * q / 100.0
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (h - lo)


def is_bdl(x):
    return x is not None and abs(x - BDL) < TOL


def fmt(x):
    return "%.6g" % x


def bid_of(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


# ---------- 官方校准表 ----------
def load_official():
    wb = openpyxl.load_workbook(os.path.join(SRC, "8.xlsx"), data_only=True)
    ws = wb["Calibrated chemical dataset"]
    data = [r for r in ws.iter_rows(values_only=True) if any(x is not None for x in r)]
    hdr = [("" if h is None else str(h)) for h in data[0]]
    rows = [[("" if v is None else v) for v in r] for r in data[1:]]
    return hdr, rows


def judge_scale(hdr_c, rows_c, header, body):
    """用官方校准表判定 log10 / raw。"""
    name2i = {h: i for i, h in enumerate(header)}
    idx1 = {r[1]: r for r in body}
    idx_c = {bid_of(r[1]): r for r in rows_c}
    ids = [k for k in idx_c if k in idx1]
    out = {}
    for c in hdr_c[3:]:
        if c not in name2i:
            continue
        p1, pc = name2i[c], hdr_c.index(c)
        ok_l = ok_r = tot = 0
        for bid in ids:
            a, b = fnum(idx1[bid][p1]), fnum(idx_c[bid][pc])
            if a is None or b is None:
                continue
            tot += 1
            if abs(a - b) <= max(1e-5, 1e-3 * abs(b)):
                ok_r += 1
            if is_bdl(a):
                if b == 0:
                    ok_l += 1
            elif b > 0 and abs(a - math.log10(b)) <= 1e-3:
                ok_l += 1
        if not tot:
            continue
        fl, fr = ok_l / tot, ok_r / tot
        if fl > 0.9 and fl >= fr:
            v = "log10"
        elif fr > 0.9 and fr > fl:
            v = "raw"
        else:
            v = "log10" if fr == 0 else "unsure"
        out[c] = (v, round(fl, 3), round(fr, 3))
    return out


# ---------- 主流程 ----------
def main():
    log("=" * 78)
    log("赛道A 数据处理 v3（对齐 Nat Commun 2024 论文口径）")
    log("=" * 78)

    rows = read_csv(os.path.join(SRC, "1.csv"))
    header, body = rows[0], rows[1:]
    comps = header[3:]
    n = len(body)
    idx_of = {c: 3 + i for i, c in enumerate(comps)}

    # ---------- 先划分，再做统计（避免验证/测试信息进入填补与阈值） ----------
    j_beer = header.index("beer")
    j_style = header.index("tasting_category_fine")
    styles = [r[j_style] for r in body]
    from sklearn.model_selection import train_test_split
    tr_list, te_list = train_test_split(
        list(range(n)), test_size=0.30, random_state=0, shuffle=True, stratify=styles)
    train_idx = set(tr_list)

    # ---------- 逐列统计 + 地板值/删失 ----------
    # 统计口径：疑似异常阈值与填补均值「仅由训练集」估计，再统一应用到全体样本；
    # 交叉验证的折内重估在建模脚本中完成（见 1_X_log_missing.csv）。
    stats = OrderedDict()
    total_na = total_bdl = 0
    for c in comps:
        ci = idx_of[c]
        vals = []
        vals_tr = []
        n_na = n_bdl = 0
        for i, r in enumerate(body):
            v = fnum(r[ci])
            if v is None:
                n_na += 1
                continue
            if is_bdl(v):
                n_bdl += 1
            vals.append(v)
            if i in train_idx:
                vals_tr.append(v)
        cnt = Counter(round(v, 6) for v in vals)
        floor_v, floor_n = cnt.most_common(1)[0] if cnt else (None, 0)
        floor_frac = floor_n / len(vals) if vals else 0.0
        censored = floor_frac >= 0.05 and floor_v is not None and floor_v <= median(vals)
        stats[c] = dict(n_na=n_na, n_bdl=n_bdl, vals=vals, vals_tr=vals_tr,
                        mean=mean(vals), median=median(vals),
                        min=min(vals) if vals else None,
                        max=max(vals) if vals else None,
                        floor_v=floor_v, floor_n=floor_n,
                        floor_frac=floor_frac, censored=censored)
        total_na += n_na
        total_bdl += n_bdl

    log("  样本数 = %d ; 化合物列 = %d" % (n, len(comps)))
    log("  NA = %d (%.4f%%) ; -1.8178 哨兵 = %d"
        % (total_na, 100.0 * total_na / (n * len(comps)), total_bdl))
    n_cens = sum(1 for s in stats.values() if s["censored"])
    log("  检出「地板值聚集」的列(占比>=5%%) = %d ; 其中地板值非 -1.8178 的 = %d"
        % (n_cens, sum(1 for s in stats.values()
                       if s["censored"] and abs(s["floor_v"] + 1.8178) > TOL)))

    # ---------- 尺度判定 ----------
    hdr_c, rows_c = load_official()
    off = judge_scale(hdr_c, rows_c, header, body)
    for c in comps:
        s = stats[c]
        if c in off and off[c][0] in ("log10", "raw"):
            s["scale"], s["conf"] = off[c][0], "official"
            s["reason"] = "官方校准表(log=%.3f/raw=%.3f)" % (off[c][1], off[c][2])
        elif s["n_bdl"] > 0:
            s["scale"], s["conf"] = "log10", "inferred"
            s["reason"] = "含-1.8178哨兵"
        elif s["max"] is not None and s["max"] > 20:
            s["scale"], s["conf"] = "raw", "inferred"
            s["reason"] = "max=%.1f超log范围" % s["max"]
        elif any(k in c.lower() for k in RAW_HINT):
            s["scale"], s["conf"] = "raw", "inferred"
            s["reason"] = "理化参数命名"
        else:
            s["scale"], s["conf"] = "log10", "inferred"
            s["reason"] = "默认log10"
    log("  尺度: log10=%d, raw=%d"
        % (sum(1 for s in stats.values() if s["scale"] == "log10"),
           sum(1 for s in stats.values() if s["scale"] == "raw")))

    # ---------- 孤立极端异常值（录入/单位错误）识别 ----------
    # 前提「极端」MAX/中位数>100，且满足以下任一形态：
    #   A「孤立」MAX/次大值>20 —— 单点远超其余所有观测；
    #   B「超分次位」MAX/P99>15 —— 存在两个相邻高值不满足孤立性、但整体分布
    #     仍被单点严重拉偏（如 X3.methylbutanal：前二高值 127.06/78.88 远超
    #     第三大 6.21，MAX/次大仅 1.6 倍，但 MAX/P99=20.9、MAX/中位数=1172）。
    # 真实高浓度(重尾分布, 如石竹烯/丁香酚)不命中, 予以保留。
    n_outlier = 0
    for c in comps:
        s = stats[c]
        s["outlier_val"] = None
        s["clean_mean"] = mean(s["vals_tr"]) if s["vals_tr"] else s["mean"]
        # 疑似异常识别只使用化学测量本身，不使用任何感官标签，属无监督的数据有效性核查，
        # 故在全样本上进行；而填补均值仍只由训练集估计（见 clean_mean）。
        vals = sorted(s["vals"], reverse=True)
        if len(vals) >= 10:
            max1, max2 = vals[0], vals[1]
            med = median(s["vals"])
            p99 = percentile(s["vals"], 99)
            extreme = med and med > 0 and max1 / med > 100
            rule_a = max2 > 0 and max1 / max2 > 20
            rule_b = p99 and p99 > 0 and max1 / p99 > 15
            if extreme and (rule_a or rule_b):
                s["outlier_val"] = max1
                # 填补均值：仅用训练集，并排除已标记的疑似异常值本身
                rest_tr = [x for x in s["vals_tr"] if x != max1]
                s["clean_mean"] = mean(rest_tr) if rest_tr else mean(s["vals_tr"])
                n_outlier += 1
                tag = "孤立" if rule_a else "超P99"
                log("  极端异常[%s]: %s = %.6g (次大=%.6g, P99=%.6g, 中位=%.6g)"
                    " -> 置缺失, 列均值填补" % (tag, c, max1, max2, p99, med))
    log("  孤立极端异常观测 = %d 个（判定为录入/单位错误，按缺失值处理）" % n_outlier)
    out_rows = [[c, fmt(stats[c]["outlier_val"]), fmt(stats[c]["clean_mean"])]
                for c in comps if stats[c]["outlier_val"] is not None]
    if out_rows:
        write_csv("1_outlier_log.csv",
                  ["column", "outlier_value", "imputed_mean_after_removal"], out_rows)

    # ---------- 构建 X_log / X_cal / X_norm ----------
    # 论文：NA 用 per-attribute mean 填补；本队补充：孤立极端异常值同样按缺失处理
    X_log, X_cal, X_nan = [], [], []
    for r in body:
        rl, rc, rn = list(r[:3]), list(r[:3]), list(r[:3])
        for c in comps:
            v = fnum(r[idx_of[c]])
            s = stats[c]
            if s["outlier_val"] is not None and v == s["outlier_val"]:
                v = None                      # 疑似异常按缺失处理
            if v is None:
                rn.append("")                 # 保留缺失标记，供折内重新估计
                v = s["clean_mean"] if s["clean_mean"] is not None else 0.0
                rl.append(fmt(v))
                rc.append(fmt(10 ** v if s["scale"] == "log10" else v))
                continue
            rn.append(fmt(v))
            rl.append(fmt(v))
            rc.append("0" if is_bdl(v) else fmt(10 ** v if s["scale"] == "log10" else v))
        X_log.append(rl)
        X_cal.append(rc)
        X_nan.append(rn)
    write_csv("1_X_log.csv", header, X_log)
    write_csv("1_X_cal.csv", header, X_cal)
    write_csv("1_X_log_missing.csv", header, X_nan)

    # ---------- 70/30 分层切分（按 beer style） ----------
    j_beer = header.index("beer")
    j_style = header.index("tasting_category_fine")
    bids = [r[1] for r in body]
    beers = [r[j_beer] for r in body]
    styles = [r[j_style] for r in body]
    # 与原文一致：train_test_split(test_size=0.30, random_state=0, shuffle=True, stratify=风格)
    from sklearn.model_selection import train_test_split
    tr_list, te_list = train_test_split(
        list(range(n)), test_size=0.30, random_state=0, shuffle=True, stratify=styles)
    train_idx = set(tr_list)
    n_styles = len(set(styles))
    split_rows = []
    for i in range(n):
        split_rows.append([beers[i], bids[i], styles[i],
                           "train" if i in train_idx else "test"])
    write_csv("split_train_test.csv", ["beer", "beer_id", "style", "split"], split_rows)
    log("  切分: train=%d, test=%d (按 %d 个风格分层, random_state=0)"
        % (len(train_idx), n - len(train_idx), n_styles))

    # ---------- 按训练集 mean/std 标准化（论文口径） ----------
    tr = sorted(train_idx)
    norm_rows = []
    for r in X_log:
        norm_rows.append(list(r[:3]))
    for c in comps:
        ci = idx_of[c]
        tv = [fnum(X_log[i][ci]) for i in tr]
        tv = [x for x in tv if x is not None]
        mu = mean(tv) or 0.0
        sd = math.sqrt(sum((x - mu) ** 2 for x in tv) / len(tv)) if tv else 1.0
        if sd == 0:
            sd = 1.0
        for i in range(n):
            norm_rows[i].append("%.6f" % ((fnum(X_log[i][ci]) - mu) / sd))
    write_csv("1_X_log_normalized.csv", header, norm_rows)

    # ---------- 列尺度 + 删失映射表 ----------
    write_csv("1_column_scale_map.csv",
              ["column", "col_index", "scale", "confidence", "n_na", "n_bdl",
               "floor_value", "floor_count", "floor_frac", "censored",
               "min_obs", "max_obs", "mean_obs", "median_obs", "judge_reason"],
              [[c, idx_of[c], stats[c]["scale"], stats[c]["conf"],
                stats[c]["n_na"], stats[c]["n_bdl"],
                fmt(stats[c]["floor_v"]) if stats[c]["floor_v"] is not None else "",
                stats[c]["floor_n"], round(stats[c]["floor_frac"], 4),
                "Y" if stats[c]["censored"] else "N",
                fmt(stats[c]["min"]) if stats[c]["min"] is not None else "",
                fmt(stats[c]["max"]) if stats[c]["max"] is not None else "",
                fmt(stats[c]["mean"]) if stats[c]["mean"] is not None else "",
                fmt(stats[c]["median"]) if stats[c]["median"] is not None else "",
                stats[c]["reason"]] for c in comps])

    # ---------- 数据字典 ----------
    dd = []
    dd.append(["1.csv", "beer", "样本编号(1-250)", "int", "主键(推荐)"])
    dd.append(["1.csv", "beer_id", "EAN/条形码", "str(13位)", "不可跨文件join(4/7.csv已丢精度)"])
    dd.append(["1.csv", "tasting_category_fine", "啤酒风格标签", "category", "22 类"])
    for c in comps:
        s = stats[c]
        dd.append(["1.csv", c, "化合物丰度",
                   ("log10(浓度)" if s["scale"] == "log10" else "原始浓度"),
                   ("删失(地板值%.6g)" % s["floor_v"]) if s["censored"] else ""])
    for i, c in enumerate(header[3:]):
        pass
    r4 = read_csv(os.path.join(SRC, "4.csv"))
    for c in r4[0][3:]:
        dd.append(["4.csv", c, "感官评分(已中心化z-score)", "continuous", "训练目标"])
    write_csv("data_dictionary.csv",
              ["file", "column", "meaning", "scale_or_type", "note"], dd)

    # ---------- 感官名称映射（4.csv ↔ 7.csv ↔ 8.xlsx） ----------
    wb = openpyxl.load_workbook(os.path.join(SRC, "8.xlsx"), data_only=True)
    ws = wb["Sensory names"]
    nr = [r for r in ws.iter_rows(values_only=True) if any(x is not None for x in r)]
    nh = [("" if x is None else str(x)) for x in nr[0]]
    ix = {k: nh.index(k) for k in ["dataset", "R_name", "Category_Paper", "Category",
                                   "CleanName_Paper", "CleanName_PaperLong", "DataType"]}
    panel = {}
    rate = {}
    for r in nr[1:]:
        ds = str(r[ix["dataset"]]).strip()
        rn = str(r[ix["R_name"]]).strip()
        rec = [str(r[ix["Category_Paper"]]).strip(), str(r[ix["Category"]]).strip(),
               str(r[ix["CleanName_Paper"]]).strip(), str(r[ix["CleanName_PaperLong"]]).strip(),
               str(r[ix["DataType"]]).strip()]
        (panel if ds == "Trained panel" else rate)[rn] = rec
    # 用 CleanName_Paper 作为桥，映射 4.csv 与 7.csv
    r7 = read_csv(os.path.join(SRC, "7.csv"))
    c4 = r4[0][3:]
    c7 = r7[0][3:]
    panel_by_clean = {}
    for rn, rec in panel.items():
        panel_by_clean.setdefault(rec[2], []).append(rn)
    rate_by_clean = {}
    for rn, rec in rate.items():
        rate_by_clean.setdefault(rec[2], []).append(rn)

    map_rows = []
    for c in c4:
        rec = panel.get(c)
        if not rec:
            map_rows.append([c, "", "", "", "", "", ""])
            continue
        clean = rec[2]
        matches = [x for x in c7 if x in rate_by_clean.get(clean, [])]
        map_rows.append([c, rec[0], rec[1], clean, rec[4],
                         ";".join(matches) if matches else "",
                         "映射成功" if matches else "无对应(仅panel独有)"])
    write_csv("sensory_name_mapping.csv",
              ["panel_column_4csv", "category_paper", "category", "clean_name",
               "data_type", "ratebeer_column_7csv", "status"], map_rows)
    n_num = sum(1 for r in map_rows if r[4] == "numeric")
    n_cat = sum(1 for r in map_rows if r[4] == "categorical")
    log("  感官映射: 4.csv 共 %d 维 -> numeric=%d, categorical=%d"
        % (len(map_rows), n_num, n_cat))

    # ---------- 其余文件导出 ----------
    for fn, out, sq in [("2.csv", "2_corr_chemical_clean.csv", True),
                        ("5.csv", "5_corr_sensory_clean.csv", True),
                        ("6.csv", "6_corr_chem_sensory_clean.csv", False)]:
        rr = read_csv(os.path.join(SRC, fn))
        bb = [r for r in rr[1:] if any(x.strip() for x in r)]
        write_csv(out, rr[0], bb)

    write_csv("4_Y_sensory_standardized.csv", r4[0], [list(r) for r in r4[1:]])
    write_csv("7_forum_sensory_imputed.csv", r7[0], [list(r) for r in r7[1:]])

    for name in wb.sheetnames:
        w = wb[name]
        d = [r for r in w.iter_rows(values_only=True) if any(x is not None for x in r)]
        if not d:
            continue
        h = [("" if x is None else str(x)) for x in d[0]]
        write_csv("8_%s.csv" % name.replace(" ", "_"), h,
                  [[("" if v is None else v) for v in r] for r in d[1:]])

    with open(os.path.join(OUT, "processing_log_v3.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    log("\n完成 -> %s" % OUT)


if __name__ == "__main__":
    main()
