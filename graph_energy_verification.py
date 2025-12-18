import networkx as nx
import numpy as np
from collections import defaultdict
import os
import multiprocessing as mp
from networkx.generators.atlas import graph_atlas_g
import pickle
import time
import signal

# ===================== 核心配置（仅需修改这里！） =====================
# macOS用户名（终端执行whoami查看）
MAC_USERNAME = "czb"
# G6文件路径（n=9/10，文件放桌面）
G6_FILE_PATHS = {
    9: f"/Users/{MAC_USERNAME}/Desktop/graphs_9.g6",
    10: f"/Users/{MAC_USERNAME}/Desktop/graphs_10.g6"
}
# 缓存文件路径（避免重复解析g6）
CACHE_FILE_PATHS = {
    n: f"/Users/{MAC_USERNAME}/Desktop/graph_cache_{n}.pkl" for n in range(3, 11)
}
# 并行配置（解决卡顿关键：8G内存设为1，16G设为2）
PARALLEL_PROCESSES = 1  # 优先单进程，避免多进程卡死
BATCH_SIZE = 100  # 小批次处理，避免内存溢出
CALC_TIMEOUT = 5  # 特征值计算超时（秒），跳过慢图
# 验证类型："positive"（正）、"negative"（负）、"both"（全部）
VERIFY_TYPE = "both"

# ===================== 内存缓存+断点续跑 =====================
GRAPH_MEM_CACHE = {}  # 单次运行内存缓存
CHECKPOINT_FILE = f"/Users/{MAC_USERNAME}/Desktop/verify_checkpoint.txt"


# ===================== 工具函数（正/负能量+基础判断） =====================
def adjacency_matrix(graph, n):
    """生成n×n邻接矩阵"""
    return nx.to_numpy_array(graph, nodelist=range(n))


def positive_cubic_sum(eigenvalues):
    """正特征值立方和（E3_plus）"""
    return sum(ev ** 3 for ev in eigenvalues if ev > 0)


def absolute_cubic_sum(eigenvalues):
    """负特征值绝对值立方和（E3^-）"""
    return sum(abs(ev) ** 3 for ev in eigenvalues if ev < 0)


def is_path_graph(G):
    """判断是否为路径图P_n"""
    return nx.is_isomorphic(G, nx.path_graph(G.number_of_nodes()))


def is_complete_graph(G, n):
    """判断是否为完全图K_n"""
    return G.number_of_edges() == n * (n - 1) // 2


# ===================== 缓存+断点工具函数 =====================
def save_graph_cache(n, graphs):
    """保存图到磁盘缓存"""
    cache_file = CACHE_FILE_PATHS[n]
    try:
        g6_mtime = os.path.getmtime(G6_FILE_PATHS[n]) if n in [9, 10] else time.time()
        with open(cache_file, 'wb') as f:
            pickle.dump({"graphs": graphs, "g6_mtime": g6_mtime}, f)
        print(f"[n={n}] 缓存已保存：{cache_file}")
    except Exception as e:
        print(f"[n={n}] 保存缓存失败：{e}")


def load_graph_cache(n):
    """加载磁盘缓存（校验有效性）"""
    cache_file = CACHE_FILE_PATHS[n]
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        if n in [9, 10] and cache_data["g6_mtime"] != os.path.getmtime(G6_FILE_PATHS[n]):
            os.remove(cache_file)
            return None
        print(f"[n={n}] 从缓存加载：{len(cache_data['graphs'])} 个图")
        return cache_data["graphs"]
    except Exception as e:
        print(f"[n={n}] 加载缓存失败：{e}")
        return None


def save_checkpoint(n, processed_idx):
    """保存断点（已处理的图索引）"""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(f"{n},{processed_idx}")


def load_checkpoint(n):
    """加载断点"""
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            line = f.read().strip()
            if line:
                ckpt_n, ckpt_idx = line.split(",")
                if int(ckpt_n) == n:
                    print(f"[n={n}] 从断点 {ckpt_idx} 继续")
                    return int(ckpt_idx)
        return 0
    except Exception as e:
        print(f"[n={n}] 加载断点失败：{e}")
        return 0


# ===================== 图加载函数（兼容n=3~10） =====================
def load_graphs(n):
    """加载n节点连通图（n≤8用atlas，n≥9用g6）"""
    # 1. 内存缓存
    if n in GRAPH_MEM_CACHE:
        return GRAPH_MEM_CACHE[n]
    # 2. 磁盘缓存
    cache_graphs = load_graph_cache(n)
    if cache_graphs:
        GRAPH_MEM_CACHE[n] = cache_graphs
        return cache_graphs
    # 3. 原始加载
    graphs = []
    if 3 <= n <= 8:
        atlas = graph_atlas_g()
        graphs = [G for G in atlas if G.number_of_nodes() == n and nx.is_connected(G)]
        print(f"[n={n}] 从atlas加载：{len(graphs)} 个图")
    elif n in [9, 10]:
        g6_file = G6_FILE_PATHS[n]
        if not os.path.exists(g6_file):
            print(f"\n⚠️ [n={n}] G6文件不存在：{g6_file}")
            print(f"   终端执行：cd ~/Desktop && geng -c {n} > graphs_{n}.g6")
            return []
        # 解析g6文件
        with open(g6_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        total = len(lines)
        print(f"[n={n}] 解析g6文件（共{total}个图）")
        for idx, line in enumerate(lines):
            try:
                G = nx.from_graph6_bytes(line.encode('utf-8'))
                if G.number_of_nodes() == n:
                    graphs.append(G)
                if idx % 50000 == 0 and idx > 0:
                    print(f"[n={n}] 解析进度：{idx}/{total}")
            except:
                continue
        print(f"[n={n}] 解析完成：{len(graphs)} 个图")
        save_graph_cache(n, graphs)  # 保存缓存
    # 4. 存入内存缓存
    GRAPH_MEM_CACHE[n] = graphs
    return graphs


# ===================== 并行处理函数（带超时） =====================
def timeout_handler(signum, frame):
    """超时处理器"""
    raise TimeoutError("特征值计算超时")


def process_single_graph(args):
    """单图验证（正/负能量）"""
    G, n, verify_type, E3_Pn_pos, bound_pos, E3_Kn_neg = args
    try:
        # 跳过完全图/P3（负能量验证逻辑）
        if verify_type in ["negative", "both"] and (is_complete_graph(G, n) or (n == 3 and is_path_graph(G))):
            return None

        # 特征值计算（带超时）
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(CALC_TIMEOUT)
        A_G = adjacency_matrix(G, n)
        eigvals_G = np.linalg.eigh(A_G)[0]  # 稳定计算实特征值
        signal.alarm(0)

        violations = []
        # 验证正能量
        if verify_type in ["positive", "both"]:
            E3_G_pos = positive_cubic_sum(eigvals_G)
            if E3_G_pos + 1e-8 < E3_Pn_pos:
                violations.append(("positive", "E3_plus(G) < E3_plus(P_n)", round(E3_G_pos, 4), round(E3_Pn_pos, 4)))
            if n > 3 and not is_path_graph(G) and E3_G_pos + 1e-8 < bound_pos:
                violations.append(("positive", "E3_plus(G) < (√5/2)*n", round(E3_G_pos, 4), round(bound_pos, 4)))
        # 验证负能量
        if verify_type in ["negative", "both"]:
            E3_G_neg = absolute_cubic_sum(eigvals_G)
            if E3_G_neg < E3_Kn_neg or E3_G_neg < n:
                violations.append(
                    ("negative", "E3^-(G) < n or E3^-(G) < E3^-(K_n)", round(E3_G_neg, 4), round(E3_Kn_neg, 4)))

        if violations:
            return (G, violations, eigvals_G)
        return None
    except TimeoutError:
        print(f"[n={n}] 图 {list(G.edges())} 计算超时，跳过")
        return None
    except Exception as e:
        print(f"[n={n}] 图 {list(G.edges())} 出错：{e}，跳过")
        return None


# ===================== 统一验证函数 =====================
def verify_graph_energy(n):
    """验证n节点图的能量条件"""
    graphs = load_graphs(n)
    if not graphs:
        return []

    # 计算基准值
    Pn = nx.path_graph(n)
    A_Pn = adjacency_matrix(Pn, n)
    eigvals_Pn = np.linalg.eigh(A_Pn)[0]
    E3_Pn_pos = positive_cubic_sum(eigvals_Pn)  # 正能量基准
    bound_pos = (np.sqrt(5) / 2) * n  # 正能量下界
    E3_Kn_neg = n - 1  # 负能量基准（K_n的E3^-）

    print(f"\n[n={n}] 验证参数：")
    if VERIFY_TYPE in ["positive", "both"]:
        print(f"  正能量-Pn: {round(E3_Pn_pos, 4)} | 正能量下界: {round(bound_pos, 4)}")
    if VERIFY_TYPE in ["negative", "both"]:
        print(f"  负能量-Kn: {round(E3_Kn_neg, 4)} | 负能量下界: {n}")

    # 加载断点
    start_idx = load_checkpoint(n)
    tasks = [(G, n, VERIFY_TYPE, E3_Pn_pos, bound_pos, E3_Kn_neg) for G in graphs[start_idx:]]
    violations = []

    # 单进程/并行处理（优先单进程避免卡死）
    if PARALLEL_PROCESSES == 1:
        for idx, task in enumerate(tasks):
            res = process_single_graph(task)
            if res:
                G, vio_list, eigvals = res
                violations.append((G, vio_list, eigvals))
                # 打印违规详情
                print(f"\n[n={n}] 违规图 {G.edges()}:")
                print(f"  特征值: {np.round(eigvals, 4)}")
                for vio_type, reason, val, ref in vio_list:
                    print(f"  {vio_type}能量违规：{reason} | 值: {val} / 参考: {ref}")
            # 保存断点（每批更新）
            current_idx = start_idx + idx + 1
            if current_idx % BATCH_SIZE == 0:
                save_checkpoint(n, current_idx)
                print(f"[n={n}] 进度：{current_idx}/{len(graphs)} | 违规数：{len(violations)}")
    else:
        with mp.Pool(processes=PARALLEL_PROCESSES) as pool:
            for idx in range(0, len(tasks), BATCH_SIZE):
                batch = tasks[idx:idx + BATCH_SIZE]
                try:
                    batch_res = pool.map_async(process_single_graph, batch).get(timeout=30)
                except mp.TimeoutError:
                    print(f"[n={n}] 批次 {idx} 超时，跳过")
                    continue
                # 收集违规
                for res in batch_res:
                    if res:
                        G, vio_list, eigvals = res
                        violations.append((G, vio_list, eigvals))
                        print(f"\n[n={n}] 违规图 {G.edges()}:")
                        print(f"  特征值: {np.round(eigvals, 4)}")
                        for vio_type, reason, val, ref in vio_list:
                            print(f"  {vio_type}能量违规：{reason} | 值: {val} / 参考: {ref}")
                # 保存断点
                current_idx = start_idx + idx + BATCH_SIZE
                save_checkpoint(n, current_idx)
                print(f"[n={n}] 进度：{min(current_idx, len(graphs))}/{len(graphs)} | 违规数：{len(violations)}")

    # 清理当前n的断点
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    return violations


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("===== 图立方能量验证（正+负）=====")
    print(f"配置：验证类型={VERIFY_TYPE} | 并行进程={PARALLEL_PROCESSES} | 批次大小={BATCH_SIZE}")
    all_violations = defaultdict(list)

    # 验证n=3~10
    for n in range(3, 11):
        print("\n=====================================")
        print(f"开始验证 n = {n}")
        print("=====================================")
        try:
            violations = verify_graph_energy(n)
            if violations:
                all_violations[n] = violations
                print(f"\n[n={n}] 完成：发现 {len(violations)} 个违规图")
            else:
                print(f"\n[n={n}] 完成：所有图满足条件")
        except Exception as e:
            print(f"\n[n={n}] 验证失败：{e}")
            continue

    # 最终结果汇总
    print("\n=====================================")
    print("============= 最终结果 ===============")
    if all_violations:
        for n, vio_list in all_violations.items():
            print(f"\n🔴 n={n} 违规图：")
            for idx, (G, vio_details, eigvals) in enumerate(vio_list, 1):
                print(f"  {idx}. 边列表：{list(G.edges())}")
                for vio_type, reason, val, ref in vio_details:
                    print(f"     {vio_type}能量：{reason} | {val} / {ref}")
    else:
        print("\n✅ 所有n=3~10的图均满足正/负立方能量条件！")