import networkx as nx
import numpy as np
from collections import defaultdict
import os
import multiprocessing as mp
from networkx.generators.atlas import graph_atlas_g
import pickle
import time
import signal

# ===================== CORE CONFIGURATION (ONLY EDIT HERE!) =====================
# macOS username (check via terminal command: whoami)
MAC_USERNAME = "czb"

# G6 file paths (n=9/10, files placed on Desktop)
G6_FILE_PATHS = {
    9: f"/Users/{MAC_USERNAME}/Desktop/graphs_9.g6",
    10: f"/Users/{MAC_USERNAME}/Desktop/graphs_10.g6"
}

# Cache file paths (avoid repeated parsing of g6 files)
CACHE_FILE_PATHS = {
    n: f"/Users/{MAC_USERNAME}/Desktop/graph_cache_{n}.pkl" for n in range(3, 11)
}

# Parallel configuration (key to avoid freezing: 1 for 8GB RAM, 2 for 16GB)
PARALLEL_PROCESSES = 1  # Prefer single process to avoid deadlock
BATCH_SIZE = 100        # Small batches to prevent memory overflow
CALC_TIMEOUT = 5        # Eigenvalue computation timeout (seconds)

# Verification type: "positive", "negative", or "both"
VERIFY_TYPE = "both"


# ===================== MEMORY CACHE + CHECKPOINT =====================
GRAPH_MEM_CACHE = {}  # In-memory cache per run
CHECKPOINT_FILE = f"/Users/{MAC_USERNAME}/Desktop/verify_checkpoint.txt"


# ===================== UTILITY FUNCTIONS (ENERGY + BASIC CHECKS) =====================
def adjacency_matrix(graph, n):
    """Generate n x n adjacency matrix"""
    return nx.to_numpy_array(graph, nodelist=range(n))


def positive_cubic_sum(eigenvalues):
    """Sum of cubes of positive eigenvalues (E3_plus)"""
    return sum(ev ** 3 for ev in eigenvalues if ev > 0)


def absolute_cubic_sum(eigenvalues):
    """Sum of cubes of absolute values of negative eigenvalues (E3_minus)"""
    return sum(abs(ev) ** 3 for ev in eigenvalues if ev < 0)


def is_path_graph(G):
    """Check whether G is a path graph P_n"""
    return nx.is_isomorphic(G, nx.path_graph(G.number_of_nodes()))


def is_complete_graph(G, n):
    """Check whether G is a complete graph K_n"""
    return G.number_of_edges() == n * (n - 1) // 2


# ===================== CACHE + CHECKPOINT UTILITIES =====================
def save_graph_cache(n, graphs):
    """Save graph list to disk cache"""
    cache_file = CACHE_FILE_PATHS[n]
    try:
        g6_mtime = os.path.getmtime(G6_FILE_PATHS[n]) if n in [9, 10] else time.time()
        with open(cache_file, 'wb') as f:
            pickle.dump({"graphs": graphs, "g6_mtime": g6_mtime}, f)
        print(f"[n={n}] Cache saved: {cache_file}")
    except Exception as e:
        print(f"[n={n}] Failed to save cache: {e}")


def load_graph_cache(n):
    """Load graph list from disk cache (with validation)"""
    cache_file = CACHE_FILE_PATHS[n]
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        if n in [9, 10] and cache_data["g6_mtime"] != os.path.getmtime(G6_FILE_PATHS[n]):
            os.remove(cache_file)
            return None
        print(f"[n={n}] Loaded from cache: {len(cache_data['graphs'])} graphs")
        return cache_data["graphs"]
    except Exception as e:
        print(f"[n={n}] Failed to load cache: {e}")
        return None


def save_checkpoint(n, processed_idx):
    """Save checkpoint (number of processed graphs)"""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(f"{n},{processed_idx}")


def load_checkpoint(n):
    """Load checkpoint"""
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            line = f.read().strip()
            if line:
                ckpt_n, ckpt_idx = line.split(",")
                if int(ckpt_n) == n:
                    print(f"[n={n}] Resume from checkpoint {ckpt_idx}")
                    return int(ckpt_idx)
        return 0
    except Exception as e:
        print(f"[n={n}] Failed to load checkpoint: {e}")
        return 0


# ===================== GRAPH LOADING (n=3~10) =====================
def load_graphs(n):
    """Load connected graphs on n vertices (atlas for n<=8, g6 for n>=9)"""
    if n in GRAPH_MEM_CACHE:
        return GRAPH_MEM_CACHE[n]

    cache_graphs = load_graph_cache(n)
    if cache_graphs:
        GRAPH_MEM_CACHE[n] = cache_graphs
        return cache_graphs

    graphs = []
    if 3 <= n <= 8:
        atlas = graph_atlas_g()
        graphs = [G for G in atlas if G.number_of_nodes() == n and nx.is_connected(G)]
        print(f"[n={n}] Loaded from atlas: {len(graphs)} graphs")
    elif n in [9, 10]:
        g6_file = G6_FILE_PATHS[n]
        if not os.path.exists(g6_file):
            print(f"\n⚠️ [n={n}] G6 file not found: {g6_file}")
            print(f"   Run in terminal: cd ~/Desktop && geng -c {n} > graphs_{n}.g6")
            return []

        with open(g6_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        total = len(lines)
        print(f"[n={n}] Parsing g6 file ({total} graphs total)")
        for idx, line in enumerate(lines):
            try:
                G = nx.from_graph6_bytes(line.encode('utf-8'))
                if G.number_of_nodes() == n:
                    graphs.append(G)
                if idx % 50000 == 0 and idx > 0:
                    print(f"[n={n}] Parsing progress: {idx}/{total}")
            except:
                continue
        print(f"[n={n}] Parsing complete: {len(graphs)} graphs")
        save_graph_cache(n, graphs)

    GRAPH_MEM_CACHE[n] = graphs
    return graphs


# ===================== SINGLE GRAPH PROCESSING (WITH TIMEOUT) =====================
def timeout_handler(signum, frame):
    raise TimeoutError("Eigenvalue computation timeout")


def process_single_graph(args):
    """Verify a single graph (positive/negative cubic energy)"""
    G, n, verify_type, E3_Pn_pos, bound_pos, E3_Kn_neg = args
    try:
        if verify_type in ["negative", "both"] and (is_complete_graph(G, n) or (n == 3 and is_path_graph(G))):
            return None

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(CALC_TIMEOUT)
        A_G = adjacency_matrix(G, n)
        eigvals_G = np.linalg.eigh(A_G)[0]
        signal.alarm(0)

        violations = []

        if verify_type in ["positive", "both"]:
            E3_G_pos = positive_cubic_sum(eigvals_G)
            if E3_G_pos + 1e-8 < E3_Pn_pos:
                violations.append(("positive", "E3_plus(G) < E3_plus(P_n)", round(E3_G_pos, 4), round(E3_Pn_pos, 4)))
            if n > 3 and not is_path_graph(G) and E3_G_pos + 1e-8 < bound_pos:
                violations.append(("positive", "E3_plus(G) < (sqrt(5)/2)*n", round(E3_G_pos, 4), round(bound_pos, 4)))

        if verify_type in ["negative", "both"]:
            E3_G_neg = absolute_cubic_sum(eigvals_G)
            if E3_G_neg < E3_Kn_neg or E3_G_neg < n:
                violations.append(("negative", "E3_minus(G) < n or < E3_minus(K_n)",
                                   round(E3_G_neg, 4), round(E3_Kn_neg, 4)))

        if violations:
            return (G, violations, eigvals_G)
        return None

    except TimeoutError:
        print(f"[n={n}] Graph {list(G.edges())} timed out, skipped")
        return None
    except Exception as e:
        print(f"[n={n}] Graph {list(G.edges())} error: {e}, skipped")
        return None


# ===================== MAIN VERIFICATION FUNCTION =====================
def verify_graph_energy(n):
    """Verify cubic energy conditions for graphs on n vertices"""
    graphs = load_graphs(n)
    if not graphs:
        return []

    Pn = nx.path_graph(n)
    eigvals_Pn = np.linalg.eigh(adjacency_matrix(Pn, n))[0]
    E3_Pn_pos = positive_cubic_sum(eigvals_Pn)
    bound_pos = (np.sqrt(5) / 2) * n
    E3_Kn_neg = n - 1

    print(f"\n[n={n}] Verification parameters:")
    if VERIFY_TYPE in ["positive", "both"]:
        print(f"  Positive energy (P_n): {round(E3_Pn_pos, 4)} | Lower bound: {round(bound_pos, 4)}")
    if VERIFY_TYPE in ["negative", "both"]:
        print(f"  Negative energy (K_n): {round(E3_Kn_neg, 4)} | Lower bound: {n}")

    start_idx = load_checkpoint(n)
    tasks = [(G, n, VERIFY_TYPE, E3_Pn_pos, bound_pos, E3_Kn_neg) for G in graphs[start_idx:]]
    violations = []

    for idx, task in enumerate(tasks):
        res = process_single_graph(task)
        if res:
            G, vio_list, eigvals = res
            violations.append((G, vio_list, eigvals))
            print(f"\n[n={n}] Violating graph {G.edges()}:")
            print(f"  Eigenvalues: {np.round(eigvals, 4)}")
            for t, r, v, ref in vio_list:
                print(f"  {t} violation: {r} | {v} / {ref}")

        current_idx = start_idx + idx + 1
        if current_idx % BATCH_SIZE == 0:
            save_checkpoint(n, current_idx)
            print(f"[n={n}] Progress: {current_idx}/{len(graphs)} | Violations: {len(violations)}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    return violations


# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    print("===== Cubic Energy Verification (Positive + Negative) =====")
    print(f"Config: VERIFY_TYPE={VERIFY_TYPE} | PROCESSES={PARALLEL_PROCESSES} | BATCH={BATCH_SIZE}")

    all_violations = defaultdict(list)

    for n in range(3, 11):
        print("\n=====================================")
        print(f"Start verification for n = {n}")
        print("=====================================")
        try:
            violations = verify_graph_energy(n)
            if violations:
                all_violations[n] = violations
                print(f"\n[n={n}] Done: {len(violations)} violating graphs found")
            else:
                print(f"\n[n={n}] Done: all graphs satisfy the conditions")
        except Exception as e:
            print(f"\n[n={n}] Verification failed: {e}")
            continue

    print("\n=====================================")
    print("============== FINAL RESULT ===============")
    if all_violations:
        for n, vio_list in all_violations.items():
            print(f"\n🔴 n={n} Violating graphs:")
            for idx, (G, vio_details, eigvals) in enumerate(vio_list, 1):
                print(f"  {idx}. Edge list: {list(G.edges())}")
                for t, r, v, ref in vio_details:
                    print(f"     {t} energy: {r} | {v} / {ref}")
    else:
        print("\n✅ All graphs with n=3~10 satisfy the positive/negative cubic energy conditions!")
