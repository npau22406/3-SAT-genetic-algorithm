from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from cse545_final_project import (
    ThreeSATInstance,
    GAConfig,
    generate_random_3sat_instance,
    load_3sat_instance,
    run_ga,
    wisdom_of_crowds_ga,
)

# Prints a clause like "(x1 ∨ ¬x3 ∨ x4)"
def clause_to_str(clause: List[Tuple[int, bool]]) -> str:
    pretty_bits: List[str] = []
    for (idx, is_neg) in clause:
        # Variables are x1..x_n for humans, but stored 0..n-1
        name = f"x{idx + 1}"
        if is_neg:
            name = f"¬{name}"
        pretty_bits.append(name)
    return "(" + " ∨ ".join(pretty_bits) + ")"

# builds a synthetic population matrix for visualization
def build_display_population(
    best_chromosome: List[int] | None,
    num_vars: int,
    pop_size: int,
    num_rows: int = 20,
) -> np.ndarray:
    rows = []
    rng = random.Random(1234)

    if best_chromosome is not None and len(best_chromosome) == num_vars:
        base = best_chromosome[:]
    else:
        base = [rng.randint(0, 1) for _ in range(num_vars)]

    rows.append(base)
    k = min(pop_size, num_rows)
    for _ in range(k - 1):
        mutant = base[:]
        # flip ~10% of bits (at least 1)
        num_flips = max(1, num_vars // 10)
        flip_indices = rng.sample(range(num_vars), num_flips)
        for i in flip_indices:
            mutant[i] = 1 - mutant[i]
        rows.append(mutant)

    grid = np.array(rows, dtype=int)
    return grid

# show formula details and a few sample clauses
def show_formula_panel(instance: ThreeSATInstance | None, max_clauses: int = 10) -> None:
    st.subheader("Formula Display")

    if instance is None:
        st.info("No formula loaded yet. Use **Load formula** or **Generate random**.")
        return

    n = instance.num_vars
    m = len(instance.clauses)

    st.write(f"**n = {n} variables**, **m = {m} clauses**")

    if m == 0:
        st.write("_This instance has no clauses._")
        return

    st.markdown("**Sample clauses:**")
    buf_lines: List[str] = []
    for i, clause in enumerate(instance.clauses[:max_clauses], start=1):
        buf_lines.append(f"**C{i}**: {clause_to_str(clause)}")
    st.markdown("\n".join(buf_lines))

# show binary population as a color-coded matrix
def show_population_panel(grid: np.ndarray | None) -> None:
    st.subheader("Population Panel – True / False Genes")

    if grid is None or grid.size == 0:
        st.info(
            "Population not available yet. Run **GA** or **WoC** to construct the visualization."
        )
        return

    num_rows, num_vars = grid.shape

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    im = ax.imshow(grid, aspect="auto", interpolation="nearest", cmap="RdYlGn")

    ax.set_xlabel("Variable index")
    ax.set_ylabel("Individual (visualized row)")
    ax.set_title("1 = True (green), 0 = False (red/yellow scale)")

    ax.set_xticks(
        np.linspace(0, num_vars - 1, min(10, num_vars), dtype=int)
    )
    ax.set_yticks(range(num_rows))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

# show fitness vs generation plot with slider
def show_progress_panel(
    avg_fitness_per_gen: List[float] | None,
    solution_gen: int | None,
    label: str,
    key_prefix: str,
) -> None:
    st.subheader(f"Progress Plot – {label}")

    if not avg_fitness_per_gen:
        st.info("No fitness history recorded yet.")
        return

    gens = list(range(len(avg_fitness_per_gen)))
    max_gen = len(gens) - 1

    # "Live" playback via slider that controls how many generations to show
    current_gen = st.slider(
        "Generation to display",
        min_value=0,
        max_value=max_gen,
        value=max_gen,
        key=f"{key_prefix}_gen_slider",
    )

    fig, ax = plt.subplots()
    ax.plot(gens[: current_gen + 1], avg_fitness_per_gen[: current_gen + 1], label="Average fitness")

    if solution_gen is not None and solution_gen <= max_gen:
        ax.axvline(solution_gen, linestyle="--", label=f"Solution found (gen {solution_gen+1})")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Average fitness")
    ax.set_title(f"Fitness vs Generation ({label})")
    ax.legend()
    st.pyplot(fig)

#show a small table summarizing WoC crowd status
def show_crowd_status_panel(
    best_fit: float | None,
    best_sat: int | None,
    num_clauses: int | None,
    woc_kwargs: Dict[str, Any] | None,
    stats: Dict[str, Any] | None,
) -> None:
    st.subheader("Crowd Status – Wisdom of Crowds Run")

    if best_fit is None or best_sat is None or num_clauses is None:
        st.info("Run **WoC** to see crowd status.")
        return

    frac = best_sat / num_clauses if num_clauses > 0 else 0.0
    sol_gen = None
    if stats is not None:
        sol_gen = stats.get("solution_generation")

    rows = [
        {
            "Metric": "Best fitness",
            "Value": f"{best_fit:.4f}",
        },
        {
            "Metric": "Best satisfied clauses",
            "Value": f"{best_sat} / {num_clauses} ({frac:.3f})",
        },
    ]
    if sol_gen is not None:
        rows.append(
            {
                "Metric": "Generation solution found",
                "Value": sol_gen + 1,
            }
        )

    st.table(rows)

    if woc_kwargs:
        with st.expander("WoC configuration (sub-GA crowd settings)"):
            st.json(woc_kwargs)

#load all JSON experiment logs from a directory
def load_experiment_logs(log_dir: str = "experiment_logs") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    p = Path(log_dir)
    if not p.exists():
        return out

    for file in sorted(p.glob("*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            out[file.name] = data
        except Exception as e:
            print(f"Failed to read {file}: {e}")
    return out

# main panel for browsing and plotting experiment logs
def show_experiment_logs_panel() -> None:
    st.header("Experiment Logs – Results & Graphs")

    logs = load_experiment_logs()
    if not logs:
        st.info(
            "No experiment logs found. Make sure JSON logs are in an `experiment_logs/` folder.\n\n"
            "These should be the files produced by your GA/WoC experiments "
            "(e.g., Baseline_small.json, WoC_K5_small.json, etc.)."
        )
        return

   # overall summary table across all experiments
    summary_rows = []
    comp_names = []
    comp_success = []
    comp_avg_gen = []
    comp_avg_best = []

    for fname, data in logs.items():
        summary = data.get("summary", {})
        exp_name = data.get("experiment_name", fname)
        num_runs = data.get("num_runs")

        success_rate = summary.get("success_rate")
        avg_best = summary.get("avg_best_fitness")
        avg_sol_gen = summary.get("avg_solution_generation")

        # crude label from filename
        if "baseline" in fname.lower():
            kind = "Baseline GA"
        elif "woc" in fname.lower():
            kind = "Wisdom of Crowds"
        else:
            kind = "Other"

        summary_rows.append(
            {
                "File": fname,
                "Experiment": exp_name,
                "Type": kind,
                "Runs": num_runs,
                "Success rate (%)": success_rate,
                "Avg best fitness": avg_best,
                "Avg solution gen": avg_sol_gen,
            }
        )

        if success_rate is not None:
            comp_names.append(exp_name)
            comp_success.append(success_rate)
        if avg_sol_gen is not None:
            comp_avg_gen.append(avg_sol_gen)
        else:
            comp_avg_gen.append(None)
        if avg_best is not None:
            comp_avg_best.append(avg_best)
        else:
            comp_avg_best.append(None)

    st.subheader("Summary of all experiments")
    st.dataframe(summary_rows, use_container_width=True)

    # Comparison graphs across experiments 
    if comp_names:
        st.subheader("Compare experiments (Baseline vs WoC, etc.)")

        colA, colB = st.columns(2)

        # 1) Success rate per experiment (left)
        with colA:
            figA, axA = plt.subplots(figsize=(5, 3))
            x = np.arange(len(comp_names))
            axA.bar(x, comp_success, color="#4C72B0")
            axA.set_xticks(x)
            axA.set_xticklabels(comp_names, rotation=45, ha="right", fontsize=8)
            axA.set_ylabel("Success rate (%)", fontsize=9)
            axA.set_title("Success rate by experiment", fontsize=10)
            axA.tick_params(axis="y", labelsize=8)
            st.pyplot(figA, use_container_width=False)

        # 2) Avg solution generation per experiment (right)
        valid_idx = [i for i, g in enumerate(comp_avg_gen) if g is not None]
        if valid_idx:
            with colB:
                figB, axB = plt.subplots(figsize=(5, 3))
                x2 = np.arange(len(valid_idx))
                names2 = [comp_names[i] for i in valid_idx]
                gens2 = [comp_avg_gen[i] for i in valid_idx]
                axB.bar(x2, gens2, color="#55A868")
                axB.set_xticks(x2)
                axB.set_xticklabels(names2, rotation=45, ha="right", fontsize=8)
                axB.set_ylabel("Avg solution gen", fontsize=9)
                axB.set_title("Avg solution generation", fontsize=10)
                axB.tick_params(axis="y", labelsize=8)
                st.pyplot(figB, use_container_width=False)

    st.markdown("---")

    # average best fitness vs instance size for baseline experiments
    if any("Baseline" in n or "baseline" in n for n in comp_names):
        st.subheader("Figure 1 – Average Best Fitness vs. Instance Size")

        # Extract baseline-only experiments in size order
        size_labels = []
        avg_best_vals = []
        for fname, data in logs.items():
            if "baseline" in fname.lower():
                size = None
                if "small" in fname.lower():
                    size = "Small (300 clauses)"
                elif "medium" in fname.lower():
                    size = "Medium (400 clauses)"
                elif "large" in fname.lower():
                    size = "Large (500 clauses)"
                if size:
                    size_labels.append(size)
                    avg_best_vals.append(data["summary"]["avg_best_fitness"])

        # Sort by instance size order
        order = ["Small (300 clauses)", "Medium (400 clauses)", "Large (500 clauses)"]
        paired = [(s, v) for s, v in zip(size_labels, avg_best_vals) if s in order]
        paired.sort(key=lambda x: order.index(x[0]))
        labels, values = zip(*paired) if paired else ([], [])

        if labels:
            figF1, axF1 = plt.subplots(figsize=(6, 4))
            axF1.plot(labels, values, marker="o", linewidth=2, color="#4C72B0")
            axF1.set_title("Average Best Fitness vs. Instance Size")
            axF1.set_xlabel("Instance Size")
            axF1.set_ylabel("Average Best Fitness")
            axF1.set_ylim(0.95, 1.1)
            axF1.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(figF1)


    # detailed view for a single selected experiment
    st.subheader("Single experiment – per-run results & graphs")

    choices = list(logs.keys())
    selected = st.selectbox("Select a log file", choices, index=0)
    data = logs[selected]

    exp_name = data.get("experiment_name", selected)
    summary = data.get("summary", {})
    num_runs = data.get("num_runs", summary.get("num_runs", None))

    st.markdown(f"**Experiment:** `{exp_name}`")

    col_a, col_b, col_c = st.columns(3)
    if num_runs is not None:
        col_a.metric("Runs", num_runs)
    sr = summary.get("success_rate")
    if sr is not None:
        col_b.metric("Success rate (%)", f"{sr:.1f}")
    ab = summary.get("avg_best_fitness")
    if ab is not None:
        col_c.metric("Avg best fitness", f"{ab:.4f}")

    # Per-run table
    results = data.get("results", {})
    rows = []
    best_vals = []
    sol_gens = []
    solved_flags = []

    for run_key, run_data in results.items():
        best = run_data.get("best_fitness")
        sat = run_data.get("satisfied")
        total = run_data.get("total_clauses")
        solved = run_data.get("solved")
        sol_gen = run_data.get("solution_generation")

        best_vals.append(best)
        sol_gens.append(sol_gen)
        solved_flags.append(bool(solved))

        rows.append(
            {
                "Run": run_key,
                "Best fitness": best,
                "Satisfied / total": f"{sat}/{total}",
                "Solved": solved,
                "Solution generation": sol_gen,
            }
        )

    st.dataframe(rows, use_container_width=True)

    # Graphs for the selected experiment
    if best_vals:
        st.markdown("### Graphs for this experiment")

        col1, col2 = st.columns(2)

        # Best fitness per run
        with col1:
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            x3 = np.arange(1, len(best_vals) + 1)
            ax3.plot(x3, best_vals, marker="o", color="#4C72B0")
            ax3.set_xlabel("Run", fontsize=9)
            ax3.set_ylabel("Best fitness", fontsize=9)
            ax3.set_title("Best fitness per run", fontsize=10)
            ax3.tick_params(axis="both", labelsize=8)
            st.pyplot(fig3, use_container_width=False)

        # Histogram of solution generations (right)
        with col2:
            solved_gens = [g for g, s in zip(sol_gens, solved_flags) if s and g is not None]
            if solved_gens:
                fig4, ax4 = plt.subplots(figsize=(5, 3))
                ax4.hist(solved_gens, bins=min(10, len(set(solved_gens))), color="#55A868")
                ax4.set_xlabel("Generation", fontsize=9)
                ax4.set_ylabel("Count", fontsize=9)
                ax4.set_title("Solution generation (solved runs)", fontsize=10)
                ax4.tick_params(axis="both", labelsize=8)
                st.pyplot(fig4, use_container_width=False)

        # Solved vs unsolved (below)
        fig5, ax5 = plt.subplots(figsize=(4.5, 2.5))
        num_solved = sum(solved_flags)
        num_unsolved = len(solved_flags) - num_solved
        ax5.bar(["Solved", "Unsolved"], [num_solved, num_unsolved], color=["#55A868", "#C44E52"])
        ax5.set_ylabel("Runs", fontsize=9)
        ax5.set_title("Solved vs Unsolved", fontsize=10)
        ax5.tick_params(axis="both", labelsize=8)
        st.pyplot(fig5, use_container_width=False)

# Initialize Streamlit session state keys
def init_state() -> None:
    if "instance" not in st.session_state:
        st.session_state.instance = None

    if "ga_result" not in st.session_state:
        st.session_state.ga_result = None  # dict with keys: best_chrom, best_fit, best_sat, stats, config

    if "woc_result" not in st.session_state:
        st.session_state.woc_result = None  # dict with keys: best_chrom, best_fit, best_sat, stats, woc_kwargs, num_clauses

    if "run_mode" not in st.session_state:
        st.session_state.run_mode = None  # "GA" or "WoC" or None

# top-level entry: tabs for demo and logs
def main() -> None:
    st.set_page_config(page_title="3-SAT GA / WoC Visualizer", layout="wide")
    st.title("3-SAT Genetic Algorithm + Wisdom of Artificial Crowds Visualizer")

    init_state()

    # Tabs: main interactive demo vs. experiment logs
    tab_demo, tab_logs = st.tabs(["Interactive GA / WoC demo", "Experiment logs"])

    with tab_demo:
        sidebar_and_demo()

    with tab_logs:
        show_experiment_logs_panel()

# Sidebar controls and main demo layout
def sidebar_and_demo() -> None:
    st.sidebar.header("GA configuration")

    pop_size = st.sidebar.number_input("Population size", 10, 1000, 200, step=10)
    num_generations = st.sidebar.number_input("Generations", 10, 2000, 500, step=10)
    crossover_rate = st.sidebar.slider("Crossover rate", 0.0, 1.0, 0.7, 0.05)
    mutation_prob = st.sidebar.slider("Mutation probability", 0.0, 0.2, 0.02, 0.005)
    tournament_size = st.sidebar.number_input("Tournament size", 2, 20, 3)
    elitism_k = st.sidebar.number_input("Elitism k", 0, 10, 2)
    alpha_bonus = st.sidebar.number_input("Alpha bonus (for fully satisfied)", 0.0, 5.0, 0.1, 0.1)
    random_seed = st.sidebar.number_input("Random seed", 0, 10_000, 42)

    ga_config = GAConfig(
        pop_size=int(pop_size),
        num_generations=int(num_generations),
        crossover_rate=float(crossover_rate),
        mutation_prob=float(mutation_prob),
        tournament_size=int(tournament_size),
        elitism_k=int(elitism_k),
        alpha_bonus=float(alpha_bonus),
        random_seed=int(random_seed),
    )

    st.sidebar.markdown("---")
    st.sidebar.header("WoC extra parameters")

    num_subpops = st.sidebar.number_input("Number of sub-populations", 2, 20, 5)
    top_k_per_subga = st.sidebar.number_input("Top-k elites per sub-GA", 1, 50, 5)
    inject_rate = st.sidebar.slider("Inject rate (fraction replaced by wisdom)", 0.05, 0.8, 0.2, 0.05)
    wisdom_mut_prob = st.sidebar.slider("Wisdom mutation probability", 0.0, 0.3, 0.05, 0.01)
    aggregation_interval = st.sidebar.number_input("Aggregation interval (generations)", 1, 200, 20)
    stagnation_limit = st.sidebar.number_input("Stagnation limit (epochs)", 1, 50, 5)
    weighted_wisdom = st.sidebar.checkbox("Use weighted wisdom", value=False)

    woc_kwargs = dict(
        num_subpops=int(num_subpops),
        top_k_per_subga=int(top_k_per_subga),
        inject_rate=float(inject_rate),
        wisdom_mut_prob=float(wisdom_mut_prob),
        aggregation_interval=int(aggregation_interval),
        stagnation_limit=int(stagnation_limit),
        weighted_wisdom=bool(weighted_wisdom),
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Load / generate formula")

    built_in_choice = st.sidebar.selectbox(
        "Built-in instance",
        ("None", "3sat_instance_small.json", "3sat_instance_medium.json", "3sat_instance_large.json"),
    )
    uploaded_file = st.sidebar.file_uploader("Or upload 3-SAT JSON", type=["json"])

    # Top control row 
    st.markdown("### Controls")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        btn_load = st.button("Load formula")
    with c2:
        btn_random = st.button("Generate random")
    with c3:
        btn_run_ga = st.button("Run GA")
    with c4:
        btn_run_woc = st.button("Run WoC")
    with c5:
        btn_pause = st.button("Pause animation")
    with c6:
        btn_reset = st.button("Reset")

    # Control logic  
    if btn_load:
        instance = None
        if uploaded_file is not None:
            # Uploaded JSON in the same format as 3SAT instances
            try:
                data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                # Minimal validation
                num_vars = data["num_vars"]
                clauses = data["clauses"]
                instance = ThreeSATInstance()
                instance.num_vars = num_vars
                instance.clauses = clauses
                st.success(f"Loaded uploaded instance: n={num_vars}, m={len(clauses)}")
            except Exception as e:
                st.error(f"Failed to parse uploaded JSON: {e}")

        elif built_in_choice != "None":
            path = Path(built_in_choice)
            if not path.exists():
                st.warning(
                    f"Built-in file `{built_in_choice}` not found in working directory.\n"
                    "Make sure the JSON files are located next to this app."
                )
            else:
                instance = load_3sat_instance(path)
                st.success(
                    f"Loaded {built_in_choice}: n={instance.num_vars}, m={len(instance.clauses)}"
                )
        else:
            st.warning("Select a built-in instance or upload a JSON file first.")

        if instance is not None:
            st.session_state.instance = instance
            st.session_state.ga_result = None
            st.session_state.woc_result = None
            st.session_state.run_mode = None

    if btn_random:
        instance = generate_random_3sat_instance(num_vars=50, num_clauses=200)
        st.session_state.instance = instance
        st.session_state.ga_result = None
        st.session_state.woc_result = None
        st.session_state.run_mode = None
        st.success(f"Generated random instance: n={instance.num_vars}, m={len(instance.clauses)}")

    if btn_run_ga:
        inst = st.session_state.instance
        if inst is None:
            st.warning("Load or generate a formula first.")
        else:
            stats: Dict[str, Any] = {}
            best_chrom, best_fit, best_sat = run_ga(
                instance=inst,
                config=ga_config,
                verbose=False,
                stats=stats,
            )
            st.session_state.ga_result = dict(
                best_chrom=best_chrom,
                best_fit=best_fit,
                best_sat=best_sat,
                stats=stats,
                config=ga_config,
            )
            st.session_state.woc_result = None
            st.session_state.run_mode = "GA"
            st.success(f"GA run finished. Best fitness = {best_fit:.4f}, satisfied = {best_sat}/{len(inst.clauses)}")

    if btn_run_woc:
        inst = st.session_state.instance
        if inst is None:
            st.warning("Load or generate a formula first.")
        else:
            stats: Dict[str, Any] = {}
            best_chrom, best_fit, best_sat = wisdom_of_crowds_ga(
                instance=inst,
                base_config=ga_config,
                verbose=False,
                stats=stats,
                **woc_kwargs,
            )
            st.session_state.woc_result = dict(
                best_chrom=best_chrom,
                best_fit=best_fit,
                best_sat=best_sat,
                stats=stats,
                woc_kwargs=woc_kwargs,
                num_clauses=len(inst.clauses),
            )
            st.session_state.ga_result = None
            st.session_state.run_mode = "WoC"
            st.success(
                f"WoC run finished. Best fitness = {best_fit:.4f}, satisfied = {best_sat}/{len(inst.clauses)}"
            )

    if btn_pause:
        st.info("Use the generation slider below the progress plot to pause at a specific generation.")

    if btn_reset:
        st.session_state.instance = None
        st.session_state.ga_result = None
        st.session_state.woc_result = None
        st.session_state.run_mode = None
        st.success("State reset. Load or generate a new formula to begin.")

    # Layout for panels 
    col_left, col_right = st.columns([1.3, 1.0])

    with col_left:
        show_formula_panel(st.session_state.instance)

        st.markdown("---")

        # Build a display population matrix from whichever run is active
        inst = st.session_state.instance
        grid = None
        if inst is not None:
            num_vars = inst.num_vars
            if st.session_state.ga_result is not None:
                res = st.session_state.ga_result
                grid = build_display_population(
                    best_chromosome=res["best_chrom"],
                    num_vars=num_vars,
                    pop_size=ga_config.pop_size,
                )
            elif st.session_state.woc_result is not None:
                res = st.session_state.woc_result
                grid = build_display_population(
                    best_chromosome=res["best_chrom"],
                    num_vars=num_vars,
                    pop_size=ga_config.pop_size,
                )

        show_population_panel(grid)

    with col_right:
        # Progress plot and crowd status
        if st.session_state.ga_result is not None:
            stats = st.session_state.ga_result["stats"]
            avg_f = stats.get("avg_fitness_per_gen", [])
            sol_gen = stats.get("solution_generation")
            show_progress_panel(avg_f, sol_gen, label="Baseline GA", key_prefix="ga")
        elif st.session_state.woc_result is not None:
            stats = st.session_state.woc_result["stats"]
            avg_f = stats.get("avg_fitness_per_gen", [])
            sol_gen = stats.get("solution_generation")
            show_progress_panel(avg_f, sol_gen, label="Wisdom of Crowds GA", key_prefix="woc")
        else:
            st.subheader("Progress Plot")
            st.info("Run GA or WoC to see fitness over generations.")

        st.markdown("---")

        # Crowd status only relevant for WoC
        if st.session_state.woc_result is not None:
            res = st.session_state.woc_result
            show_crowd_status_panel(
                best_fit=res["best_fit"],
                best_sat=res["best_sat"],
                num_clauses=res["num_clauses"],
                woc_kwargs=res["woc_kwargs"],
                stats=res["stats"],
            )
        else:
            st.subheader("Crowd Status – Wisdom of Crowds Run")
            st.info("Run **WoC** to view crowd status information.")


if __name__ == "__main__":
    main()
