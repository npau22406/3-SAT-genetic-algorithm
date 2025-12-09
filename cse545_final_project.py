from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import List, Tuple, Sequence, Dict, Any
import random
import json
from pathlib import Path

# Save experiment results to a JSON file
def save_experiment_log(data: Dict[str, Any], filename: str):
    from pathlib import Path
    Path("experiment_logs").mkdir(exist_ok=True)
    filepath = Path("experiment_logs") / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f" Saved experiment log to {filepath.resolve()}")

Literal = Tuple[int, bool]      
Clause = Tuple[Literal, Literal, Literal]

# Represents a 3-SAT instance
@dataclass
class ThreeSATInstance:
    num_vars: int
    clauses: List[Clause]

    # Evaluate a chromosome and count satisfied clauses
    def evaluate_assignment(
        self,
        chromosome: Sequence[int],
    ) -> Tuple[int, float, bool]:
        
        if len(chromosome) != self.num_vars:
            raise ValueError(
                f"Chromosome length {len(chromosome)} "
                f"does not match num_vars {self.num_vars}"
            )

        satisfied = 0
        for clause in self.clauses:
            if self._clause_satisfied(clause, chromosome):
                satisfied += 1

        total = len(self.clauses)
        fraction = satisfied / total if total > 0 else 0.0
        return satisfied, fraction, (satisfied == total)
    
    # Return truth value of one literal under a chromosome
    @staticmethod
    def _literal_value(literal: Literal, chromosome: Sequence[int]) -> bool:
        var_index, is_negated = literal
        bit = chromosome[var_index]
        var_val = bool(bit)            # 1 -> True, 0 -> False
        return not var_val if is_negated else var_val

    # Return True if a clause is satisfied
    def _clause_satisfied(self, clause: Clause, chromosome: Sequence[int]) -> bool:
        l1, l2, l3 = clause
        return (
            self._literal_value(l1, chromosome) or
            self._literal_value(l2, chromosome) or
            self._literal_value(l3, chromosome)
        )

# Create a random bitstring of given length
def random_chromosome(num_vars: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(num_vars)]

# Convert Boolean assignment to bitstring
def assignment_to_chromosome(assignment: Sequence[bool]) -> List[int]:
    return [1 if val else 0 for val in assignment]

# Convert bitstring to Boolean assignment
def chromosome_to_assignment(chromosome: Sequence[int]) -> List[bool]:
    return [bool(bit) for bit in chromosome]

# Compute fitness of a chromosome
def fitness_for_chromosome(
    instance: ThreeSATInstance,
    chromosome: Sequence[int],
    alpha: float = 0.0,
) -> float:
    num_sat, frac_sat, all_sat = instance.evaluate_assignment(chromosome)

    # base term s/m is exactly frac_sat
    f = frac_sat

    # add bonus if fully satisfying
    if all_sat and alpha > 0.0:
        f += alpha

    return f


# Generate a random 3-SAT instance
def generate_random_3sat_instance(num_vars: int, num_clauses: int) -> ThreeSATInstance:
    clauses: List[Clause] = []

    for _ in range(num_clauses):
        # randomly select 3 distinct variables from 0..n-1
        vars_in_clause = random.sample(range(num_vars), 3)
        clause: Clause = tuple(
            (var_idx, bool(random.getrandbits(1))) for var_idx in vars_in_clause
        )
        clauses.append(clause)

    return ThreeSATInstance(num_vars=num_vars, clauses=clauses)

# Save a 3-SAT instance to JSON
def save_3sat_instance(instance: ThreeSATInstance, filepath: str | Path):
    data = {
        "num_vars": instance.num_vars,
        "clauses": [
            [[var_idx, is_neg] for (var_idx, is_neg) in clause]
            for clause in instance.clauses
        ],
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved 3-SAT instance to {filepath.resolve()}")

# Load a 3-SAT instance from JSON
def load_3sat_instance(filepath: str | Path) -> ThreeSATInstance:
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_vars = data["num_vars"]
    clauses = [tuple((v, bool(n)) for v, n in clause) for clause in data["clauses"]]

    return ThreeSATInstance(num_vars=num_vars, clauses=clauses)

# GA parameter configuration
@dataclass
class GAConfig:
    pop_size: int = 200
    num_generations: int = 500
    crossover_rate: float = 0.7
    mutation_prob: float = 0.01
    tournament_size: int = 3
    elitism_k: int = 2
    alpha_bonus: float = 0.1   # bonus for fully satisfying all clauses
    random_seed: int | None = 42

# Initialize random population
def init_population(num_vars: int, pop_size: int) -> List[List[int]]:
    population: List[List[int]] = []
    for _ in range(pop_size):
        chrom = [random.randint(0, 1) for _ in range(num_vars)]
        population.append(chrom)
    return population

# Tournament selection for parent choice
def tournament_selection(
    population: List[List[int]],
    fitnesses: Sequence[float],
    tournament_size: int,
) -> List[int]:
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(indices, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]  # return a copy

# One-point crossover operation
def one_point_crossover(
    parent1: Sequence[int],
    parent2: Sequence[int],
) -> Tuple[List[int], List[int]]:
    assert len(parent1) == len(parent2)
    n = len(parent1)
    if n < 2:
        # no sensible crossover for length 0 or 1
        return list(parent1), list(parent2)

    point = random.randint(1, n - 1)
    child1 = list(parent1[:point]) + list(parent2[point:])
    child2 = list(parent2[:point]) + list(parent1[point:])
    return child1, child2

# Flip bits with given mutation probability
def bit_flip_mutation(
    chromosome: List[int],
    mutation_prob: float,
) -> None:
    for i in range(len(chromosome)):
        if random.random() < mutation_prob:
            chromosome[i] = 1 - chromosome[i]  # flip 0 <-> 1

# Run baseline Genetic Algorithm
def run_ga(
    instance: ThreeSATInstance,
    config: GAConfig,
    verbose: bool = True,
    stats: Dict[str, Any] | None = None,
) -> Tuple[List[int], float, int]:
    if config.random_seed is not None:
        random.seed(config.random_seed)

    num_vars = instance.num_vars
    num_clauses = len(instance.clauses)

    # Initialize population
    population = init_population(num_vars, config.pop_size)

    best_chrom_overall: List[int] | None = None
    best_fit_overall: float = float("-inf")
    best_sat_overall: int = -1

    if stats is not None:
        stats["avg_fitness_per_gen"] = []
        stats["solution_generation"] = None

    for gen in range(config.num_generations):
        # Evaluate population
        fitnesses: List[float] = []
        satisfied_counts: List[int] = []

        for chrom in population:
            num_sat, frac_sat, all_sat = instance.evaluate_assignment(chrom)
            fit = fitness_for_chromosome(
                instance,
                chrom,
                alpha=config.alpha_bonus,
            )
            fitnesses.append(fit)
            satisfied_counts.append(num_sat)

        # average population fitness for this generation
        if stats is not None and fitnesses:
            avg_fit = sum(fitnesses) / len(fitnesses)
            stats["avg_fitness_per_gen"].append(avg_fit)

        # Track best individual this generation 
        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_sat = satisfied_counts[gen_best_idx]
        gen_best_chrom = population[gen_best_idx][:]

        if gen_best_fit > best_fit_overall:
            best_fit_overall = gen_best_fit
            best_sat_overall = gen_best_sat
            best_chrom_overall = gen_best_chrom

        if verbose and (gen % 10 == 0 or gen == config.num_generations - 1):
            print(
                f"Generation {gen:4d} | "
                f"Best fitness: {gen_best_fit:.4f} | "
                f"Best satisfied: {gen_best_sat}/{num_clauses}"
            )

        # If fully satisfied, capture solution generation (for stats)
        if gen_best_sat == num_clauses:
            if stats is not None and stats["solution_generation"] is None:
                stats["solution_generation"] = gen
            if verbose:
                print(
                    f"Full satisfaction reached at generation {gen} "
                    f"({gen_best_sat}/{num_clauses} clauses)."
                )
            break

        # Create next generation with elitism and GA operators
        # Elitism: copy top-k individuals directly
        elite_indices = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )[: config.elitism_k]

        next_population: List[List[int]] = [
            population[i][:] for i in elite_indices
        ]

        # Fill the rest of the population
        while len(next_population) < config.pop_size:
            # Select parents
            parent1 = tournament_selection(
                population, fitnesses, config.tournament_size
            )
            parent2 = tournament_selection(
                population, fitnesses, config.tournament_size
            )

            # Crossover
            if random.random() < config.crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutation
            bit_flip_mutation(child1, config.mutation_prob)
            bit_flip_mutation(child2, config.mutation_prob)

            next_population.append(child1)
            if len(next_population) < config.pop_size:
                next_population.append(child2)

        population = next_population

    # Safety: if best_chrom_overall is still None (should not happen), return something
    if best_chrom_overall is None:
        best_chrom_overall = population[0][:]
        num_sat, _, _ = instance.evaluate_assignment(best_chrom_overall)
        best_sat_overall = num_sat
        best_fit_overall = fitness_for_chromosome(
            instance,
            best_chrom_overall,
            alpha=config.alpha_bonus,
        )

    return best_chrom_overall, best_fit_overall, best_sat_overall

#
def wisdom_of_crowds_ga(
    instance: ThreeSATInstance,
    base_config: GAConfig,
    num_subpops: int = 5,
    top_k_per_subga: int = 5,
    inject_rate: float = 0.2,
    wisdom_mut_prob: float = 0.05,
    aggregation_interval: int = 20,
    stagnation_limit: int = 5,
    weighted_wisdom: bool = False,
    verbose: bool = True,
    stats: Dict[str, Any] | None = None,
) -> Tuple[List[int], float, int]:
    num_vars = instance.num_vars
    num_clauses = len(instance.clauses)

    # Initialize sub-populations and slightly-perturbed configs
    subpops: List[List[List[int]]] = []
    subconfigs: List[GAConfig] = []

    for k in range(num_subpops):
        cfg = GAConfig(
            pop_size=base_config.pop_size,
            num_generations=base_config.num_generations,
            crossover_rate=max(
                0.5,
                min(0.9, base_config.crossover_rate + random.uniform(-0.1, 0.1)),
            ),
            mutation_prob=max(
                0.005,
                min(0.05, base_config.mutation_prob + random.uniform(-0.01, 0.01)),
            ),
            tournament_size=base_config.tournament_size,
            elitism_k=base_config.elitism_k,
            alpha_bonus=base_config.alpha_bonus,
            random_seed=(base_config.random_seed or 42) + k,
        )
        random.seed(cfg.random_seed)
        subpops.append(init_population(num_vars, cfg.pop_size))
        subconfigs.append(cfg)

    best_chrom: List[int] | None = None
    best_fit: float = float("-inf")
    best_sat: int = -1

    if stats is not None:
        stats["avg_fitness_per_gen"] = []
        stats["solution_generation"] = None

    # For stagnation tracking at the epoch level
    stagnant_epochs = 0
    last_epoch_best_fit = float("-inf")
    max_epochs = max(1, base_config.num_generations // aggregation_interval)

    # Evolution loop over generations
    for gen in range(base_config.num_generations):
        all_fitnesses_this_gen: List[float] = []

        # Evolve each sub-population one generation
        for s in range(num_subpops):
            pop = subpops[s]
            cfg = subconfigs[s]

            # Evaluate population and track best
            fitnesses: List[float] = []
            satisfied_counts: List[int] = []
            for chrom in pop:
                num_sat, _, _ = instance.evaluate_assignment(chrom)
                fit = fitness_for_chromosome(instance, chrom, alpha=cfg.alpha_bonus)
                fitnesses.append(fit)
                satisfied_counts.append(num_sat)

                if fit > best_fit:
                    best_fit = fit
                    best_sat = num_sat
                    best_chrom = chrom[:]

            all_fitnesses_this_gen.extend(fitnesses)

            # Elitism + reproduction within this sub-population
            elite_indices = sorted(
                range(len(pop)),
                key=lambda i: fitnesses[i],
                reverse=True,
            )[: cfg.elitism_k]

            next_pop: List[List[int]] = [pop[i][:] for i in elite_indices]

            while len(next_pop) < cfg.pop_size:
                p1 = tournament_selection(pop, fitnesses, cfg.tournament_size)
                p2 = tournament_selection(pop, fitnesses, cfg.tournament_size)

                if random.random() < cfg.crossover_rate:
                    c1, c2 = one_point_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                bit_flip_mutation(c1, cfg.mutation_prob)
                bit_flip_mutation(c2, cfg.mutation_prob)

                next_pop.append(c1)
                if len(next_pop) < cfg.pop_size:
                    next_pop.append(c2)

            subpops[s] = next_pop

        # Record average population fitness across all subpops
        if stats is not None and all_fitnesses_this_gen:
            avg_fit = sum(all_fitnesses_this_gen) / len(all_fitnesses_this_gen)
            stats["avg_fitness_per_gen"].append(avg_fit)

        # Immediate stopping if all clauses are satisfied 
        if best_sat == num_clauses:
            if stats is not None and stats["solution_generation"] is None:
                stats["solution_generation"] = gen
            if verbose:
                print(f"✅ All clauses satisfied at generation {gen+1}.")
            break

        # Wisdom aggregation every 'aggregation_interval' generations 
        if (gen + 1) % aggregation_interval == 0:
            epoch = (gen + 1) // aggregation_interval

            # Collect top-k elites from each sub-population
            elite_pool: List[List[int]] = []
            elite_fitnesses: List[float] = []
            for s in range(num_subpops):
                pop = subpops[s]
                fits = [
                    fitness_for_chromosome(
                        instance, c, alpha=base_config.alpha_bonus
                    )
                    for c in pop
                ]
                top_idx = sorted(
                    range(len(pop)),
                    key=lambda i: fits[i],
                    reverse=True,
                )[: top_k_per_subga]
                for i in top_idx:
                    elite_pool.append(pop[i][:])
                    elite_fitnesses.append(fits[i])

            # Build wisdom chromosome
            wisdom: List[int] = []
            if weighted_wisdom:
                # Fitness-weighted vote per bit position
                total_weight = sum(elite_fitnesses) or 1.0
                for i in range(num_vars):
                    weighted_ones = sum(
                        w * ch[i] for w, ch in zip(elite_fitnesses, elite_pool)
                    )
                    wisdom_bit = 1 if weighted_ones > 0.5 * total_weight else 0
                    wisdom.append(wisdom_bit)
            else:
                # Simple majority vote per bit position
                for i in range(num_vars):
                    ones = sum(ch[i] for ch in elite_pool)
                    wisdom_bit = 1 if ones > len(elite_pool) / 2 else 0
                    wisdom.append(wisdom_bit)

            # Inject wisdom and mutated variants back into each sub-population
            for s in range(num_subpops):
                pop = subpops[s]
                num_inject = max(1, int(inject_rate * len(pop)))
                variants: List[List[int]] = [wisdom[:]]
                # Create additional mutated variants
                for _ in range(num_inject - 1):
                    w_mut = wisdom[:]
                    bit_flip_mutation(w_mut, wisdom_mut_prob)
                    variants.append(w_mut)

                replace_indices = random.sample(range(len(pop)), num_inject)
                for idx, w in zip(replace_indices, variants):
                    pop[idx] = w

            # Epoch-level logging and stopping criteria
            if verbose:
                mode = "Weighted" if weighted_wisdom else "Unweighted"
                print(
                    f"[Epoch {epoch}] ({mode} wisdom) "
                    f"best fitness={best_fit:.4f}, "
                    f"satisfied={best_sat}/{num_clauses}"
                )

            # All clauses satisfied 
            if best_sat == num_clauses or abs(best_fit - 1.0) < 1e-9:
                if stats is not None and stats["solution_generation"] is None:
                    stats["solution_generation"] = gen
                if verbose:
                    print(f"✅ All clauses satisfied at epoch {epoch} (gen {gen+1}).")
                break

            # Stagnation: no improvement in best fitness across epochs
            if best_fit <= last_epoch_best_fit + 1e-12:
                stagnant_epochs += 1
            else:
                stagnant_epochs = 0
                last_epoch_best_fit = best_fit

            if stagnant_epochs >= stagnation_limit:
                if verbose:
                    print(
                        f"⚠️ Stopped due to stagnation: "
                        f"{stagnant_epochs} consecutive epochs with no improvement."
                    )
                break

            # Max epochs reached
            if epoch >= max_epochs:
                if verbose:
                    print(f"⏹️ Reached maximum epochs ({max_epochs}).")
                break

    # Safety fallback
    if best_chrom is None:
        # If somehow nothing was set, pick a random chromosome from the first sub-pop
        best_chrom = subpops[0][0][:]
        best_sat, _, _ = instance.evaluate_assignment(best_chrom)
        best_fit = fitness_for_chromosome(
            instance, best_chrom, alpha=base_config.alpha_bonus
        )

    return best_chrom, best_fit, best_sat

# Benchmark GA on multiple instances
def benchmark_ga_on_instances(
    base_config: GAConfig,
    instance_prefix: str = "3sat_instance_",
    labels: Sequence[str] = ("small", "medium", "large"),
    verbose_each_run: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for label in labels:
        path = f"{instance_prefix}{label}.json"
        print(f"\n=== Running GA on {label.upper()} instance ===")
        print(f"Loading instance from: {path}")

        instance = load_3sat_instance(path)
        num_vars = instance.num_vars
        num_clauses = len(instance.clauses)
        print(f"  Variables: {num_vars}, Clauses: {num_clauses}")

        # Copy config so each run is independent
        config = GAConfig(
            pop_size=base_config.pop_size,
            num_generations=base_config.num_generations,
            crossover_rate=base_config.crossover_rate,
            mutation_prob=base_config.mutation_prob,
            tournament_size=base_config.tournament_size,
            elitism_k=base_config.elitism_k,
            alpha_bonus=base_config.alpha_bonus,
            random_seed=base_config.random_seed,
        )

        t0 = perf_counter()
        best_chrom, best_fit, best_sat = run_ga(
            instance,
            config,
            verbose=verbose_each_run,
        )
        t1 = perf_counter()
        elapsed = t1 - t0

        frac_sat = best_sat / num_clauses if num_clauses > 0 else 0.0

        summary = {
            "label": label,
            "num_vars": num_vars,
            "num_clauses": num_clauses,
            "best_fitness": best_fit,
            "best_satisfied": best_sat,
            "fraction_satisfied": frac_sat,
            "time_seconds": elapsed,
        }
        results.append(summary)

        print(f"  Done in {elapsed:.3f} s")
        print(f"  Best fitness: {best_fit:.4f}")
        print(f"  Best satisfied: {best_sat}/{num_clauses} "
              f"({frac_sat:.3%} of clauses)")

    return results

# Print a summary table of benchmark results
def print_benchmark_table(results: Sequence[Dict[str, Any]]) -> None:
    print("\n==================== Benchmark Summary ====================")
    header = (
        f"{'Size':<8}"
        f"{'#Vars':>8}  "
        f"{'#Clauses':>10}  "
        f"{'BestFit':>8}  "
        f"{'SatClauses':>12}  "
        f"{'FracSat':>8}  "
        f"{'Time(s)':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['label']:<8}"
            f"{r['num_vars']:>8}  "
            f"{r['num_clauses']:>10}  "
            f"{r['best_fitness']:>8.4f}  "
            f"{r['best_satisfied']:>12d}  "
            f"{r['fraction_satisfied']:>8.3f}  "
            f"{r['time_seconds']:>8.3f}"
        )
    print("===========================================================\n")

# Run repeated baseline GA experiments
def run_repeated_baseline_experiment(
    label: str,
    instance_path: str,
    base_config: GAConfig,
    num_runs: int = 10,
    verbose_each_run: bool = False,
    save_json: bool = True,
) -> None:
    instance = load_3sat_instance(instance_path)
    num_clauses = len(instance.clauses)

    all_runs: Dict[str, Any] = {}
    best_fits: List[float] = []
    solved_flags: List[bool] = []
    solution_gens: List[int] = []

    for r in range(1, num_runs + 1):
        cfg = GAConfig(
            pop_size=base_config.pop_size,
            num_generations=base_config.num_generations,
            crossover_rate=base_config.crossover_rate,
            mutation_prob=base_config.mutation_prob,
            tournament_size=base_config.tournament_size,
            elitism_k=base_config.elitism_k,
            alpha_bonus=base_config.alpha_bonus,
            random_seed=(base_config.random_seed or 42) + r,
        )

        stats: Dict[str, Any] = {}
        _, best_fit, best_sat = run_ga(
            instance,
            cfg,
            verbose=verbose_each_run,
            stats=stats,
        )

        solved = (best_sat == num_clauses)
        best_fits.append(best_fit)
        solved_flags.append(solved)

        sol_gen = stats.get("solution_generation")
        if solved and sol_gen is not None:
            solution_gens.append(sol_gen)

        all_runs[f"run_{r}"] = {
            "best_fitness": best_fit,
            "satisfied": best_sat,
            "total_clauses": num_clauses,
            "solved": solved,
            "solution_generation": sol_gen,
        }

    # Summary stats
    success_rate = 100.0 * sum(solved_flags) / num_runs
    avg_best_fit = sum(best_fits) / num_runs
    avg_sol_gen = sum(solution_gens) / len(solution_gens) if solution_gens else None

    summary = {
        "success_rate": success_rate,
        "avg_best_fitness": avg_best_fit,
        "avg_solution_generation": avg_sol_gen,
    }

    experiment_data = {
        "experiment_name": label,
        "num_runs": num_runs,
        "results": all_runs,
        "summary": summary,
    }

    print(f"\n[Baseline Experiment: {label}]")
    print(f"  Runs: {num_runs}")
    print(f"  Avg best fitness: {avg_best_fit:.4f}")
    print(f"  Success rate: {success_rate:.1f}%")
    if avg_sol_gen is not None:
        print(f"  Avg solution generation (successful runs): {avg_sol_gen:.1f}")
    else:
        print("  No fully satisfying runs; no solution generation to report.")

    if save_json:
        filename = f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '_')}.json"
        save_experiment_log(experiment_data, filename)

# Run repeated Wisdom-of-Crowds GA experiments
def run_repeated_woc_experiment(
    label: str,
    instance_path: str,
    base_config: GAConfig,
    num_runs: int = 10,
    verbose_each_run: bool = False,
    save_json: bool = True,
    **woc_kwargs,
) -> None:
    instance = load_3sat_instance(instance_path)
    num_clauses = len(instance.clauses)

    all_runs: Dict[str, Any] = {}
    best_fits: List[float] = []
    solved_flags: List[bool] = []
    solution_gens: List[int] = []

    for r in range(1, num_runs + 1):
        cfg = GAConfig(
            pop_size=base_config.pop_size,
            num_generations=base_config.num_generations,
            crossover_rate=base_config.crossover_rate,
            mutation_prob=base_config.mutation_prob,
            tournament_size=base_config.tournament_size,
            elitism_k=base_config.elitism_k,
            alpha_bonus=base_config.alpha_bonus,
            random_seed=(base_config.random_seed or 42) + r,
        )
        stats: Dict[str, Any] = {}
        _, best_fit, best_sat = wisdom_of_crowds_ga(
            instance=instance,
            base_config=cfg,
            verbose=verbose_each_run,
            stats=stats,
            **woc_kwargs,
        )

        solved = (best_sat == num_clauses)
        best_fits.append(best_fit)
        solved_flags.append(solved)
        sol_gen = stats.get("solution_generation")

        # Store per-run data
        all_runs[f"run_{r}"] = {
            "best_fitness": best_fit,
            "satisfied": best_sat,
            "total_clauses": num_clauses,
            "solved": solved,
            "solution_generation": sol_gen,
        }
        if solved and sol_gen is not None:
            solution_gens.append(sol_gen)

    # Compute summary
    success_rate = 100.0 * sum(solved_flags) / num_runs
    avg_best_fit = sum(best_fits) / num_runs
    avg_sol_gen = sum(solution_gens) / len(solution_gens) if solution_gens else None

    summary = {
        "success_rate": success_rate,
        "avg_best_fitness": avg_best_fit,
        "avg_solution_generation": avg_sol_gen,
    }

    # Create experiment record 
    experiment_data = {
        "experiment_name": label,
        "num_runs": num_runs,
        "results": all_runs,
        "summary": summary,
    }

    # Print summary 
    print(f"\n[WoC Experiment: {label}]")
    print(f"  Runs: {num_runs}")
    print(f"  Avg best fitness: {avg_best_fit:.4f}")
    print(f"  Success rate: {success_rate:.1f}%")
    if avg_sol_gen is not None:
        print(f"  Avg solution generation (successful runs): {avg_sol_gen:.1f}")
    else:
        print("  No fully satisfying runs; no solution generation to report.")

    # Save to JSON 
    if save_json:
        filename = f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '_')}.json"
        save_experiment_log(experiment_data, filename)



if __name__ == "__main__":
    # Base GA configuration used for all instance sizes
    base_config = GAConfig(
        pop_size=200,
        num_generations=500, 
        crossover_rate=0.7,
        mutation_prob=0.02,
        tournament_size=3,
        elitism_k=2,
        alpha_bonus=0.1,
        random_seed=42,
    )

    
    results = benchmark_ga_on_instances(
        base_config=base_config,
        instance_prefix="3sat_instance_",
        labels=("small", "medium", "large"),
        verbose_each_run=False,  
    )

    # Print a compact summary table
    print_benchmark_table(results)


    #  Single demo run per instance: small, medium, large
    print("\n=== Step 6–7: Wisdom of Artificial Crowds (all instances) ===")

    for label in ("small", "medium", "large"):
        path = f"3sat_instance_{label}.json"
        print(f"\n--- Running Wisdom-of-Crowds GA on {label.upper()} instance ---")
        instance = load_3sat_instance(path)

        best_chrom, best_fit, best_sat = wisdom_of_crowds_ga(
            instance=instance,
            base_config=base_config,
            num_subpops=5,          
            top_k_per_subga=5,       
            inject_rate=0.2,       
            wisdom_mut_prob=0.05,   
            aggregation_interval=20,
            stagnation_limit=5,      
            weighted_wisdom=False,   
            verbose=True,
        )

        total_clauses = len(instance.clauses)
        print(
            f"[{label.upper()}] best fitness={best_fit:.4f}, "
            f"satisfied={best_sat}/{total_clauses} "
            f"({best_sat/total_clauses:.2%} clauses)"
        )


    print("\n=== Step 8: Baseline Experiments (JSON-logged) ===")
    # 1) Baseline GA (reference) on SMALL:
    run_repeated_baseline_experiment(
        label="Baseline_small",
        instance_path="3sat_instance_small.json",
        base_config=base_config,
        num_runs=20,
        verbose_each_run=False,
    )

    # 2) Baseline GA (reference) on MEDIUM:
    run_repeated_baseline_experiment(
        label="Baseline_medium",
        instance_path="3sat_instance_medium.json",
        base_config=base_config,
        num_runs=20,
        verbose_each_run=False,
    )

    # 3) Baseline GA (reference) on LARGE:
    run_repeated_baseline_experiment(
        label="Baseline_large",
        instance_path="3sat_instance_large.json",
        base_config=base_config,
        num_runs=20,
        verbose_each_run=False,
    )


    #2) WoC (K = 3) vs baseline (medium):
    run_repeated_woc_experiment(
        label="WoC_K3_medium",
        instance_path="3sat_instance_medium.json",
        base_config=base_config,
        num_runs=20,
        num_subpops=3,
        top_k_per_subga=5,
        inject_rate=0.2,
        wisdom_mut_prob=0.05,
        aggregation_interval=20,
        stagnation_limit=5,
        weighted_wisdom=False,
    )

    # 3) WoC (K = 10) vs baseline (medium):
    run_repeated_woc_experiment(
        label="WoC_K10_medium",
        instance_path="3sat_instance_medium.json",
        base_config=base_config,
        num_runs=20,
        num_subpops=10,
        top_k_per_subga=5,
        inject_rate=0.2,
        wisdom_mut_prob=0.05,
        aggregation_interval=20,
        stagnation_limit=5,
        weighted_wisdom=False,
    )

    # 4) Mutation 0.01 vs 0.05 (on medium, K=5):
    low_mut = GAConfig(**{**base_config.__dict__, "mutation_prob": 0.01})
    high_mut = GAConfig(**{**base_config.__dict__, "mutation_prob": 0.05})
    
    run_repeated_woc_experiment(
        label="WoC_K5_mut0_01_medium",
        instance_path="3sat_instance_medium.json",
        base_config=low_mut,
        num_runs=20,
        num_subpops=5,
        top_k_per_subga=5,
        inject_rate=0.2,
        wisdom_mut_prob=0.05,
        aggregation_interval=20,
        stagnation_limit=5,
        weighted_wisdom=False,
    )
    
    run_repeated_woc_experiment(
        label="WoC_K5_mut0_05_medium",
        instance_path="3sat_instance_medium.json",
        base_config=high_mut,
        num_runs=20,
        num_subpops=5,
        top_k_per_subga=5,
        inject_rate=0.2,
        wisdom_mut_prob=0.05,
        aggregation_interval=20,
        stagnation_limit=5,
        weighted_wisdom=False,
    )

    # 5) Small vs Large (problem size) with WoC (K=5):
    run_repeated_woc_experiment(
        label="WoC_K5_small",
        instance_path="3sat_instance_small.json",
        base_config=base_config,
        num_runs=20,
        num_subpops=5,
        top_k_per_subga=5,
        inject_rate=0.2,
        wisdom_mut_prob=0.05,
        aggregation_interval=20,
        stagnation_limit=5,
        weighted_wisdom=False,
    )
    
    run_repeated_woc_experiment(
        label="WoC_K5_large",
        instance_path="3sat_instance_large.json",
        base_config=base_config,
        num_runs=20,
        num_subpops=5,
        top_k_per_subga=5,
        inject_rate=0.2,
        wisdom_mut_prob=0.05,
        aggregation_interval=20,
        stagnation_limit=5,
        weighted_wisdom=False,
    )

    # 6) Weighted vs Unweighted wisdom (aggregation method) on MEDIUM:
    run_repeated_woc_experiment(
        label="WoC_Unweighted_medium",
        instance_path="3sat_instance_medium.json",
        base_config=base_config,
        num_runs=20,
        num_subpops=5,
        top_k_per_subga=5,
        inject_rate=0.2,
        wisdom_mut_prob=0.05,
        aggregation_interval=20,
        stagnation_limit=5,
        weighted_wisdom=False,
    )
    
    run_repeated_woc_experiment(
        label="WoC_Weighted_medium",
        instance_path="3sat_instance_medium.json",
        base_config=base_config,
        num_runs=20,
        num_subpops=5,
        top_k_per_subga=5,
        inject_rate=0.2,
        wisdom_mut_prob=0.05,
        aggregation_interval=20,
        stagnation_limit=5,
        weighted_wisdom=True,
    )
