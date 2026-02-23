import random
from typing import Dict, List, Tuple, Optional


# ============================================================
# Part 1: Market Setup
# ============================================================

def generate_market(
    n_students: int = 18,
    schools: Optional[List[str]] = None,
    capacity_each: int = 6,
    seed: int = 2026,
) -> Tuple[List[str], List[str], Dict[str, int], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate a random school choice market:
      - students: i1,...,i18
      - schools: s1,s2,s3 (default)
      - each school has capacity_each seats
      - prefs: strict random preference ordering for each student over schools
      - priority: strict random priority ordering for each school over students

    IMPORTANT: prefs and priority are generated from independent RNG streams.
    """
    if schools is None:
        schools = ["s1", "s2", "s3"]

    students = [f"i{k}" for k in range(1, n_students + 1)]
    capacity = {s: capacity_each for s in schools}

    # Independent RNG streams for prefs vs. priority
    rng_prefs = random.Random(seed)
    rng_prio = random.Random(seed + 1_000_003)

    prefs: Dict[str, List[str]] = {i: rng_prefs.sample(schools, k=len(schools)) for i in students}
    priority: Dict[str, List[str]] = {s: rng_prio.sample(students, k=len(students)) for s in schools}

    return students, schools, capacity, prefs, priority


def build_rank(priority: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """rank[s][i] smaller => higher priority at school s."""
    return {s: {i: r for r, i in enumerate(lst)} for s, lst in priority.items()}


# ============================================================
# Part 2: DA (Deferred Acceptance), student-proposing
# ============================================================

def da_student_proposing(
    students: List[str],
    schools: List[str],
    capacity: Dict[str, int],
    prefs: Dict[str, List[str]],
    priority: Dict[str, List[str]],
) -> Dict[str, str]:
    rank = build_rank(priority)

    next_idx = {i: 0 for i in students}     # next preference position to apply to
    held = {s: [] for s in schools}         # tentative admits
    free = set(students)                    # students who need to apply this round

    while True:
        applications = {s: [] for s in schools}
        any_app = False

        # Free students apply to next school on their list
        for i in list(free):
            if next_idx[i] >= len(prefs[i]):
                continue
            s = prefs[i][next_idx[i]]
            next_idx[i] += 1
            applications[s].append(i)
            any_app = True

        if not any_app:
            break

        new_free = set()

        # Each school keeps highest-priority up to capacity
        for s in schools:
            candidates = held[s] + applications[s]
            candidates.sort(key=lambda i: rank[s][i])
            kept = candidates[: capacity[s]]
            rejected = candidates[capacity[s]:]
            held[s] = kept
            new_free.update(rejected)

        free = new_free

    matching = {i: "unassigned" for i in students}
    for s in schools:
        for i in held[s]:
            matching[i] = s
    return matching


# ============================================================
# Part 2: IA (Immediate Acceptance / Boston)
# ============================================================

def ia_boston(
    students: List[str],
    schools: List[str],
    capacity: Dict[str, int],
    prefs: Dict[str, List[str]],
    priority: Dict[str, List[str]],
) -> Dict[str, str]:
    rank = build_rank(priority)

    remaining = {s: capacity[s] for s in schools}
    matching = {i: "unassigned" for i in students}
    unassigned = set(students)

    # Round k: apply to k-th choice
    for k in range(len(schools)):
        applications = {s: [] for s in schools}

        for i in list(unassigned):
            s = prefs[i][k]
            applications[s].append(i)

        newly_assigned = set()

        for s in schools:
            if remaining[s] <= 0:
                continue
            applicants = applications[s]
            applicants.sort(key=lambda i: rank[s][i])
            accepted = applicants[: remaining[s]]

            for i in accepted:
                matching[i] = s
                newly_assigned.add(i)

            remaining[s] -= len(accepted)

        unassigned -= newly_assigned
        if not unassigned:
            break

    return matching


# ============================================================
# Part 2: TTC (Top Trading Cycles) with capacities via seat objects
# ============================================================

def ttc_school_choice(
    students: List[str],
    schools: List[str],
    capacity: Dict[str, int],
    prefs: Dict[str, List[str]],
    priority: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    TTC using seat objects:
      - each school s has seats s_1, ..., s_q
      - each remaining student points to one seat from their top available school
      - each remaining seat points to the top-priority remaining student for that seat's school
      - find directed cycles and execute trades
    """
    matching = {i: "unassigned" for i in students}

    # Build seats
    seats: List[str] = []
    seat_to_school: Dict[str, str] = {}
    school_to_seats: Dict[str, List[str]] = {s: [] for s in schools}
    for s in schools:
        for k in range(1, capacity[s] + 1):
            seat_id = f"{s}_{k}"
            seats.append(seat_id)
            seat_to_school[seat_id] = s
            school_to_seats[s].append(seat_id)

    remaining_students = set(students)
    remaining_seats = set(seats)

    # Track each school's current best remaining student in its priority list
    school_priority_idx = {s: 0 for s in schools}

    def top_priority_student_for_school(s: str) -> Optional[str]:
        idx = school_priority_idx[s]
        plist = priority[s]
        while idx < len(plist) and plist[idx] not in remaining_students:
            idx += 1
        school_priority_idx[s] = idx
        if idx >= len(plist):
            return None
        return plist[idx]

    def first_available_seat(s: str) -> Optional[str]:
        for seat_id in school_to_seats[s]:
            if seat_id in remaining_seats:
                return seat_id
        return None

    def student_top_available_school(i: str) -> Optional[str]:
        for s in prefs[i]:
            if first_available_seat(s) is not None:
                return s
        return None

    def find_cycles(next_node: Dict[str, str], starts: List[str]) -> List[List[str]]:
        """
        Standard functional-graph cycle finding.
        Only need to start from student nodes (starts).
        """
        cycles: List[List[str]] = []
        globally_seen = set()

        for start in starts:
            if start in globally_seen or start not in next_node:
                continue

            cur = start
            path: List[str] = []
            pos: Dict[str, int] = {}

            while cur in next_node and cur not in globally_seen:
                globally_seen.add(cur)
                pos[cur] = len(path)
                path.append(cur)
                cur = next_node[cur]

                if cur in pos:
                    cycles.append(path[pos[cur]:])
                    break

        return cycles

    while remaining_students and remaining_seats:
        next_node: Dict[str, str] = {}

        # Students point to a seat in their top available school
        for i in list(remaining_students):
            s = student_top_available_school(i)
            if s is None:
                continue
            seat_id = first_available_seat(s)
            if seat_id is not None:
                next_node[i] = seat_id

        # Seats point to their school's top-priority remaining student
        for seat_id in list(remaining_seats):
            s = seat_to_school[seat_id]
            top_i = top_priority_student_for_school(s)
            if top_i is not None:
                next_node[seat_id] = top_i

        if not any(i in next_node for i in remaining_students):
            break

        cycles = find_cycles(next_node, list(remaining_students))
        if not cycles:
            break

        assigned_students = set()
        used_seats = set()

        # Execute cycles: whenever a student is in the cycle, they get the seat they point to
        for cycle in cycles:
            for node in cycle:
                if node.startswith("i"):
                    seat_id = next_node[node]
                    matching[node] = seat_to_school[seat_id]
                    assigned_students.add(node)
                    used_seats.add(seat_id)

        remaining_students -= assigned_students
        remaining_seats -= used_seats

    return matching


# ============================================================
# Part 3: Efficiency Analysis (Simulation)
# ============================================================

def assigned_rank(student: str, assigned_school: str, prefs: Dict[str, List[str]]) -> int:
    """
    rank = position of assigned_school in student's preference list:
      1 for top choice, 2 for second, ...
    """
    if assigned_school == "unassigned":
        return len(prefs[student]) + 1
    return prefs[student].index(assigned_school) + 1


def simulate_efficiency(
    N: int = 1000,
    base_seed: int = 2026,
    n_students: int = 18,
    schools: Optional[List[str]] = None,
    capacity_each: int = 6,
) -> Dict[str, float]:
    if schools is None:
        schools = ["s1", "s2", "s3"]

    sum_rank = {"DA": 0, "IA": 0, "TTC": 0}
    total_obs = N * n_students

    for t in range(N):
        seed_t = base_seed + t  # new market each iteration
        students, schools_now, capacity, prefs, priority = generate_market(
            n_students=n_students,
            schools=schools,
            capacity_each=capacity_each,
            seed=seed_t,
        )

        da_match = da_student_proposing(students, schools_now, capacity, prefs, priority)
        ia_match = ia_boston(students, schools_now, capacity, prefs, priority)
        ttc_match = ttc_school_choice(students, schools_now, capacity, prefs, priority)

        for i in students:
            sum_rank["DA"] += assigned_rank(i, da_match[i], prefs)
            sum_rank["IA"] += assigned_rank(i, ia_match[i], prefs)
            sum_rank["TTC"] += assigned_rank(i, ttc_match[i], prefs)

    return {k: v / total_obs for k, v in sum_rank.items()}


# ============================================================
# Printing / PDF-friendly outputs
# ============================================================

def print_market_summary(students, schools, capacity, prefs, priority):
    print("=== Market summary (single draw) ===")
    print("Students:", students)
    print("Schools:", schools)
    print("Capacity:", capacity)
    print("\nExample student preferences (first 5):")
    for i in students[:5]:
        print(f"  {i}: {prefs[i]}")
    print("\nExample school priorities (first 2 schools, top 8):")
    for s in schools[:2]:
        print(f"  {s}: {priority[s][:8]} ...")


def print_matching_table(title: str, matching: Dict[str, str], schools: List[str]):
    print(f"\n=== {title} ===")
    print(f"{'Student':<8} | {'Assigned school':<14}")
    print("-" * 26)
    for i in sorted(matching, key=lambda x: int(x[1:])):
        print(f"{i:<8} | {matching[i]:<14}")

    counts = {s: 0 for s in schools}
    counts["unassigned"] = 0
    for _, s in matching.items():
        if s in counts:
            counts[s] += 1
        else:
            counts["unassigned"] += 1
    print("\nCounts:", counts)


def print_efficiency_table(avg_rank: Dict[str, float]):
    print("\n=== Part 3: Efficiency Analysis (Average Student Rank) ===")
    print(f"{'Mechanism':<10} | {'Average Rank (lower is better)':>30}")
    print("-" * 45)
    for mech in ["DA", "IA", "TTC"]:
        print(f"{mech:<10} | {avg_rank[mech]:>30.4f}")

    best = min(avg_rank, key=avg_rank.get)
    print("\nBrief interpretation (use in your PDF):")
    print("Lower average rank means students receive more-preferred schools on average (1 = top choice).")
    print(f"In this simulation (N=1000), {best} has the lowest average rank among DA, IA, and TTC.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    # ---- Part 1 + Part 2: single market draw (seed fixed) ----
    students, schools, capacity, prefs, priority = generate_market(seed=2026)
    print_market_summary(students, schools, capacity, prefs, priority)

    da_match = da_student_proposing(students, schools, capacity, prefs, priority)
    ia_match = ia_boston(students, schools, capacity, prefs, priority)
    ttc_match = ttc_school_choice(students, schools, capacity, prefs, priority)

    print_matching_table("DA (Deferred Acceptance) (student -> school)", da_match, schools)
    print_matching_table("IA (Immediate Acceptance / Boston) (student -> school)", ia_match, schools)
    print_matching_table("TTC (Top Trading Cycles) (student -> school)", ttc_match, schools)

    # ---- Part 3: simulation ----
    avg_rank = simulate_efficiency(N=1000, base_seed=2026)
    print_efficiency_table(avg_rank)
 