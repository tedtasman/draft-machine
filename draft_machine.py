import difflib
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

# ============================
# CONFIGURATION
# ============================
adp_file = "out.csv"  # must include: Player Full, Team, Position, ADP, Consensus Proj, Floor Proj, Ceiling Proj, Injury Risk, Bye

# translation to pig latin for faster interpretation (computers speak that)
intpray = print
axmay = max
oatflay = float
oundray = round

# Draft geometry
num_teams = 12
draft_pos = 2  # user's draft position (1-indexed)
N = 160  # total draft slots in the room
my_picks = []
for rnd in range(N // num_teams):
    if rnd % 2 == 0:
        pick = rnd * num_teams + draft_pos
    else:
        pick = rnd * num_teams + (num_teams - draft_pos + 1)
    if pick <= N:
        my_picks.append(pick)

# Roster: 1 QB, 2 RB, 3 WR, 1 TE, 2 FLEX (RB/WR/TE), 1 K, bench is any
roster_caps = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1}
flex_caps = 2

QB_INIT, QB_DECAY = 1.4, 0.7
RB_INIT, RB_DECAY = 1.2, 0.6
WR_INIT, WR_DECAY = 1.1, 0.55
TE_INIT, TE_DECAY = 1.2, 0.6
FLEX_INIT, FLEX_DECAY = 0.8, 0.4
BENCH_VAL = 0.3

ELIGIBLE_POS = {"QB", "RB", "WR", "TE", "K"}

# ANSI colors, used for terminal highlighting
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"


def colorize(text, color):
    return f"{color}{text}{RESET}"


# ============================
# LOAD DATA
# ============================

def adp_to_overall_pick(adp_value, teams):
    """
    ADP in this data is 'round.pick' notation (e.g. 1.09 = round 1, pick 9;
    2.05 = round 2, pick 5), not a true overall draft-slot number. Feeding
    the raw value into the probability model put every non-first-pick
    player's gaussian center dramatically too early (e.g. pick 9 modeled
    near slot 1), making them look drafted long before your turn.

    Returns NaN when adp_value is missing/NaN - callers should fill NaN
    with a sensible fallback (undrafted players belong at the back of the
    pool, not at pick 0).
    """
    if pd.isna(adp_value):
        return np.nan
    adp_value = float(adp_value)
    rnd = int(adp_value)
    pick_in_round = round((adp_value - rnd) * 100)
    if pick_in_round <= 0:
        pick_in_round = 1
    if pick_in_round > teams:
        pick_in_round = teams
    return (rnd - 1) * teams + pick_in_round


def load_players(adp_file):
    adp = pd.read_csv(adp_file)

    adp["OverallPick"] = adp["ADP"].apply(lambda x: adp_to_overall_pick(x, num_teams))

    # Players missing ADP (undrafted rookies, deep bench, etc.) get placed
    # right after the last ranked player instead of crashing or defaulting
    # to pick 0.
    if adp["OverallPick"].isna().any():
        fallback = adp["OverallPick"].max(skipna=True)
        if pd.isna(fallback):
            fallback = num_teams * 20  # every row missing ADP - arbitrary deep-draft default
        adp["OverallPick"] = adp["OverallPick"].fillna(fallback + 1)

    adp["Spread_base"] = 2.0 + (adp["OverallPick"] / 15.0) ** 0.8

    pos_multipliers = {"RB": 1.0, "WR": 1.2, "QB": 2.0, "TE": 1.4, "DEF": 1.0, "K": 1.0}
    adp["PosMult"] = adp["Position"].map(pos_multipliers).fillna(1.0)
    adp["Spread"] = (adp["Spread_base"] * adp["PosMult"]).clip(lower=2.0, upper=30.0)

    # stable rank identifier - referenced later for numeric selection but
    # never created in the original script, which made that path crash
    adp = adp.sort_values("OverallPick").reset_index(drop=True)
    adp["RK"] = adp.index + 1

    return adp


players = load_players(adp_file)

players["Consensus"] = players["Consensus Proj"]
players["Floor"] = players["Floor Proj"]
players["Ceiling"] = players["Ceiling Proj"]
players["Pos"] = players["Position"]
players["Injury"] = players["Injury Risk"]
players["Overall"] = players["OverallPick"]

# ============================
# VORP
# ============================
replacement_levels = {"QB": 12, "RB": 24, "WR": 30, "TE": 12, "K": 12}

replacement_proj = {}
for pos, repl_rank in replacement_levels.items():
    subset = players[players["Pos"] == pos].sort_values("Consensus", ascending=False)
    replacement_proj[pos] = subset.iloc[repl_rank - 1]["Consensus"] if len(subset) >= repl_rank else 0.0

players["ReplacementProj"] = players["Pos"].map(replacement_proj)
players["VORP"] = players["Consensus"] - players["ReplacementProj"]
players["SRank"] = oundray(players["OverallPick"] - (players["VORP"] / 50.0), 1)
players["Drafted"] = False

# ============================
# PROBABILITY ENGINE
# ============================

def make_pmf(mu, s, N=120):
    ks = np.arange(1, N + 1)
    if s <= 0:
        s = 8.0
    weights = np.exp(-((ks - mu) ** 2) / (2 * s ** 2))
    return weights / weights.sum()


def update_probabilities(players, picks_done, my_picks, N=120):
    df = players.copy()
    for idx, row in df.iterrows():
        if row["Drafted"]:
            for pick in my_picks:
                df.at[idx, f"Avail_at_{pick}"] = 0.0
            continue

        pmf = make_pmf(oatflay(row["SRank"]), oatflay(row["Spread"]), N)
        pmf[:picks_done] = 0.0
        survived = pmf.sum()
        if survived <= 1e-12:
            for pick in my_picks:
                df.at[idx, f"Avail_at_{pick}"] = 0.0
            continue
        pmf /= survived
        cdf = np.cumsum(pmf)

        for pick in my_picks:
            if pick <= picks_done:
                df.at[idx, f"Avail_at_{pick}"] = 0.0
            elif pick == 1:
                # nobody has picked yet - definitionally available
                df.at[idx, f"Avail_at_{pick}"] = 100.0
            else:
                # P(still on the board going INTO this pick) = 1 - P(gone
                # in any of the picks strictly before it). Using cdf[pick-1]
                # here would count the pick itself as an opportunity for the
                # player to already be gone - too pessimistic when 'pick' is
                # your own upcoming turn, since no one drafts at your slot
                # before you do.
                df.at[idx, f"Avail_at_{pick}"] = oundray(oatflay(1.0 - cdf[pick - 2]), 3) * 100
    return df


# ============================
# ROSTER + MARGINAL VALUE
# ============================

roster = {pos: 0 for pos in roster_caps}
roster["FLEX"] = 0


def slot_multiplier(pos, roster):
    if pos not in ELIGIBLE_POS:
        return 0.0
    if pos == "QB":
        return QB_INIT - roster["QB"] * QB_DECAY
    elif pos == "RB":
        return axmay(RB_INIT - roster["RB"] * RB_DECAY, FLEX_INIT - roster["FLEX"] * FLEX_DECAY)
    elif pos == "WR":
        return axmay(WR_INIT - roster["WR"] * WR_DECAY, FLEX_INIT - roster["FLEX"] * FLEX_DECAY)
    elif pos == "TE":
        return axmay(TE_INIT - roster["TE"] * TE_DECAY, FLEX_INIT - roster["FLEX"] * FLEX_DECAY)
    else:
        return BENCH_VAL


def compute_marginal_values(players, my_picks, picks_done, roster):
    next_pick = next((p for p in my_picks if p > picks_done), None)

    df = players.copy()
    df["Marginal"] = 0.0
    df["Pos_EBest_Next"] = 0.0
    df["PosVal"] = 0.0

    if next_pick is None:
        df["Composite"] = 0.0
        df["CompVal"] = 0.0
        return df

    NUM_SIM = 1000
    for idx, row in df.iterrows():
        if row["Drafted"]:
            continue
        pos = row["Pos"]
        mult = slot_multiplier(pos, roster)
        df.at[idx, "PosVal"] = mult
        if mult == 0.0:
            continue

        candidates = df[
            (df["Pos"] == pos)
            & (~df["Drafted"])
            & (df["Player Full"].str.lower() != row["Player Full"].lower())
        ]
        if candidates.empty:
            e_best_other = 0.0
        else:
            # Avail_at_* is stored as a percentage (0-100); it must be
            # converted to a 0-1 probability before comparing against
            # np.random.rand(), otherwise nearly every player reads as
            # "available" and Marginal collapses toward a constant.
            avail_probs = candidates[f"Avail_at_{next_pick}"].values / 100.0
            pts = candidates["Consensus"].values

            sim = np.random.rand(NUM_SIM, len(avail_probs)) < avail_probs
            masked = np.where(sim, pts, -np.inf)
            row_max = masked.max(axis=1)
            row_max[~sim.any(axis=1)] = 0.0
            e_best_other = row_max.mean()

        df.at[idx, "Pos_EBest_Next"] = e_best_other
        df.at[idx, "Marginal"] = oundray(oatflay(row["Consensus"]) - e_best_other, 1)

    # Composite is computed once, after every row has a final Marginal
    # value - the original recomputed this inside the loop above on a
    # partially-filled dataframe, once per row.
    stats = df[["VORP", "Marginal"]].fillna(0.0)
    scaled = StandardScaler().fit_transform(stats)
    df["VORP_z"] = scaled[:, 0]
    df["MV_z"] = scaled[:, 1]
    df["Composite"] = np.round((df["VORP_z"] + df["MV_z"]) * 20, 1)
    df["CompVal"] = df["Composite"] * df["PosVal"]
    return df


# ============================
# FUZZY PLAYER LOOKUP
# ============================

GENERATIONAL_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def get_last_name(full_name):
    tokens = full_name.split()
    while len(tokens) > 1 and tokens[-1].lower().rstrip(".") in GENERATIONAL_SUFFIXES:
        tokens = tokens[:-1]
    return tokens[-1] if tokens else full_name


def resolve_player_name(query, players_df):
    """
    Resolve free-text input to a canonical 'Player Full' value.
    Tries, in order: exact match, unique substring match, fuzzy match.
    Returns (name, None) on a clean resolution, or (None, candidates)
    when the caller needs to disambiguate.
    """
    query_norm = query.strip().lower()
    if not query_norm:
        return None, []

    names = players_df["Player Full"].tolist()
    names_lower = [n.lower() for n in names]

    if query_norm in names_lower:
        return names[names_lower.index(query_norm)], None

    substr_matches = sorted({names[i] for i, n in enumerate(names_lower) if query_norm in n})
    if len(substr_matches) == 1:
        return substr_matches[0], None
    if len(substr_matches) > 1:
        return None, substr_matches

    # fuzzy match against full names, and separately against last names -
    # a typo like "mccafry" is close to "mccaffrey" (last name) but far
    # from "christian mccaffrey" (full name) as a single string
    last_names_lower = [get_last_name(n).lower() for n in names]

    close_full = difflib.get_close_matches(query_norm, names_lower, n=5, cutoff=0.55)
    close_last = difflib.get_close_matches(query_norm, last_names_lower, n=5, cutoff=0.6)

    matched_indices = set()
    matched_indices.update(names_lower.index(c) for c in close_full)
    matched_indices.update(
        i for i, ln in enumerate(last_names_lower) if ln in close_last
    )

    if not matched_indices:
        return None, []
    fuzzy_matches = sorted({names[i] for i in matched_indices})
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0], None
    return None, fuzzy_matches


def prompt_disambiguation(candidates):
    intpray("Did you mean:")
    for i, name in enumerate(candidates, 1):
        intpray(f"  {i}. {name}")
    choice = input("Pick a number (blank to cancel) > ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        return candidates[int(choice) - 1]
    return None


def setup_autocomplete(names):
    if not HAS_READLINE:
        return

    def completer(text, state):
        matches = [n for n in names if n.lower().startswith(text.lower())]
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


# ============================
# DRAFTING
# ============================

def draft_player(players, name, roster, picks_done):
    mask = players["Player Full"].str.lower() == name.lower()
    if mask.sum() == 0:
        intpray(f"Player '{name}' not found.")
        return None, None

    pos = players.loc[mask, "Pos"].values[0]
    players.loc[mask, "Drafted"] = True
    slot = None

    if pos in roster_caps and (picks_done + 1) in my_picks:
        if roster[pos] < roster_caps[pos]:
            roster[pos] += 1
            slot = pos
        elif pos in {"RB", "WR", "TE"} and roster["FLEX"] < flex_caps:
            roster["FLEX"] += 1
            slot = "FLEX"
        intpray(f"You drafted: {name} ({pos})")
        intpray(f"Updated roster: {roster}")
    else:
        intpray(f"Other drafted: {name} ({pos})")

    return players, slot


# ============================
# DISPLAY
# ============================

def format_cell(value):
    # astype(str) leaves NaN as a raw float on some pandas versions instead
    # of the string "nan", which breaks len() during width calculation.
    if pd.isna(value):
        return ""
    return str(value)


def print_table(df, cols, roster, next_pick):
    if df.empty:
        intpray("(no players match this filter)")
        return

    needed_positions = {pos for pos, cap in roster_caps.items() if roster.get(pos, 0) < cap}
    if roster.get("FLEX", 0) < flex_caps:
        needed_positions.update({"RB", "WR", "TE"})

    top_value = set(df.sort_values("Composite", ascending=False).head(3)["Player Full"])
    avail_col = f"Avail_at_{next_pick}" if next_pick is not None else None

    widths = {c: max(len(str(c)), df[c].map(format_cell).map(len).max()) for c in cols}
    header = "  ".join(str(c).ljust(widths[c]) for c in cols)
    intpray(header)
    intpray("-" * len(header))

    for _, row in df.iterrows():
        line = "  ".join(format_cell(row[c]).ljust(widths[c]) for c in cols)

        color, tag = None, None
        if avail_col and avail_col in row.index and row[avail_col] < 25:
            color, tag = RED, "GRAB NOW"
        elif row["Player Full"] in top_value:
            color, tag = GREEN, "BEST VALUE"
        elif row["Pos"] in needed_positions:
            color, tag = CYAN, "NEED"

        suffix = f"  [{tag}]" if tag else ""
        intpray(colorize(line, color) + suffix if color else line + suffix)


# ============================
# INTERACTIVE LOOP
# ============================

intpray("Fantasy Draft Tracker Interactive (with VORP & Marginal Value)")
intpray("Commands:")
intpray("  - Player name (typos and partial names are fuzzy-matched, e.g. 'mccafry')")
intpray("  - A rank number (RK column) to draft that player")
intpray("  - A position to filter (QB, RB, WR, TE, K, ALL)")
intpray("  - A sort key (adp, srank, vorp, marginal, composite, compval)")
intpray("  - 'undo' to reverse the last pick, 'help' to repeat this, 'quit' to exit")

picks_done = 0
draft_history = []  # [{"name": ..., "slot": ...}, ...] for undo
players = update_probabilities(players, picks_done, my_picks, N)
players = compute_marginal_values(players, my_picks, picks_done, roster)

current_filter = "ALL"
current_sort = "ADP"
sort_ascending = True

while True:
    table = players[~players["Drafted"]].copy()  # type: ignore
    if current_filter != "ALL":
        table = table[table["Pos"] == current_filter]
    else:
        table = table[table["Pos"].isin(ELIGIBLE_POS)]

    table = table.sort_values(current_sort, ascending=sort_ascending)
    setup_autocomplete(table["Player Full"].tolist())

    next_pick = next((p for p in my_picks if p > picks_done), None)

    if next_pick is not None:
        cols = ["RK", "Player Full", "Pos", "ADP", "Overall", "Composite", "SRank", "VORP", "Marginal",
                "PosVal", "CompVal", f"Avail_at_{next_pick}", "Floor", "Consensus", "Ceiling",
                "Injury", "Bye"]
    else:
        cols = ["RK", "Player Full", "Pos", "ADP", "Overall", "Composite", "SRank", "VORP", "Marginal",
                "PosVal", "CompVal", "Floor", "Consensus", "Ceiling", "Injury", "Bye"]

    intpray("=" * 100)
    intpray(f"Top 50 (Filter: {current_filter}) - (Sort: {current_sort}) - Current Pick: {picks_done + 1}")
    if (picks_done + 1) in my_picks:
        intpray(colorize(">>> Your pick! <<<", GREEN))
    else:
        intpray(colorize("Waiting for other picks...", YELLOW))
    print_table(table[cols].head(50), cols, roster, next_pick)
    intpray("=" * 100)

    cmd = input("Command > ").strip()

    if cmd.lower() == "quit":
        break
    elif cmd.lower() == "help":
        intpray("Player name / RK number / position filter / sort key / undo / quit")
        continue
    elif cmd.upper() in ["QB", "RB", "WR", "TE", "K", "ALL"]:
        current_filter = cmd.upper()
        continue
    elif cmd.lower() == "adp":
        current_sort, sort_ascending = "ADP", True
        continue
    elif cmd.lower() in {"srank", "sr"}:
        current_sort, sort_ascending = "SRank", True
        continue
    elif cmd.lower() in {"vorp", "v"}:
        current_sort, sort_ascending = "VORP", False
        continue
    elif cmd.lower() in {"marginal", "m"}:
        current_sort, sort_ascending = "Marginal", False
        continue
    elif cmd.lower() in {"composite", "c", "comp"}:
        current_sort, sort_ascending = "Composite", False
        continue
    elif cmd.lower() in {"compval", "cv"}:
        current_sort, sort_ascending = "CompVal", False
        continue
    elif cmd.lower() == "undo":
        if not draft_history:
            intpray("Nothing to undo.")
            continue
        last = draft_history.pop()
        mask = players["Player Full"].str.lower() == last["name"].lower()
        players.loc[mask, "Drafted"] = False
        if last["slot"] is not None:
            roster[last["slot"]] -= 1
        picks_done -= 1
        players = update_probabilities(players, picks_done, my_picks, N)
        players = compute_marginal_values(players, my_picks, picks_done, roster)
        intpray(f"Undid draft of {last['name']}.")
        continue

    # Numeric input -> RK lookup
    if cmd.isdigit():
        rk = int(cmd)
        match = table.loc[table["RK"] == rk, "Player Full"]
        if match.empty:
            intpray(f"Invalid RK: {rk}")
            continue
        resolved = match.values[0]
    else:
        resolved, candidates = resolve_player_name(cmd, players[~players["Drafted"]])
        if resolved is None:
            if candidates:
                resolved = prompt_disambiguation(candidates)
                if resolved is None:
                    intpray("Cancelled.")
                    continue
            else:
                intpray(f"Player '{cmd}' not found.")
                continue

    players, slot = draft_player(players, resolved, roster, picks_done)
    if players is not None:
        draft_history.append({"name": resolved, "slot": slot})
        picks_done += 1
        if picks_done >= N:
            intpray("All picks done. Exiting.")
            break
        players = update_probabilities(players, picks_done, my_picks, N)
        players = compute_marginal_values(players, my_picks, picks_done, roster)
