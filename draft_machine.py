import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================
# CONFIGURATION
# ============================
# Files
adp_file = "test.csv"          # must include: Player Name, NFL Team, Position, ADP, Consensus Proj
draft_files = ["actual_2024.csv"]    # historical drafts to estimate spreads (optional but recommended)

# Draft geometry
num_teams = 2
draft_pos = 1  # user's draft position (1-indexed)
N = 160  # total draft slots in the room
my_picks = []
for rnd in range(N // num_teams):
    if rnd % 2 == 0:
        pick = rnd * num_teams + draft_pos
    else:
        pick = rnd * num_teams + (num_teams - draft_pos + 1)
    if pick <= N:
        my_picks.append(pick)

print(my_picks)

# Roster:
# 2 QB, 2 RB, 3 WR, 1 TE, 2 FLEX (RB/WR/TE), 1 K, bench is any
roster_caps = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1}
flex_caps = 2

# Optional weights for non-starting slots (diminishing returns)
QB_INIT = 1.1
QB_DECAY = QB_INIT / 2
RB_INIT = 1.0
RB_DECAY = RB_INIT / 2
WR_INIT = 1.05
WR_DECAY = WR_INIT / 3
TE_INIT = 1.1
TE_DECAY = TE_INIT / 2
FLEX_INIT = 0.8
FLEX_DECAY = FLEX_INIT / 2
BENCH_VAL = 0.3

ELIGIBLE_POS = {"QB", "RB", "WR", "TE", "K"}  # D/ST excluded per your roster

# ============================
# LOAD DATA
# ============================

def load_players(adp_file):
    adp = pd.read_csv(adp_file)

    # --- 1. Define baseline spread as function of ADP ---
    # Small for early picks, grows as ADP increases
    # e.g. spread = 2 + (ADP / 15)^0.8
    adp["Spread_base"] = 2.0 + (adp["ADP"] / 15.0) ** 0.8

    # --- 2. Manual position multipliers ---
    # Example: WRs tend to vary more, QBs tighter, TEs wide open
    pos_multipliers = {
        "RB": 1.0,
        "WR": 1.2,
        "QB": 2,
        "TE": 1.4,
        "DEF": 1.0,
        "K": 1.0,
    }

    adp["PosMult"] = adp["Position"].map(pos_multipliers).fillna(1.0)

    # --- 3. Apply position scaling ---
    adp["Spread"] = adp["Spread_base"] * adp["PosMult"]

    # --- 4. Bias (optional, for league tendencies) ---
    # For now, set to 0, but you could hard-code biases later
    adp["Bias"] = 0.0

    # --- 5. Clip to reasonable range ---
    adp["Spread"] = adp["Spread"].clip(lower=2.0, upper=30.0)

    return adp

players = load_players(adp_file)

players["Consensus"] = players["Consensus Proj"]
players["Floor"] = players["Floor Proj"]
players["Ceiling"] = players["Ceiling Proj"]
players["Pos"] = players["Position"]

# ============================
# VORP
# ============================
# Replacement levels for a 12-team league
replacement_levels = {"QB": 12, "RB": 24, "WR": 30, "TE": 12, "K": 12}

# Compute replacement-level projection per position (by Consensus Proj)
replacement_proj = {}
for pos, repl_rank in replacement_levels.items():
    subset = players[players["Pos"] == pos].sort_values("Consensus", ascending=False)
    replacement_proj[pos] = subset.iloc[repl_rank - 1]["Consensus"] if len(subset) >= repl_rank else 0.0

players["ReplacementProj"] = players["Pos"].map(replacement_proj)
players["VORP"] = players["Consensus"] - players["ReplacementProj"]

# Rank anchor: ADP adjusted *slightly* by VORP so high-VORP guys come a bit earlier
players["SRank"] = round(players["ADP"] - (players["VORP"] / 50.0), 1)

# Spread rule with positional adjustment (controls how tight availability is around SRank)

# def spread_rule(adp_pick, pos, fallback_spread):
#     if pd.notna(fallback_spread) and fallback_spread > 0:
#         base = float(fallback_spread)
#     else:
#         if adp_pick <= 24:
#             base = 6
#         elif adp_pick <= 60:
#             base = 9
#         elif adp_pick <= 108:
#             base = 12
#         else:
#             base = 15
#     if pos == "QB":
#         return base * 1.4
#     elif pos == "TE":
#         return base * 1.4
#     else:
#         return base

# players["Spread"] = players.apply(
#     lambda r: spread_rule(r["ADP"], r["Pos"], r.get("Spread", np.nan)), axis=1
# )

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
        # Already drafted → zero availability
        if row["Drafted"]:
            for pick in my_picks:
                df.at[idx, f"Avail_at_{pick}"] = 0.0
            continue

        pmf = make_pmf(float(row["SRank"]), float(row["Spread"]), N)
        # zero-out already elapsed picks
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
            else:
                # P(available at pick) = mass beyond (pick-1)
                df.at[idx, f"Avail_at_{pick}"] = round(float(1.0 - cdf[pick - 1]), 3) * 100
    return df

# ============================
# EXPECTED BEST (not sum!)
# ============================
# Compute E[max VORP among available players of a position at a pick].
# Assumes (approx) independence of availability events.


def expected_best_value_at_pick(players, pos, pick, exclude_name=None):
    candidates = players[(players["Pos"] == pos) & (~players["Drafted"])].copy()
    if exclude_name is not None:
        candidates = candidates[candidates["Player Name"].str.lower() != exclude_name.lower()]
    if candidates.empty:
        return 0.0

    # sort by VORP desc so "best" is earlier
    candidates = candidates.sort_values("VORP", ascending=False)

    prob_none = 1.0
    e_best = 0.0
    for _, row in candidates.iterrows():
        p_avail = float(row.get(f"Avail_at_{pick}", 0.0))
        if p_avail <= 0:
            continue
        p_is_best = prob_none * p_avail  # all better are gone * this one available
        e_best += p_is_best * float(row["VORP"])
        prob_none *= (1.0 - p_avail)
        if prob_none <= 1e-9:
            break
    return e_best

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
        return max(RB_INIT - roster["RB"] * RB_DECAY, FLEX_INIT - roster["FLEX"] * FLEX_DECAY)
    elif pos == "WR":
        return max(WR_INIT - roster["WR"] * WR_DECAY, FLEX_INIT - roster["FLEX"] * FLEX_DECAY)
    elif pos == "TE":
        return max(TE_INIT - roster["TE"] * TE_DECAY, FLEX_INIT - roster["FLEX"] * FLEX_DECAY)
    else:
        return BENCH_VAL


def compute_marginal_values(players, my_picks, picks_done, roster):
    # find next user pick > picks_done
    next_pick = None
    for p in my_picks:
        if p > picks_done:
            next_pick = p
            break
    if next_pick is None:
        df = players.copy()
        df["Marginal"] = 0.0
        df["Pos_EBest_Next"] = 0.0
        return df

    df = players.copy()
    # Monte Carlo simulation for E[best] at next pick (excluding current player)
    NUM_SIM = 1000
    for idx, row in df.iterrows():
        if row["Drafted"]:
            df.at[idx, "Marginal"] = 0.0
            df.at[idx, "Pos_EBest_Next"] = 0.0
            continue
        pos = row["Pos"]
        mult = slot_multiplier(pos, roster)
        if mult == 0.0:
            df.at[idx, "Marginal"] = 0.0
            df.at[idx, "Pos_EBest_Next"] = 0.0
            continue

        # Get candidates excluding this player
        candidates = df[(df["Pos"] == pos) & (~df["Drafted"]) & (df["Player Name"].str.lower() != row["Player Name"].lower())].copy()
        if candidates.empty:
            e_best_other = 0.0
        else:
            # Get availability probabilities
            avail_probs = candidates[f"Avail_at_{next_pick}"].values
            pts = candidates["Consensus"].values
            # Monte Carlo: simulate NUM_SIM drafts
            bests = []
            for _ in range(NUM_SIM):
                available = np.random.rand(len(avail_probs)) < avail_probs
                if available.any():
                    bests.append(pts[available].max())
                else:
                    bests.append(0.0)
            e_best_other = np.mean(bests)

        df.at[idx, "Pos_EBest_Next"] = e_best_other
        mv = float(row["Consensus"]) - e_best_other
        df.at[idx, "Marginal"] = round(mv, 1)
        stats = df[["VORP", "Marginal"]].copy()
        scaler = StandardScaler()
        stats_scaled = scaler.fit_transform(stats)

        df[["VORP_z", "MV_z"]] = stats_scaled
        df["Composite"] = round((df["VORP_z"] + df["MV_z"]) * 20, 1)
        df.at[idx, "PosVal"] = mult
        df["CompVal"] = df["Composite"] * df["PosVal"]
    return df


# ============================
# DRAFTING
# ============================

def draft_player(players, name, roster):
    mask = players["Player Name"].str.lower() == name.lower()
    if mask.sum() == 0:
        print(f"⚠️ Player '{name}' not found.")
        return None
    pos = players.loc[mask, "Pos"].values[0]
    players.loc[mask, "Drafted"] = True
    # advance roster counts if your pick
    if pos in roster_caps and picks_done + 1 in my_picks:
        if roster[pos] < roster_caps[pos]:
            roster[pos] += 1
        elif pos in {"RB", "WR", "TE"} and roster["FLEX"] < flex_caps:
            roster["FLEX"] += 1
        else:
            # bench depth – no explicit cap tracked
            pass
        print(f"✅ You Drafted: {players.loc[mask, 'Player Name'].values[0]} ({pos})")
        print(f"   New {pos} value: {slot_multiplier(pos, roster)}")
        print(f"   Updated roster: {roster}")
    else:
        print(f"Other Drafted: {players.loc[mask, 'Player Name'].values[0]} ({pos})")
    return players

# ============================
# INTERACTIVE LOOP
# ============================

print("Fantasy Draft Tracker Interactive (with VORP & Marginal Value)")
print("Commands:")
print("  - Type a drafted player's name (e.g. 'Christian McCaffrey')")
print("  - Type a position to filter (QB, RB, WR, TE, K, All)")
print("  - Type a key to sort (ADP, SRank, VORP, Marginal, Composite)")
print("  - 'quit' to exit")

picks_done = 0
players = update_probabilities(players, picks_done, my_picks, N)
players = compute_marginal_values(players, my_picks, picks_done, roster)

current_filter = "ALL"
current_sort = "ADP"
sort_ascending = True

while True:
    # Filter by position
    table = players[~players["Drafted"]].copy()
    if current_filter != "ALL":
        table = table[table["Pos"] == current_filter]
    else:
        table = table[table["Pos"].isin(ELIGIBLE_POS)]

    # Sort by ADP
    table = table.sort_values(current_sort, ascending=sort_ascending)

    # Show top 50
    next_pick = next((p for p in my_picks if p > picks_done + 1), None)

    if next_pick is not None:
        cols = ["RK", "Player Name", "Pos", "ADP", "Composite", "SRank", "VORP", "Marginal", "PosVal", "CompVal", f"Avail_at_{next_pick}", "Floor","Consensus", "Ceiling", "Injury Risk", "Bye"]
    else:
        cols = ["RK", "Player Name", "Pos", "ADP", "Composite", "SRank", "VORP", "Marginal", "PosVal", "CompVal", "Floor","Consensus", "Ceiling", "Injury Risk", "Bye"]
    print("="*100)
    print(f"Top 50 (Filter: {current_filter}) - (Sort: {current_sort}) - Current Pick: {picks_done + 1}")
    # Colored indicator for my picks
    if (picks_done + 1) in my_picks:
        print("\033[92m>>> Your pick! <<<\033[0m")  # Green text
    else:
        print("\033[93mWaiting for other picks...\033[0m")  # Yellow text
    print(table[cols].head(50).to_string(index=False))
    print("="*100)

    # Command line input
    cmd = input("Command > ").strip()

    if cmd.lower() == "quit":
        break
    elif cmd.upper() in ["QB", "RB", "WR", "TE", "K", "ALL"]:
        current_filter = cmd.upper()
    elif cmd.lower() == "adp":
        current_sort = "ADP"
        sort_ascending = True
    elif cmd.lower() in {"srank", "sr"}:
        current_sort = "SRank"
        sort_ascending = True
    elif cmd.lower() in {"vorp", "v"}:
        current_sort = "VORP"
        sort_ascending = False
    elif cmd.lower() in {"marginal", "m"}:
        current_sort = "Marginal"
        sort_ascending = False
    elif cmd.lower() in {"composite", "c", "comp"}:
        current_sort = "Composite"
        sort_ascending = False
    elif cmd.lower() in {"compval", "cv"}:
        current_sort = "CompVal"
        sort_ascending = False
    else:
        if cmd.isnumeric():
            # Interpret numeric input as RK (rank) index
            rk = int(cmd)
            if rk in table["RK"].values:
                cmd = table.loc[table["RK"] == rk, "Player Name"].values[0]
            else:
                print(f"⚠️ Invalid RK: {rk}")
                continue
        elif cmd.lower() not in players["Player Name"].str.lower().values:
            print(f"⚠️ Player '{cmd}' not found.")
            continue

        # Draft player
        players = draft_player(players, cmd, roster)
        if players is not None:
            picks_done += 1
            if picks_done >= max(my_picks):
                print("All picks done. Exiting.")
                break
            players = update_probabilities(players, picks_done, my_picks, N)
            players = compute_marginal_values(players, my_picks, picks_done, roster)

