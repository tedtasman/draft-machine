import sys
import pandas as pd

def detect_cliffs(position, df):
    # Get the player's data
    player_data = df[df["Position"] == position]

    # Calculate delta to next player
    player_data = player_data.copy()
    player_data["Delta"] = player_data["Consensus Proj"].diff().fillna(0)
    return player_data

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 cleaner.py input.csv output.csv")
        sys.exit(1)

    input_df = pd.read_csv(sys.argv[1])

    output_df = detect_cliffs("WR", input_df)
    output_df.to_csv(sys.argv[2], index=False)
