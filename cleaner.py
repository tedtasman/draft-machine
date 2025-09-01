import sys
import pandas as pd


POSITIONS = ["WR", "TE", "QB", "K", "RB"]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 cleaner.py input.csv output.csv")
        sys.exit(1)

    input_df = pd.read_csv(sys.argv[1])

    # Drop unnecessary columns
    input_df.drop(columns=["Player Short"], inplace=True)

    # Rename columns for consistency
    input_df.rename(columns={"Player Full": "Player Name"}, inplace=True)
    input_df.rename(columns={"Position": "Position Rank"}, inplace=True)
    input_df.rename(columns={"Team": "NFL Team"}, inplace=True)

    # Extract position name
    input_df["Position"] = input_df["Position Rank"].astype(str).str.replace(r'\d+', '', regex=True)

    # Extract position rank number
    input_df["Position Rank"] = input_df["Position Rank"].astype(str).str.replace(r'[A-Za-z]+', '', regex=True)

    # Clean up percentage columns
    input_df["SOS"] = input_df["SOS"].astype(str).str.replace('%', '')
    input_df["Injury Risk"] = input_df["Injury Risk"].astype(str).str.replace('%', '')

    # Filter for relevant positions
    input_df = input_df[input_df["Position"].isin(POSITIONS)]

    # Remove 0 ADP
    input_df = input_df[input_df["ADP"] > 0]

    # Convert adp to absolute order
    input_df["ADP"] = input_df["ADP"].apply(lambda x: int(round((x // 1 - 1) * 12 + (x % 1) * 100)))

    input_df.to_csv(sys.argv[2], index=False)

