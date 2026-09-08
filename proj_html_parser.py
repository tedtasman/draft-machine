import csv
import sys

from bs4 import BeautifulSoup

# data-attribute values, in the order they should appear as columns
STAT_ATTRS = [
    "games_played",
    "adp",
    "player.team.bye",
    "strength_of_schedule",
    "player.sipPlayerProfile.injury_prob",
    "floor_points",
    "consensus_projection",
    "fantasy_points",
    "ceiling_points",
    "dsValue",
]

HEADER = [
    "Rank", "Player Full", "Player Short", "Team", "Position",
    "Games", "ADP", "Bye", "SOS", "Injury Risk",
    "Floor Proj", "Consensus Proj", "DS Proj", "Ceiling Proj", "3D Value",
]


def parse_players(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for i, tbody in enumerate(soup.find_all("tbody", attrs={"data-player-row": True})):
        # rank_span = tbody.select_one(".rank-index div")
        # rank = rank_span.get_text(strip=True) if rank_span else ""
        rank = i + 1

        full_name = tbody.get("data-player-name", "").strip()
        position = tbody.get("data-fantasy-position", "").strip()

        name_tag = tbody.find("player-name")
        first_name = name_tag.get("first-name", "") if name_tag else ""
        last_name = name_tag.get("last-name", "") if name_tag else ""
        short_name = f"{first_name[:1]}. {last_name}".strip() if first_name else full_name

        team_span = tbody.select_one(".player-details-group__team-name")
        team = team_span.get_text(strip=True) if team_span else ""

        row = [rank, full_name, short_name, team, position]

        for attr in STAT_ATTRS:
            cell = tbody.find("td", attrs={"data-attribute": attr})
            value = cell.get("data-value", "").strip() if cell else ""
            row.append(value)

        rows.append(row)

    return rows


def convert_to_csv(rows, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 proj_html_parser.py input.html output.csv")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        html = f.read()

    rows = parse_players(html)
    convert_to_csv(rows, sys.argv[2])