import sys

from bs4 import BeautifulSoup

POSITIONS = [
    "WR", "TE", "QB", "K", "D/ST"
]

EXTRAS = ["Player", "NO.", "Team"]


def extract_visible_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    texts = []
    for element in soup.find_all(text=True):
        # Skip script, style, head, title, meta, [hidden]
        if element.parent.name in ['script', 'style', 'head', 'title', 'meta', '[document]']: # type: ignore
            continue
        if element.strip(): # type: ignore
            texts.append(element.strip().replace(',', '').replace('"', '')) # type:ignore
    return texts

def convert_to_csv(texts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        column = 0
        for i, text in enumerate(texts):
            if "Dynamic columns based on selected Fantasy Position Scoring and Reserch Depth" in text or "Desktop Sticky Header" in text:
                continue
            elif text == "Player":
                f.write("Player Full,Player Short,Team,Position,")
                column += 4
            elif "Tier" not in text:
                if len(texts) > i + 1 and texts[i + 1] in ["Proj", "Risk"]:
                    f.write(f"{text} ")
                elif column == 14:
                    f.write(f"{text}\n")
                    column = 0
                else:
                    f.write(f"{text},")
                    column += 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 proj_html_parser.py input.html output.csv")
        sys.exit(1)

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        html = f.read()

    visible_texts = extract_visible_text(html)

    convert_to_csv(visible_texts, sys.argv[2])

