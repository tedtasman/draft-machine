from bs4 import BeautifulSoup
import sys

EXTRAS = ["Player", "NO.", "Team"]


def extract_visible_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    texts = []
    for element in soup.find_all(text=True):
        # Skip script, style, head, title, meta, [hidden]
        if element.parent.name in ['script', 'style', 'head', 'title', 'meta', '[document]']:
            continue
        if element.strip():
            texts.append(element.strip().replace(',', '').replace('"', ''))
    return texts

def convert_to_csv(texts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        column = 0
        f.write("1,")
        i = 2
        for text in texts:
            if (
                not text.isnumeric() and
                text not in EXTRAS and
                "Round" not in text
            ):
                if column == 3:
                    f.write(f"{text}\n{i},")
                    i += 1
                    column = 0
                else:
                    f.write(f"{text},")
                    column += 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 html_parser.py input.html output.csv")
        sys.exit(1)

    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        html = f.read()

    visible_texts = extract_visible_text(html)

    convert_to_csv(visible_texts, sys.argv[2])

