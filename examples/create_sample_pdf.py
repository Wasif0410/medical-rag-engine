"""Generate synthetic source material so the quickstart needs no private textbooks."""

import argparse
from pathlib import Path

import pymupdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/sample.pdf"))
    target = parser.parse_args().output
    target.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as doc:
        for title, text in [
            (
                "Cardiology",
                "This synthetic textbook passage discusses heart anatomy and cardiac structure.",
            ),
            ("Assessment", "This synthetic passage discusses assessment of heart rhythms."),
            ("Burns", "This synthetic passage discusses burn wounds and skin assessment."),
        ]:
            page = doc.new_page()
            page.insert_text((72, 72), title, fontsize=18)
            page.insert_textbox(pymupdf.Rect(72, 110, 520, 250), text, fontsize=12)
        doc.set_toc([[1, "Cardiology", 1], [1, "Burns", 3]])
        doc.save(target)
    print(target)


if __name__ == "__main__":
    main()
