from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _prepare_blocks(text: str, wrap_width: int = 88) -> list[dict[str, str]]:
    blocks: list[dict] = []
    pending_table: list[list[str]] = []

    def flush_table() -> None:
        nonlocal pending_table
        if pending_table:
            blocks.append({"kind": "table", "rows": pending_table})
            blocks.append({"kind": "spacer", "text": ""})
            pending_table = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            flush_table()
            blocks.append({"kind": "spacer", "text": ""})
            continue

        if line.startswith("#"):
            flush_table()
            level = len(line) - len(line.lstrip("#"))
            heading = line.lstrip("#").strip()
            blocks.append({"kind": f"h{level}", "text": heading})
            blocks.append({"kind": "spacer", "text": ""})
            continue

        if line.startswith("|"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if parts and not all(set(part) <= {"-", ":"} for part in parts):
                pending_table.append(parts)
            continue

        flush_table()

        if line.startswith("- "):
            wrapped = textwrap.wrap(
                line[2:].strip(),
                width=wrap_width - 4,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for idx, item in enumerate(wrapped or [""]):
                prefix = u"\u2022 " if idx == 0 else "  "
                blocks.append({"kind": "bullet", "text": f"{prefix}{item}"})
            continue

        if line[:3] in {"1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. "}:
            marker = line[:3]
            wrapped = textwrap.wrap(
                line[3:].strip(),
                width=wrap_width - 4,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for idx, item in enumerate(wrapped or [""]):
                prefix = marker if idx == 0 else "   "
                blocks.append({"kind": "bullet", "text": f"{prefix}{item}"})
            continue

        wrapped = textwrap.wrap(
            line,
            width=wrap_width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for item in wrapped or [""]:
            blocks.append({"kind": "body", "text": item})
    flush_table()
    return blocks


def render_pdf(input_path: Path, output_path: Path, title: str | None = None) -> None:
    text = input_path.read_text(encoding="utf-8")
    blocks = _prepare_blocks(text)

    pages: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    remaining = 0.80

    def block_height(kind: str) -> float:
        if kind == "h1":
            return 0.040
        if kind == "h2":
            return 0.032
        if kind == "h3":
            return 0.027
        if kind == "spacer":
            return 0.012
        return 0.021

    def estimated_height(block: dict) -> float:
        if block["kind"] != "table":
            return block_height(block["kind"])
        rows = block.get("rows", [])
        row_count = max(len(rows), 1)
        return 0.035 + row_count * 0.045

    for block in blocks:
        h = estimated_height(block)
        if current and remaining - h < 0.08:
            pages.append(current)
            current = []
            remaining = 0.84
        current.append(block)
        remaining -= h
    if current or not pages:
        pages.append(current)

    with PdfPages(output_path) as pdf:
        for idx, page_lines in enumerate(pages):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("#f8fafc")

            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")

            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.90, 0.90, facecolor="white", edgecolor="#dbe3ee", linewidth=1.2))

            if idx == 0 and title:
                ax.add_patch(plt.Rectangle((0.05, 0.905), 0.90, 0.07, facecolor="#1d4ed8", edgecolor="#1d4ed8"))
                fig.text(0.08, 0.947, title, fontsize=20, fontweight="bold", color="white", va="center", ha="left")
                fig.text(0.08, 0.915, "Fairness audit comparison summary", fontsize=10.5, color="#dbeafe", va="center", ha="left")
                y = 0.875
            else:
                fig.text(0.08, 0.94, title or input_path.stem.replace("_", " ").title(), fontsize=12, fontweight="bold", color="#1f2937", va="top", ha="left")
                y = 0.90

            for block in page_lines:
                kind = block["kind"]
                if kind == "h1":
                    line = block["text"]
                    fig.text(0.08, y, line, fontsize=15, fontweight="bold", color="#0f172a", va="top", ha="left")
                    y -= 0.040
                elif kind == "h2":
                    line = block["text"]
                    fig.text(0.08, y, line, fontsize=13, fontweight="bold", color="#1d4ed8", va="top", ha="left")
                    y -= 0.032
                elif kind == "h3":
                    line = block["text"]
                    fig.text(0.08, y, line, fontsize=11.5, fontweight="bold", color="#334155", va="top", ha="left")
                    y -= 0.027
                elif kind == "table":
                    rows = block.get("rows", [])
                    if not rows:
                        continue
                    col_count = max(len(r) for r in rows)
                    normalized_rows = [r + [""] * (col_count - len(r)) for r in rows]
                    table_height = 0.028 + len(normalized_rows) * 0.040
                    table_ax = fig.add_axes([0.075, y - table_height + 0.008, 0.85, table_height - 0.005])
                    table_ax.axis("off")
                    table = table_ax.table(
                        cellText=normalized_rows[1:] if len(normalized_rows) > 1 else normalized_rows,
                        colLabels=normalized_rows[0],
                        loc="upper left",
                        cellLoc="left",
                        colLoc="left",
                        bbox=[0, 0, 1, 1],
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(8.6)
                    table.scale(1, 1.18)
                    for (row, col), cell in table.get_celld().items():
                        cell.set_edgecolor("#cbd5e1")
                        cell.set_linewidth(0.8)
                        cell.PAD = 0.03
                        cell.get_text().set_wrap(True)
                        if row == 0:
                            cell.set_facecolor("#dbeafe")
                            cell.get_text().set_fontweight("bold")
                            cell.get_text().set_color("#0f172a")
                        else:
                            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
                            cell.get_text().set_color("#1f2937")
                    y -= table_height
                elif kind == "bullet":
                    line = block["text"]
                    fig.text(0.10, y, line, fontsize=10.2, color="#1f2937", va="top", ha="left")
                    y -= 0.021
                elif kind == "spacer":
                    y -= 0.012
                else:
                    line = block["text"]
                    fig.text(0.08, y, line, fontsize=10.4, color="#1f2937", va="top", ha="left")
                    y -= 0.021

            fig.text(0.08, 0.035, "AI Fairness Dashboard", fontsize=8.5, color="#64748b", ha="left", va="bottom")
            fig.text(0.92, 0.035, f"Page {idx + 1} of {len(pages)}", fontsize=8.5, color="#64748b", ha="right", va="bottom")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a markdown/text report to PDF.")
    parser.add_argument("--input", required=True, help="Input markdown/text file")
    parser.add_argument("--output", required=True, help="Output PDF path")
    parser.add_argument("--title", default="", help="Optional title shown on first page")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_pdf(input_path, output_path, title=args.title or None)


if __name__ == "__main__":
    main()
