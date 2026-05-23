import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FONT_NAME = "STSong-Light"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Markdown file to PDF.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def make_styles():
    registerFont(UnicodeCIDFont(FONT_NAME))
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "ChineseTitle",
            parent=base["Title"],
            fontName=FONT_NAME,
            fontSize=20,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "ChineseH2",
            parent=base["Heading2"],
            fontName=FONT_NAME,
            fontSize=15,
            leading=22,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h3": ParagraphStyle(
            "ChineseH3",
            parent=base["Heading3"],
            fontName=FONT_NAME,
            fontSize=13,
            leading=20,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ChineseBody",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=10.5,
            leading=17,
            firstLineIndent=0,
            spaceAfter=6,
        ),
        "list": ParagraphStyle(
            "ChineseList",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=10.5,
            leading=17,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "ChineseCode",
            parent=base["Code"],
            fontName=FONT_NAME,
            fontSize=8.5,
            leading=12,
            leftIndent=6,
            rightIndent=6,
            borderColor=colors.HexColor("#dddddd"),
            borderWidth=0.5,
            borderPadding=6,
            backColor=colors.HexColor("#f7f7f7"),
            spaceBefore=4,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "ChineseTable",
            parent=base["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.5,
            leading=12,
        ),
    }
    return styles


def inline_markdown(text):
    text = html.escape(text.strip())
    text = re.sub(r"`([^`]+)`", r"<font color='#333333'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def split_table_row(line):
    row = line.strip().strip("|")
    return [cell.strip() for cell in row.split("|")]


def is_table_separator(line):
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def add_table(story, rows, styles, doc_width):
    if not rows:
        return
    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    data = [
        [Paragraph(inline_markdown(cell), styles["table"]) for cell in row]
        for row in normalized
    ]
    col_width = doc_width / max_cols
    table = Table(data, colWidths=[col_width] * max_cols, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_NAME),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bbbbbb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 6))


def build_story(markdown_text, styles, doc_width):
    lines = markdown_text.splitlines()
    story = []
    i = 0
    in_code = False
    code_lines = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = []
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not line.strip():
            story.append(Spacer(1, 4))
            i += 1
            continue

        if line.strip() == "\\pagebreak":
            story.append(PageBreak())
            i += 1
            continue

        if line.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            table_rows = [split_table_row(line)]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                table_rows.append(split_table_row(lines[i]))
                i += 1
            add_table(story, table_rows, styles, doc_width)
            continue

        if line.startswith("# "):
            story.append(Paragraph(inline_markdown(line[2:]), styles["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markdown(line[3:]), styles["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline_markdown(line[4:]), styles["h3"]))
        elif re.match(r"^\s*[-*]\s+", line):
            content = re.sub(r"^\s*[-*]\s+", "", line)
            story.append(Paragraph("• " + inline_markdown(content), styles["list"]))
        elif re.match(r"^\s*\d+\.\s+", line):
            story.append(Paragraph(inline_markdown(line.strip()), styles["list"]))
        else:
            story.append(Paragraph(inline_markdown(line), styles["body"]))

        i += 1

    if code_lines:
        story.append(Preformatted("\n".join(code_lines), styles["code"]))

    return story


def convert_markdown_to_pdf(input_path, output_path):
    styles = make_styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=input_path.stem,
        author="王书缘",
    )
    text = input_path.read_text(encoding="utf-8")
    story = build_story(text, styles, doc.width)
    doc.build(story)


def main():
    args = parse_args()
    convert_markdown_to_pdf(args.input, args.output)
    print(f"Saved PDF: {args.output}")


if __name__ == "__main__":
    main()
