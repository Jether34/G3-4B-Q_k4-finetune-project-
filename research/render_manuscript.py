"""Render manuscript.md as a NeurIPS-like, single-column Times PDF.

The attached NeurIPS paper was used only as a visual reference. This renderer
does not reproduce its text, figures, or conference branding.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "manuscript.md"
OUTPUT = ROOT / "manuscript.pdf"
FONT_DIR = Path(r"C:\Windows\Fonts")


def register_fonts() -> tuple[str, str, str, str]:
    files = {
        "TimesNewRoman": FONT_DIR / "times.ttf",
        "TimesNewRoman-Bold": FONT_DIR / "timesbd.ttf",
        "TimesNewRoman-Italic": FONT_DIR / "timesi.ttf",
        "TimesNewRoman-BoldItalic": FONT_DIR / "timesbi.ttf",
    }
    if all(path.exists() for path in files.values()):
        for name, path in files.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily(
            "TimesNewRoman",
            normal="TimesNewRoman",
            bold="TimesNewRoman-Bold",
            italic="TimesNewRoman-Italic",
            boldItalic="TimesNewRoman-BoldItalic",
        )
        return tuple(files.keys())
    return ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic")


REGULAR, BOLD, ITALIC, BOLD_ITALIC = register_fonts()


def inline_markup(text: str) -> str:
    text = escape(text)
    text = re.sub(r"`([^`]+)`", rf'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#204a87">\1</link>',
        text,
    )
    return text


def build_styles():
    return {
        "body": ParagraphStyle(
            "Body",
            fontName=REGULAR,
            fontSize=9.25,
            leading=11.15,
            alignment=TA_JUSTIFY,
            spaceAfter=5.2,
            splitLongWords=True,
            allowWidows=0,
            allowOrphans=0,
        ),
        "abstract": ParagraphStyle(
            "AbstractBody",
            fontName=REGULAR,
            fontSize=9.1,
            leading=10.9,
            alignment=TA_JUSTIFY,
            spaceAfter=2,
            splitLongWords=True,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName=BOLD,
            fontSize=11,
            leading=13,
            alignment=TA_LEFT,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            fontName=BOLD,
            fontSize=9.75,
            leading=11.5,
            alignment=TA_LEFT,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "subsubsection": ParagraphStyle(
            "Subsubsection",
            fontName=BOLD,
            fontSize=9.25,
            leading=11,
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "abstract_title": ParagraphStyle(
            "AbstractTitle",
            fontName=BOLD,
            fontSize=10.5,
            leading=12,
            alignment=TA_CENTER,
            spaceBefore=12,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "author": ParagraphStyle(
            "Author",
            fontName=REGULAR,
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
        ),
        "meta": ParagraphStyle(
            "Meta",
            fontName=ITALIC,
            fontSize=8.2,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
        ),
        "table": ParagraphStyle(
            "TableCell",
            fontName=REGULAR,
            fontSize=7.6,
            leading=9,
            alignment=TA_LEFT,
            splitLongWords=True,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            fontName=BOLD,
            fontSize=7.6,
            leading=9,
            alignment=TA_LEFT,
            splitLongWords=True,
        ),
        "list": ParagraphStyle(
            "List",
            fontName=REGULAR,
            fontSize=9.25,
            leading=11.1,
            alignment=TA_JUSTIFY,
            leftIndent=14,
            firstLineIndent=-14,
            spaceAfter=3,
            splitLongWords=True,
        ),
        "reference": ParagraphStyle(
            "Reference",
            fontName=REGULAR,
            fontSize=8.7,
            leading=10.4,
            alignment=TA_LEFT,
            leftIndent=10,
            firstLineIndent=-10,
            spaceAfter=5,
            splitLongWords=True,
        ),
    }


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(REGULAR, 8)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(letter[0] / 2, 38, str(doc.page))
    canvas.restoreState()


def parse_table(lines: list[str], styles) -> Table:
    parsed = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        if cells:
            is_header = not parsed
            cell_style = styles["table_head"] if is_header else styles["table"]
            parsed.append([Paragraph(inline_markup(cell), cell_style) for cell in cells])
    table = Table(parsed, colWidths=None, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#B0B0B0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8").splitlines()
    if not text or not text[0].startswith("# "):
        raise ValueError("Manuscript must begin with a level-one title")
    title = text[0][2:].strip()
    author = re.sub(r"\*\*|\*", "", text[2]).strip()
    meta = text[4].strip("*")
    abstract_index = next(i for i, line in enumerate(text) if line == "## Abstract")
    body_index = next(i for i in range(abstract_index + 1, len(text)) if text[i].startswith("## 1 "))
    abstract = " ".join(line.strip() for line in text[abstract_index + 1:body_index] if line.strip())
    body_lines = text[body_index:]
    styles = build_styles()

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=1.5 * inch,
        rightMargin=1.5 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
        title=title,
        author=author,
        subject="Preliminary technical report on a 49.4M-parameter defensive cybersecurity Transformer",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="single-column",
    )
    doc.addPageTemplates([PageTemplate(id="paper", frames=[frame], onPage=footer)])

    title_style = ParagraphStyle(
        "PaperTitle",
        fontName=BOLD,
        fontSize=15.2,
        leading=17.4,
        alignment=TA_CENTER,
        spaceBefore=9,
        spaceAfter=10,
    )
    story = [
        Spacer(1, 4),
        Table([[""]], colWidths=[doc.width], rowHeights=[2.4], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.black),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])),
        Paragraph(inline_markup(title), title_style),
        Table([[""]], colWidths=[doc.width], rowHeights=[0.8], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.black),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])),
        Spacer(1, 15),
        Paragraph(inline_markup(author), styles["author"]),
        Spacer(1, 2),
        Paragraph(inline_markup(meta), styles["meta"]),
        Paragraph("Abstract", styles["abstract_title"]),
        Table([[Paragraph(inline_markup(abstract), styles["abstract"])]], colWidths=[doc.width - 16], hAlign="CENTER", style=TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])),
        Spacer(1, 12),
    ]

    i = 0
    in_references = False
    while i < len(body_lines):
        line = body_lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(body_lines) and body_lines[i].strip().startswith("|"):
                table_lines.append(body_lines[i].strip())
                i += 1
            story.extend([Spacer(1, 3), parse_table(table_lines, styles), Spacer(1, 5)])
            continue
        if line.startswith("## "):
            heading = inline_markup(line[3:])
            story.append(Paragraph(heading, styles["section"]))
            if line[3:].strip().casefold() == "references":
                in_references = True
            i += 1
            continue
        if line.startswith("# "):
            heading = inline_markup(line[2:])
            story.append(Paragraph(heading, styles["section"]))
            if line[2:].strip().casefold() == "references":
                in_references = True
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["subsection"]))
            i += 1
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if numbered:
            story.append(Paragraph(f"{numbered.group(1)}.&nbsp;&nbsp;{inline_markup(numbered.group(2))}", styles["list"]))
            i += 1
            continue
        if line.startswith("**") and line.endswith("**"):
            story.append(Paragraph(inline_markup(line), styles["body"]))
            i += 1
            continue
        para = [line]
        i += 1
        while i < len(body_lines):
            nxt = body_lines[i].strip()
            if not nxt or nxt.startswith(("#", "|")) or re.match(r"^\d+\.\s+", nxt):
                break
            para.append(nxt)
            i += 1
        story.append(Paragraph(inline_markup(" ".join(para)), styles["reference"] if in_references else styles["body"]))

    doc.build(story)
    print(f"Rendered {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
