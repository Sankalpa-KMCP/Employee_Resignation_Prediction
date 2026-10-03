"""Build the requested Word report from its editable Markdown source.

Run with the document-authoring Python runtime that provides python-docx.
The script reads saved evidence only; it never trains models or opens holdout
labels. Render and inspect the Word document separately after any revision.
"""

from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/FINAL_REPORT_DRAFT.md"
OUTPUT = ROOT / "reports/FINAL_REPORT.docx"
PAGE_WIDTH = 6.87


def inline(paragraph, value):
    """Preserve simple emphasis without showing Markdown punctuation in Word."""
    for token in re.split(r"(\*\*.*?\*\*|\*[^*]+\*|`.*?`|https?://\S+)", value):
        if token.startswith("**") and token.endswith("**"):
            paragraph.add_run(token[2:-2]).bold = True
        elif token.startswith("`") and token.endswith("`"):
            run = paragraph.add_run(token[1:-1].replace("/", "/\u200b"))
            run.font.name = "Calibri"
            run.font.size = Pt(10)
        elif token.startswith("*") and token.endswith("*"):
            paragraph.add_run(token[1:-1]).italic = True
        elif token.startswith(("https://", "http://")):
            link = OxmlElement("w:hyperlink")
            link.set(qn("r:id"), paragraph.part.relate_to(token, RT.HYPERLINK, is_external=True))
            run = OxmlElement("w:r")
            text = OxmlElement("w:t")
            text.text = token
            run.append(text)
            link.append(run)
            paragraph._p.append(link)
        else:
            paragraph.add_run(token)


def table(document, records):
    """Use readable borders, widths, padding and repeating headers."""
    headers = records[0]
    rows = [r for r in records[1:] if not all(re.fullmatch(r":?-+:?", x) for x in r)]
    count = len(headers)
    if headers[0] == "Member" and count == 4:
        widths = [.75, 1.60, 2.27, 2.25]
    elif count == 3:
        widths = [1.0, 1.55, 4.32]
    elif count == 5 and headers[1] == "Input":
        widths = [2.02, 1.52, 1.11, 1.11, 1.11]
    elif headers[0] == "Model" and count == 5:
        widths = [1.5, 1.07, 1.07, 1.0, 2.23]
    elif headers[0] == "Selection strategy":
        widths = [1.65, 1.63, 1.18, 1.18, 1.23]
    else:
        widths = [2.35] + [(PAGE_WIDTH - 2.35) / (count - 1)] * (count - 1)
    result = document.add_table(rows=1, cols=count)
    if document.paragraphs:
        document.paragraphs[-1].paragraph_format.keep_with_next = True
        document.paragraphs[-1].paragraph_format.keep_together = True
    result.autofit = False
    for column, width in zip(result.columns, widths):
        column.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        for key, value in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
            element.set(qn(f"w:{key}"), value)
        borders.append(element)
    result._tbl.tblPr.append(borders)
    repeat = OxmlElement("w:tblHeader")
    result.rows[0]._tr.get_or_add_trPr().append(repeat)
    for index, values in enumerate([headers] + rows):
        row = result.rows[0] if index == 0 else result.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for column, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            properties = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for side, amount in (("top", 85), ("bottom", 85), ("left", 100), ("right", 100)):
                margin = OxmlElement(f"w:{side}")
                margin.set(qn("w:w"), str(amount))
                margin.set(qn("w:type"), "dxa")
                margins.append(margin)
            properties.append(margins)
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "19354A" if index == 0 else "F2F4F6" if index % 2 == 0 else "FFFFFF")
            properties.append(shade)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.05
            # These compact comparison tables stay together on one page.
            # Header repetition still supports future longer source tables.
            if len(rows) <= 8:
                paragraph.paragraph_format.keep_with_next = index < len(rows)
            # Numeric result columns are centered; labels remain left aligned.
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if re.fullmatch(r"[\d.,± %]+", value) else WD_ALIGN_PARAGRAPH.LEFT
            inline(paragraph, value)
            for run in paragraph.runs:
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor.from_string("FFFFFF" if index == 0 else "000000")
                run.bold = index == 0
    document.add_paragraph().paragraph_format.space_after = Pt(2)


def build():
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.left_margin = section.right_margin = Inches(.7)
    section.top_margin = section.bottom_margin = Inches(.65)
    section.header_distance = section.footer_distance = Inches(.3)
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    for name, size in (("Title", 23), ("Subtitle", 12), ("Heading 1", 15), ("Heading 2", 12), ("Heading 3", 11), ("Header", 9)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.underline = False
        style.paragraph_format.space_before = Pt(14 if "Heading" in name else 0)
        style.paragraph_format.space_after = Pt(7)
        if "Heading" in name:
            style.paragraph_format.keep_with_next = True
    header = section.header.paragraphs[0]
    header.text = "IT2011   Employee Resignation Prediction"
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Page ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    footer.paragraph_format.space_after = Pt(0)
    for run in footer.runs:
        run.font.size = Pt(9)

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            records = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                records.append([x.strip() for x in lines[index].strip().strip("|").split("|")])
                index += 1
            table(document, records)
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            caption, relative = image_match.groups()
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.keep_with_next = True
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(str(SOURCE.parent / relative), width=Inches(4.5))
            paragraph = document.add_paragraph(caption)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(10)
            for run in paragraph.runs:
                run.italic = True
                run.font.size = Pt(9)
        elif line.startswith("# "):
            document.add_paragraph(line[2:], style="Title")
        elif line.startswith("## "):
            paragraph = document.add_paragraph(line[3:], style="Heading 1")
            if any(line.startswith(f"## {number} ") for number in (1, 4, 9)):
                paragraph.paragraph_format.page_break_before = True
        elif line.startswith("### "):
            document.add_paragraph(line[4:], style="Heading 2")
        else:
            bullet = line.startswith("- ")
            paragraph = document.add_paragraph(style="List Bullet" if bullet else "Normal")
            inline(paragraph, line[2:] if bullet else line)
        index += 1
    document.core_properties.title = "Employee Resignation Prediction Final Group Report"
    document.core_properties.subject = "IT2011 final evaluation development results"
    document.core_properties.author = "[INSERT GROUP MEMBER NAMES]"
    # Word's stock Title style can inherit a blue rule. Strip paragraph
    # borders from both styles and content; table borders are unaffected.
    for element in (document.styles.element, document._element):
        for border in element.xpath(".//w:pBdr"):
            border.getparent().remove(border)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
