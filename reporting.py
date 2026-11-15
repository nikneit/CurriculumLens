"""PDF report generation for a completed comparison."""
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def build_pdf(proposed, results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=32, leftMargin=32, topMargin=32, bottomMargin=32)
    styles = getSampleStyleSheet()
    story = [Paragraph("CurriculumLens — Syllabus Comparison Report", styles["Title"]),
             Spacer(1, 10), Paragraph(f"Proposed syllabus: {proposed.get('title','')}", styles["Heading2"]),
             Paragraph(f"Courses compared: {len(results)}", styles["BodyText"]), Spacer(1, 12)]
    rows = [["Course", "Rank", "Similarity", "Topic coverage", "Novelty"]]
    for r in results:
        rows.append([r["course"], f'{r["rank"]:.1f}', f'{r["similarity"]:.0%}',
                     f'{r["topic_coverage"]:.0%}', f'{r["novelty"]:.0%}'])
    table = Table(rows, repeatRows=1, colWidths=[220, 55, 65, 75, 60])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e9eefb")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#24324a")),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#d9dee8")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("PADDING", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    story += [table, Spacer(1, 14), Paragraph("Classifier predictions", styles["Heading2"])]
    for r in results:
        preds = " | ".join(f'{k}: {v["prediction"]} ({v["probability"]:.0%})' for k,v in r["predictions"].items())
        story.append(Paragraph(f'{r["course"]}: {preds}', styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()
