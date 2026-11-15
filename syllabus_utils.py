"""File extraction and syllabus parsing."""
import io, re
from pathlib import Path

def split_lines(value):
    if not value:
        return []
    return [x.strip(" •-\t") for x in re.split(r"[\n;]+", value) if x.strip(" •-\t")]

def parse_syllabus(text, filename="Uploaded syllabus"):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    code = ""
    title = Path(filename).stem.replace("_", " ").title()
    desc, outcomes, topics = [], [], []
    section = "desc"
    for line in lines:
        low = line.lower().strip()
        if low.startswith(("course code:", "code:")):
            code = line.split(":", 1)[1].strip()
        elif low.startswith(("course title:", "title:")):
            title = line.split(":", 1)[1].strip()
        elif any(k in low for k in ["course description", "overview", "about the course"]):
            section = "desc"
        elif any(k in low for k in ["learning outcomes", "course outcomes", "outcomes"]):
            section = "outcomes"
        elif any(k in low for k in ["course topics", "syllabus contents", "units", "topics", "course content"]):
            section = "topics"
        elif low.startswith(("unit ", "module ", "week ")):
            section = "topics"
            topics.append(line)
        elif section == "outcomes":
            outcomes.append(line)
        elif section == "topics":
            topics.append(line)
        else:
            desc.append(line)
    return {"code": code, "title": title, "description": " ".join(desc),
            "outcomes": outcomes, "topics": topics}

def read_upload(upload):
    data = upload.getvalue()
    name = upload.name.lower()
    if name.endswith(".pdf"):
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return data.decode("utf-8", errors="ignore")
