from __future__ import annotations

import re
from pydantic import BaseModel, Field


class SummaryItem(BaseModel):
    title: str = ""
    body: str = ""


class SummarySection(BaseModel):
    title: str = ""
    summary: str = ""
    bullet_items: list[SummaryItem] = Field(default_factory=list)
    variant: str = "section"


def clean_summary_text(value: str) -> str:
    text = value.strip().replace("**", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_summary_line(line: str) -> dict[str, str]:
    raw = line.strip()
    if not raw:
        return {"kind": "empty", "title": "", "body": ""}

    is_list_item = bool(re.match(r"^(\d+\.|[-*])\s+", raw))
    content = re.sub(r"^(\d+\.|[-*])\s+", "", raw)
    match = re.match(r"^\*\*(.+?)\*\*:?\s*(.*)$", content)
    if match:
        return {
            "kind": "item" if is_list_item else "label",
            "title": clean_summary_text(match.group(1)).rstrip(":"),
            "body": clean_summary_text(match.group(2)),
        }

    return {
        "kind": "item" if is_list_item else "text",
        "title": "",
        "body": clean_summary_text(content if is_list_item else raw),
    }


def parse_summary_sections(text: str) -> list[SummarySection]:
    if not text.strip():
        return []

    sections: list[SummarySection] = []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    for index, block in enumerate(blocks):
        section = SummarySection()
        for line in [item.strip() for item in block.splitlines() if item.strip()]:
            parsed = parse_summary_line(line)
            kind = parsed["kind"]
            body = parsed["body"]
            if kind == "empty":
                continue
            if kind == "label":
                section.title = parsed["title"]
                section.summary = body
                continue
            if kind == "item":
                section.bullet_items.append(SummaryItem(title=parsed["title"], body=body))
                continue
            if section.bullet_items:
                last_item = section.bullet_items[-1]
                last_item.body = " ".join(filter(None, [last_item.body, body])).strip()
            else:
                section.summary = " ".join(filter(None, [section.summary, body])).strip()

        if not section.title:
            section.variant = "intro" if index == 0 else "note"
        sections.append(section)
    return sections
