from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import reflex as rx

from app_config import load_config
from reflex_frontend.summary_formatting import SummaryItem, SummarySection, parse_summary_sections
from runtime_logging import configure_logging


UPLOAD_ID = "retinal-upload"
BG = "#f2ede3"
PANEL = "rgba(255,250,242,0.9)"
BORDER = "1px solid rgba(30,42,40,0.12)"
TEXT = "#1e2a28"
MUTED = "#66756f"
ACCENT = "#0f766e"
WARN = "#c2410c"

SURFACE = {
    "background": PANEL,
    "border": BORDER,
    "border_radius": "24px",
    "box_shadow": "0 18px 60px rgba(30,42,40,0.08)",
}

CARD = {
    "background": "rgba(255,255,255,0.84)",
    "border": "1px solid rgba(30,42,40,0.08)",
    "border_radius": "20px",
    "padding": "18px",
    "width": "100%",
}


def _api_base_url() -> str:
    return load_config().effective_frontend_api_url.rstrip("/")


configure_logging()
logger = logging.getLogger(__name__)


def _parse_json_response(response: httpx.Response) -> dict[str, Any]:
    logger.info(
        "Frontend received response status=%s content_type=%s url=%s",
        response.status_code,
        response.headers.get("content-type"),
        response.request.url,
    )
    try:
        payload = response.json()
    except json.JSONDecodeError:
        body = response.text.strip()
        snippet = body[:300] if body else "<empty response body>"
        raise RuntimeError(
            f"Backend returned a non-JSON response (status {response.status_code}): {snippet}"
        ) from None

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Backend returned an unexpected response payload (status {response.status_code})."
        )
    return payload


async def _check_backend_health(client: httpx.AsyncClient, base_url: str) -> None:
    health_url = f"{base_url}/api/health"
    logger.info("Checking backend health at %s", health_url)
    response = await client.get(health_url)
    payload = _parse_json_response(response)
    if response.status_code >= 400:
        raise RuntimeError(payload.get("detail", f"Backend health check failed with status {response.status_code}."))
    if not payload.get("ready", False):
        raise RuntimeError(
            f"Backend is not ready. startup_error={payload.get('startup_error')!r}"
        )
    logger.info(
        "Backend ready artifact_dir=%s chunk_count=%s",
        payload.get("artifact_dir"),
        payload.get("chunk_count"),
    )


class DemoState(rx.State):
    selected_condition: str = ""
    image_filename: str = ""
    image_preview_src: str = ""
    prediction: dict[str, Any] = {}
    initial_summary_text: str = ""
    followup_question: str = ""
    followup_answer_text: str = ""
    citations: list[dict[str, Any]] = []
    warnings: list[str] = []
    timings: dict[str, float] = {}
    raw_followup_payload: dict[str, Any] = {}
    error_message: str = ""
    is_loading: bool = False
    is_followup_loading: bool = False

    @rx.var
    def has_result(self) -> bool:
        return bool(self.prediction)

    @rx.var
    def prediction_label(self) -> str:
        return str(self.prediction.get("label", "Unknown"))

    @rx.var
    def confidence_text(self) -> str:
        return f"{float(self.prediction.get('confidence', 0.0)) * 100:.1f}%"

    @rx.var
    def probability_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in self.prediction.get("top_predictions", []):
            probability = float(item.get("probability", 0.0))
            rows.append(
                {
                    "label": str(item.get("label", "unknown")),
                    "percent": f"{probability * 100:.1f}%",
                    "width": f"{max(4.0, probability * 100):.2f}%",
                }
            )
        return rows

    @rx.var
    def timing_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        mapping = (
            ("classification_ms", "Classification"),
            ("rag_ms", "Initial RAG"),
        )
        for key, label in mapping:
            if key in self.timings:
                rows.append({"label": label, "value": f"{float(self.timings[key]):.0f} ms"})
        followup_rag = self.raw_followup_payload.get("timings", {}).get("rag_ms")
        if followup_rag is not None:
            rows.append({"label": "Follow-up RAG", "value": f"{float(followup_rag):.0f} ms"})
        return rows

    @rx.var
    def citation_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for citation in self.citations:
            metadata = citation.get("metadata", {}) or {}
            rows.append(
                {
                    "title": str(metadata.get("title", "Untitled source")),
                    "source": str(metadata.get("source", "Unknown")),
                    "year": str(metadata.get("year", "Year unknown")),
                    "score": f"{float(citation.get('score', 0.0)):.3f}",
                    "url": str(metadata.get("url", "") or ""),
                    "text": str(citation.get("text", "")),
                }
            )
        return rows

    @rx.var
    def summary_sections(self) -> list[SummarySection]:
        return parse_summary_sections(self.initial_summary_text)

    @rx.var
    def followup_blocks(self) -> list[str]:
        return [block.strip() for block in self.followup_answer_text.split("\n\n") if block.strip()]

    @rx.event
    def set_followup_question(self, value: str) -> None:
        self.followup_question = value

    @rx.event
    async def submit_initial(self, files: list[rx.UploadFile]) -> None:
        if not files:
            self.error_message = "Select an image first."
            return

        upload = files[0]
        self.is_loading = True
        self.error_message = ""
        self.followup_answer_text = ""
        self.raw_followup_payload = {}
        self.warnings = []

        try:
            payload = await upload.read()
            if not payload:
                raise ValueError("Uploaded file is empty.")

            suffix = Path(upload.filename or "upload.png").suffix or ".png"
            stored_name = f"{uuid4().hex}{suffix.lower()}"
            stored_path = rx.get_upload_dir() / stored_name
            stored_path.parent.mkdir(parents=True, exist_ok=True)
            stored_path.write_bytes(payload)
            self.image_filename = stored_name
            mime_type = upload.content_type or "image/png"
            encoded_preview = base64.b64encode(payload).decode("ascii")
            self.image_preview_src = f"data:{mime_type};base64,{encoded_preview}"
            base_url = _api_base_url()
            logger.info(
                "Submitting initial upload filename=%s size=%s api_base=%s",
                upload.filename or stored_name,
                len(payload),
                base_url,
            )

            async with httpx.AsyncClient(timeout=90.0) as client:
                await _check_backend_health(client, base_url)
                response = await client.post(
                    f"{base_url}/api/answer",
                    files={"image": (upload.filename or stored_name, payload, upload.content_type or "image/png")},
                )
            data = _parse_json_response(response)
            if response.status_code >= 400:
                raise RuntimeError(data.get("detail", "Initial analysis failed."))

            self.prediction = data.get("prediction", {})
            self.selected_condition = str(self.prediction.get("label", ""))
            self.initial_summary_text = data.get("initial_summary", {}).get("text", "")
            self.citations = data.get("citations", [])
            self.warnings = data.get("warnings", [])
            self.timings = data.get("timings", {})
        except Exception as exc:
            self.error_message = str(exc)
        finally:
            self.is_loading = False

    @rx.event
    async def submit_followup(self) -> None:
        if not self.selected_condition:
            self.error_message = "Run the initial analysis first."
            return
        if not self.followup_question.strip():
            self.error_message = "Enter a follow-up question."
            return

        self.is_followup_loading = True
        self.error_message = ""
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                base_url = _api_base_url()
                logger.info(
                    "Submitting follow-up condition=%s api_base=%s",
                    self.selected_condition,
                    base_url,
                )
                await _check_backend_health(client, base_url)
                response = await client.post(
                    f"{base_url}/api/followup",
                    json={
                        "condition": self.selected_condition,
                        "question": self.followup_question.strip(),
                    },
                )
            data = _parse_json_response(response)
            if response.status_code >= 400:
                raise RuntimeError(data.get("detail", "Follow-up failed."))

            self.raw_followup_payload = data
            self.followup_answer_text = data.get("answer", {}).get("text", "")
            if data.get("citations"):
                self.citations = data["citations"]
            self.warnings = list(dict.fromkeys([*self.warnings, *data.get("warnings", [])]))
        except Exception as exc:
            self.error_message = str(exc)
        finally:
            self.is_followup_loading = False


def section_card(title: str, *children: rx.Component, **props) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text(title, size="2", weight="bold", text_transform="uppercase", letter_spacing="0.08em", color=MUTED),
            *children,
            spacing="3",
            align="start",
            width="100%",
        ),
        **(CARD | props),
    )


def probability_row(item: dict[str, str]) -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text(item["label"], color=TEXT, weight="medium"),
            rx.spacer(),
            rx.text(item["percent"], color=ACCENT, weight="bold"),
            width="100%",
        ),
        rx.box(
            rx.box(
                width=item["width"],
                height="100%",
                border_radius="999px",
                background="linear-gradient(90deg, #0f766e, #155e75)",
            ),
            width="100%",
            height="10px",
            background="rgba(30,42,40,0.09)",
            border_radius="999px",
            overflow="hidden",
        ),
        spacing="2",
        width="100%",
    )


def text_blocks(blocks: list[str]) -> rx.Component:
    return rx.vstack(
        rx.foreach(
            blocks,
            lambda block: rx.text(block, white_space="pre-wrap", line_height="1.75", color=TEXT, size="3"),
        ),
        spacing="3",
        width="100%",
        align="start",
    )


def summary_item(item: SummaryItem) -> rx.Component:
    return rx.hstack(
        rx.box(
            width="8px",
            height="8px",
            border_radius="999px",
            background="linear-gradient(135deg, #0f766e, #155e75)",
            margin_top="9px",
            flex_shrink="0",
        ),
        rx.vstack(
            rx.cond(
                item.title != "",
                rx.text(item.title, size="2", weight="bold", color=TEXT, letter_spacing="0.01em"),
            ),
            rx.text(item.body, color=TEXT, size="3", line_height="1.75"),
            spacing="1",
            align="start",
            width="100%",
        ),
        spacing="3",
        width="100%",
        align="start",
    )


def summary_section(section: SummarySection) -> rx.Component:
    return rx.cond(
        section.variant == "intro",
        rx.box(
            rx.vstack(
                rx.badge(
                    "Grounded overview",
                    radius="full",
                    style={
                        "background": "rgba(15,118,110,0.12)",
                        "color": ACCENT,
                        "padding": "6px 10px",
                    },
                ),
                rx.text(section.summary, color=TEXT, size="4", line_height="1.9", weight="medium"),
                spacing="3",
                align="start",
                width="100%",
            ),
            width="100%",
            padding="20px",
            border_radius="20px",
            background=(
                "linear-gradient(135deg, rgba(15,118,110,0.12), rgba(21,94,117,0.06)), "
                "rgba(255,255,255,0.82)"
            ),
            border="1px solid rgba(15,118,110,0.14)",
        ),
        rx.box(
            rx.vstack(
                rx.cond(
                    section.title != "",
                    rx.badge(
                        section.title,
                        radius="full",
                        style={
                            "background": "rgba(30,42,40,0.06)",
                            "color": TEXT,
                            "padding": "6px 10px",
                        },
                    ),
                ),
                rx.cond(
                    section.summary != "",
                    rx.text(section.summary, color=MUTED, size="3", line_height="1.75"),
                ),
                rx.cond(
                    section.bullet_items.length() > 0,
                    rx.vstack(
                        rx.foreach(section.bullet_items, summary_item),
                        spacing="3",
                        width="100%",
                        align="start",
                    ),
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            width="100%",
            padding="18px",
            border_radius="18px",
            background="rgba(255,255,255,0.74)",
            border="1px solid rgba(30,42,40,0.08)",
            box_shadow="inset 0 1px 0 rgba(255,255,255,0.65)",
        ),
    )


def literature_summary_card() -> rx.Component:
    return section_card(
        "Literature summary",
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.text("Evidence brief", size="2", weight="bold", color=ACCENT, letter_spacing="0.08em"),
                    rx.text(
                        "Structured from the retrieved context so the answer scans faster.",
                        color=MUTED,
                        size="3",
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.spacer(),
                rx.badge(
                    DemoState.prediction_label,
                    radius="full",
                    style={
                        "background": "rgba(15,118,110,0.12)",
                        "color": ACCENT,
                        "padding": "6px 10px",
                    },
                ),
                width="100%",
                align="center",
            ),
            rx.foreach(DemoState.summary_sections, summary_section),
            spacing="4",
            width="100%",
            align="start",
        ),
        flex="1",
        background="linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,244,237,0.94))",
        border="1px solid rgba(15,118,110,0.12)",
        box_shadow="0 18px 60px rgba(30,42,40,0.08)",
    )


def citation_item(citation: dict[str, str]) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(citation["title"], weight="bold", color=TEXT, flex="1"),
                rx.badge(citation["source"], color_scheme="teal"),
                width="100%",
                align="center",
            ),
            rx.hstack(
                rx.text(citation["year"], color=MUTED, size="2"),
                rx.text(f"Score {citation['score']}", color=MUTED, size="2"),
                rx.cond(
                    citation["url"] != "",
                    rx.link("Open source", href=citation["url"], is_external=True, color=ACCENT, size="2"),
                ),
                spacing="4",
                wrap="wrap",
                width="100%",
            ),
            rx.text(citation["text"], color=TEXT, line_height="1.7", size="3"),
            spacing="3",
            width="100%",
            align="start",
        ),
        border_top="1px solid rgba(30,42,40,0.08)",
        padding_top="14px",
        width="100%",
    )


def controls_panel() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text("Internal demo", size="2", color=ACCENT, weight="bold"),
            rx.heading("Retinal Image + RAG Demo", size="8", color=TEXT),
            rx.text(
                "Upload a retinal image to get a condition summary first. Use the follow-up box for deeper questions after the result appears.",
                color=MUTED,
                size="3",
            ),
            rx.upload(
                rx.vstack(
                    rx.text("Retinal image", weight="bold", color=TEXT),
                    rx.text(
                        rx.cond(
                            rx.selected_files(UPLOAD_ID).length() > 0,
                            rx.selected_files(UPLOAD_ID)[0],
                            "Drop an image here or click to upload",
                        ),
                        color=MUTED,
                    ),
                    spacing="2",
                    width="100%",
                    align="start",
                ),
                id=UPLOAD_ID,
                accept={"image/*": [".png", ".jpg", ".jpeg", ".webp"]},
                max_files=1,
                border="2px dashed rgba(15,118,110,0.35)",
                border_radius="20px",
                padding="18px",
                width="100%",
            ),
            rx.button(
                "Generate condition summary",
                width="100%",
                border_radius="999px",
                background="linear-gradient(135deg, #0f766e, #155e75)",
                color="white",
                loading=DemoState.is_loading,
                on_click=[
                    DemoState.submit_initial(rx.upload_files(upload_id=UPLOAD_ID)),
                    rx.clear_selected_files(UPLOAD_ID),
                ],
            ),
            rx.cond(
                DemoState.has_result,
                rx.vstack(
                    rx.text("Follow-up question", weight="bold", color=TEXT),
                    rx.text_area(
                        placeholder="Ask a follow-up question about the predicted condition.",
                        value=DemoState.followup_question,
                        on_change=DemoState.set_followup_question,
                        min_height="132px",
                    ),
                    rx.button(
                        "Ask follow-up question",
                        width="100%",
                        border_radius="999px",
                        background="linear-gradient(135deg, #c2410c, #9a3412)",
                        color="white",
                        loading=DemoState.is_followup_loading,
                        on_click=DemoState.submit_followup,
                    ),
                    spacing="3",
                    width="100%",
                    align="start",
                ),
            ),
            rx.cond(
                DemoState.error_message != "",
                rx.callout(DemoState.error_message, icon="triangle_alert", color_scheme="red", width="100%"),
            ),
            rx.text(
                "For literature assistance only. This is not a diagnostic or treatment tool.",
                color=WARN,
                size="2",
            ),
            spacing="4",
            width="100%",
            align="start",
        ),
        **(SURFACE | {"padding": "24px", "position": "sticky", "top": "20px"}),
    )


def result_canvas() -> rx.Component:
    return rx.vstack(
        rx.grid(
            rx.box(
                rx.cond(
                    DemoState.image_preview_src != "",
                    rx.image(
                        src=DemoState.image_preview_src,
                        alt="Uploaded retinal image preview",
                        width="100%",
                        height="100%",
                        object_fit="cover",
                    ),
                    rx.center(
                        rx.text("Uploaded image stays visible here.", color=MUTED),
                        width="100%",
                        height="100%",
                    ),
                ),
                min_height="360px",
                width="100%",
                overflow="hidden",
                border_radius="24px",
                border=BORDER,
                background="linear-gradient(160deg, rgba(15,118,110,0.14), rgba(194,65,12,0.12))",
            ),
            rx.cond(
                DemoState.has_result,
                rx.vstack(
                    section_card(
                        "Prediction",
                        rx.hstack(
                            rx.heading(DemoState.prediction_label, size="8", color=TEXT),
                            rx.spacer(),
                            rx.badge(DemoState.confidence_text, color_scheme="teal", radius="full"),
                            width="100%",
                            align="center",
                        ),
                        rx.vstack(rx.foreach(DemoState.probability_rows, probability_row), spacing="3", width="100%"),
                    ),
                    literature_summary_card(),
                    spacing="4",
                    width="100%",
                    align="start",
                ),
                section_card(
                    "Ready",
                    rx.text(
                        "Upload an image to begin. The right side expands into a summary canvas with prediction, literature answer, timings, and evidence.",
                        color=MUTED,
                    ),
                ),
            ),
            columns=rx.breakpoints(initial="1", md="2"),
            spacing="4",
            width="100%",
        ),
        rx.cond(
            DemoState.followup_answer_text != "",
            section_card(
                "Follow-up answer",
                rx.text(DemoState.followup_question, color=MUTED, weight="medium"),
                text_blocks(DemoState.followup_blocks),
            ),
        ),
        rx.cond(
            DemoState.warnings.length() > 0,
            section_card(
                "Warnings",
                rx.vstack(
                    rx.foreach(DemoState.warnings, lambda item: rx.text(item, color=WARN, line_height="1.6")),
                    spacing="2",
                    width="100%",
                    align="start",
                ),
                background="rgba(255,247,237,0.92)",
                border="1px solid rgba(194,65,12,0.15)",
            ),
        ),
        rx.cond(
            DemoState.timing_rows.length() > 0,
            rx.grid(
                rx.foreach(
                    DemoState.timing_rows,
                    lambda item: section_card(item["label"], rx.text(item["value"], size="6", weight="bold", color=TEXT)),
                ),
                columns=rx.breakpoints(initial="1", sm="3"),
                spacing="4",
                width="100%",
            ),
        ),
        section_card(
            "Evidence",
            rx.cond(
                DemoState.citation_rows.length() > 0,
                rx.vstack(
                    rx.foreach(DemoState.citation_rows, citation_item),
                    width="100%",
                    spacing="0",
                    align="start",
                ),
                rx.text("No citations returned yet.", color=MUTED),
            ),
        ),
        spacing="4",
        width="100%",
        align="start",
    )


def index() -> rx.Component:
    return rx.box(
        rx.grid(
            controls_panel(),
            result_canvas(),
            columns=rx.breakpoints(initial="1", xl="340px 1fr"),
            spacing="6",
            width="100%",
            max_width="1500px",
            margin="0 auto",
        ),
        min_height="100vh",
        padding=rx.breakpoints(initial="20px", md="28px", xl="36px"),
        background=(
            "radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 30%), "
            "radial-gradient(circle at bottom right, rgba(194,65,12,0.14), transparent 28%), "
            f"{BG}"
        ),
    )


app = rx.App()
app.add_page(index, route="/", title="Retinal Demo")
