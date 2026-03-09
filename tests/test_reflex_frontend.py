from __future__ import annotations

from reflex_frontend.summary_formatting import SummaryItem, parse_summary_sections


def test_parse_summary_sections_extracts_intro_sections_and_items():
    text = """
    Based solely on the provided context:

    **Common Symptoms:** The context does not describe specific symptoms of myopia.

    **Common Treatments:** Current therapeutic approaches include:
    1. **Optical strategies:** Defocus-incorporated lenses and orthokeratology may help slow progression [1][2].
    2. **Pharmacological treatment:** Low-dose atropine is described as a cornerstone therapy [1][2].
    """

    sections = parse_summary_sections(text)

    assert sections[0].variant == "intro"
    assert sections[0].summary == "Based solely on the provided context:"

    assert sections[1].title == "Common Symptoms"
    assert sections[1].summary == "The context does not describe specific symptoms of myopia."

    assert sections[2].title == "Common Treatments"
    assert sections[2].summary == "Current therapeutic approaches include:"
    assert sections[2].bullet_items == [
        SummaryItem(
            title="Optical strategies",
            body="Defocus-incorporated lenses and orthokeratology may help slow progression [1][2].",
        ),
        SummaryItem(
            title="Pharmacological treatment",
            body="Low-dose atropine is described as a cornerstone therapy [1][2].",
        ),
    ]


def test_parse_summary_sections_appends_wrapped_lines_to_previous_item():
    text = """
    **Common Treatments:** Current therapeutic approaches include:
    1. **Optical strategies:** Defocus-incorporated lenses may help.
    They work by modulating peripheral retinal defocus.
    """

    sections = parse_summary_sections(text)

    assert sections[0].bullet_items == [
        SummaryItem(
            title="Optical strategies",
            body="Defocus-incorporated lenses may help. They work by modulating peripheral retinal defocus.",
        )
    ]
