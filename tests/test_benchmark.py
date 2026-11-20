from examshield.benchmark import load_items, render_items_pdf
from examshield.pdfio import open_pdf_bytes, page_text
from examshield.verify.metrics import legibility


def test_item_set_has_valid_answers_and_unique_ids() -> None:
    item_set = load_items()

    assert item_set.version == "v1"
    assert len(item_set.items) == 12
    assert len({item.id for item in item_set.items}) == 12
    for item in item_set.items:
        assert len(item.choices) == 4
        assert 1 <= item.answer <= 4


def test_rendered_pdf_is_byte_identical_across_runs() -> None:
    items = load_items().items

    assert render_items_pdf(items) == render_items_pdf(items)


def test_rendered_pdf_has_one_page_per_item_with_the_item_text() -> None:
    items = load_items().items
    doc = open_pdf_bytes(render_items_pdf(items))

    assert doc.page_count == len(items)
    for page, item in zip(doc, items, strict=True):
        assert legibility(page_text(page), item.text) == 1.0
