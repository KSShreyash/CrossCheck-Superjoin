import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


@pytest.fixture
def make_pdf(tmp_path):
    def _make(pages: list[list[str]], name="t.pdf"):
        path = tmp_path / name
        c = canvas.Canvas(str(path), pagesize=A4)
        for lines in pages:
            y = 800
            for line in lines:
                c.drawString(60, y, line)
                y -= 18
            c.showPage()
        c.save()
        return path
    return _make
