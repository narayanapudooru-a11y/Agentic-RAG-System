import tempfile
from docx import Document
from src.ingest import ingest


def simple_embedding(texts):
    # deterministic simple embedding used for tests
    out = []
    for t in texts:
        s = sum(ord(c) for c in t)
        out.append([float(s % 100) / 100.0, float((s // 100) % 100) / 100.0, float((s // 10000) % 100) / 100.0])
    return out


def test_ingest_txt_and_docx(tmp_path):
    txt_path = tmp_path / "a.txt"
    txt_path.write_text("This is a test document. " * 50, encoding="utf-8")

    doc = Document()
    doc.add_paragraph("This is a docx test. " * 60)
    docx_path = tmp_path / "b.docx"
    doc.save(str(docx_path))

    stats = ingest([str(txt_path), str(docx_path)], collection_name="test_ingest", chunk_size=500, overlap=100, embedding_fn=simple_embedding)
    assert stats["num_docs"] == 2
    assert stats["num_chunks"] > 0
