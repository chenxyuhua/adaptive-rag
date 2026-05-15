from adaptive_rag.decision.parsing import extract_json_object


def test_extract_json_object_direct():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_embedded():
    text = 'Here you go:\n```\n{"need_retrieval": false, "confidence": 0.2}\n```'
    obj = extract_json_object(text)
    assert obj is not None
    assert obj["need_retrieval"] is False
