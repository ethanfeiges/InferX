import pytest

mem0_port = pytest.importorskip("inferx.mem0_port", exc_type=ImportError)


def test_reads_only_first_object_when_model_keeps_writing():
    reply = ('{"facts" : ["Boiler inspection on 29 March"]}  Input: I like tea. '
             'Output: {"facts" : ["Likes tea"]}')
    assert mem0_port._json_object(reply) == {"facts": ["Boiler inspection on 29 March"]}


def test_code_fences_and_leading_words():
    assert mem0_port._json_object('Sure:\n```json\n{"memory": []}\n```') == {"memory": []}


def test_skips_broken_braces_before_the_object():
    assert mem0_port._json_object('{oops} then {"facts": ["a"]}') == {"facts": ["a"]}


def test_no_object():
    assert mem0_port._json_object("no json here") == {}
