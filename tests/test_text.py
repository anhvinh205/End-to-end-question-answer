import pytest

from nq_qa.data.text import (
    detokenize,
    find_token_span,
    first_html_tag,
    is_html_token,
    split_multi_span,
    strip_html_tokens,
)


@pytest.mark.parametrize("token", ["<P>", "</P>", "<Table>", "</Td>", '<Td_colspan="2">', "<H1>"])
def test_html_tokens_detected(token):
    assert is_html_token(token)


@pytest.mark.parametrize("token", ["P", "<3", "a<b>", "word", "<", ">", "</>", "x/y"])
def test_plain_tokens_not_html(token):
    assert not is_html_token(token)


def test_first_html_tag():
    assert first_html_tag("<P> Hello </P>") == "P"
    assert first_html_tag("  <Table> <Tr> x") == "Table"
    assert first_html_tag("No markup here") == "plain"


def test_strip_html_tokens():
    assert strip_html_tokens("<P> A common example </P>") == ["A", "common", "example"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("firm 's customers", "firm's customers"),
        ("Brady , the player .", "Brady, the player."),
        ("do n't stop", "don't stop"),
        ("film `` The Mother '' was", 'film "The Mother" was'),
        ("an Oscar - winning film", "an Oscar-winning film"),
        ("singer / songwriter", "singer/songwriter"),
        ("airs 7 : 00 to 9 : 00 a.m.", "airs 7:00 to 9:00 a.m."),
        ("Note : see ( below ) .", "Note: see (below)."),
        ("costs $ 100 or 50 %", "costs $100 or 50%"),
        ('He said " hello " today', 'He said "hello" today'),
    ],
)
def test_detokenize_text(raw, expected):
    text, _ = detokenize(raw.split())
    assert text == expected


def test_detokenize_offsets_point_at_each_token():
    tokens = ["Tom", "Brady", "'s", "wins", ",", "220", "."]
    text, spans = detokenize(tokens)
    assert text == "Tom Brady's wins, 220."
    assert [text[s:e] for s, e in spans] == ["Tom", "Brady", "'s", "wins", ",", "220", "."]


def test_detokenize_empty():
    assert detokenize([]) == ("", [])


def test_find_token_span():
    hay = ["the", "Tom", "Brady", "holds", "the", "record"]
    assert find_token_span(hay, ["Tom", "Brady"]) == 1
    assert find_token_span(hay, ["tom", "brady"]) == -1
    assert find_token_span(hay, ["tom", "brady"], ignore_case=True) == 1
    assert find_token_span(hay, []) == -1
    assert find_token_span(hay, ["absent"]) == -1
    assert find_token_span(["a"], ["a", "b"]) == -1


def test_split_multi_span_only_splits_on_joins():
    # commas inside a span are tokenised as " , ", joins are ", "
    assert split_multi_span("Guy , Gunnar and Garth, Michael , Daniel") == [
        "Guy , Gunnar and Garth",
        "Michael , Daniel",
    ]
    assert split_multi_span("United States, Canada, Mexico") == [
        "United States",
        "Canada",
        "Mexico",
    ]
    assert split_multi_span("a newsletter sent to firms") == ["a newsletter sent to firms"]
    assert split_multi_span("") == []
