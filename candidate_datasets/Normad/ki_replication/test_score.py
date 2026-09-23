"""Sanity tests for score.py. Run: python test_score.py"""
from score import paper_predict, strict_predict


def test_paper_scorer_matches_accuracy_single_py():
    assert paper_predict("\n\nYes \n\n") == "yes"
    assert paper_predict("No") == "no"
    assert paper_predict("Neither") == "neutral"
    # yes takes priority over everything else in the text
    assert paper_predict("No. Yes, wait") == "yes"
    # substring quirk: "Neither" + any later 'no' substring (norm/not/know) is scored "no"
    assert paper_predict("Neither, the norms do not say") == "no"
    assert paper_predict("Neither") == "neutral"


def test_strict_parser_needs_label_as_first_word():
    assert strict_predict("\n\nYes \n\n") == "yes"
    assert strict_predict("Neither, the norms do not say") == "neutral"
    assert strict_predict("No.") == "no"
    assert strict_predict("The answer is yes") == "unparsed"
    assert strict_predict("Nothing here") == "unparsed"  # 'no' must be a whole word
    assert strict_predict("") == "unparsed"


if __name__ == "__main__":
    test_paper_scorer_matches_accuracy_single_py()
    test_strict_parser_needs_label_as_first_word()
    print("all tests passed")
