"""
Offline tests (no torch, no GPU): prompt substitution for both debate variants, the debate
stage flow with fake generators, judge routing/prompt, and the debate scorer.

    python test_logic.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

from common import (prompt_initial, prompt_feedback_A, prompt_feedback_B, prompt_final_A,
                    prompt_final_B, prompt_judge, debate_chunk, parse_final_answer, parse_response)
from run_judge import agreed_label

ROW = {"ID": 7, "Country": "south_korea", "Gold Label": "yes",
       "Story": "STORY_TEXT", "Rule-of-Thumb": "RULE_TEXT_UNIQUE"}
A1, B1, A2, B2 = "A_INIT", "B_INIT", "A_FB", "B_FB"
PLACEHOLDER = re.compile(r"\{\{\w+\}\}")


def test_released_reproduces_upstream_bugs():
    p = prompt_initial("released", ROW)
    assert "RULE_TEXT_UNIQUE" not in p and "{{rot}}" in p           # rule never inserted
    fa = prompt_feedback_A("released", ROW, A1, B1)
    assert "You: A_INIT" in fa and "Discussant: B_INIT" in fa
    fb = prompt_feedback_B("released", ROW, A1, B1, A2)
    assert "You: A_FB" in fb and "Discussant: A_INIT" in fb          # B's "You" = A's feedback (bug)
    fin_a = prompt_final_A("released", ROW, A1, B1, A2, B2)
    assert "{{other_feedback}}" in fin_a and "Your feedback: A_FB" in fin_a and "B_FB" not in fin_a
    fin_b = prompt_final_B("released", ROW, A1, B1, A2, B2)
    assert "{{other_feedback}}" in fin_b and "Your feedback: B_FB" in fin_b


def test_fixed_matches_paper_prompts():
    for p in (prompt_initial("fixed", ROW), prompt_feedback_A("fixed", ROW, A1, B1),
              prompt_feedback_B("fixed", ROW, A1, B1, A2),
              prompt_final_A("fixed", ROW, A1, B1, A2, B2), prompt_final_B("fixed", ROW, A1, B1, A2, B2)):
        assert not PLACEHOLDER.search(p), p
        assert "Rule: RULE_TEXT_UNIQUE" in p
    fb = prompt_feedback_B("fixed", ROW, A1, B1, A2)
    assert "You: B_INIT" in fb and "Discussant: A_INIT" in fb        # own answer in own slot
    fin_a = prompt_final_A("fixed", ROW, A1, B1, A2, B2)
    assert "Your feedback: A_FB" in fin_a and "Discussant feedback: B_FB" in fin_a
    fin_b = prompt_final_B("fixed", ROW, A1, B1, A2, B2)
    assert "You: B_INIT" in fin_b and "Discussant: A_INIT" in fin_b
    assert "Your feedback: B_FB" in fin_b and "Discussant feedback: A_FB" in fin_b


class FakeGen:
    """Answers from the last non-empty prompt line so stages are distinguishable."""
    def __init__(self, tag):
        self.tag = tag
        self.calls = []

    def generate(self, prompts, mnt, bs):
        self.calls.append(len(prompts))
        outs = []
        for p in prompts:
            if p.rstrip().endswith("Answer (Yes, No or Neither):"):
                outs.append(" Yes <end_of_turn>")
            elif p.rstrip().endswith("Response:"):
                outs.append(f"{self.tag} feedback text")
            else:
                outs.append(f"Answer: Yes because {self.tag}")
        return outs


def test_debate_chunk_flow():
    rows = [ROW, dict(ROW, ID=8)]
    A, B = FakeGen("gemma"), FakeGen("llama")
    recs, first = debate_chunk(A, B, rows, "fixed", 64)
    assert [r["ID"] for r in recs] == [7, 8]
    assert A.calls == [2, 2, 2] and B.calls == [2, 2, 2]              # 3 stages, whole chunk batched
    r = recs[0]
    assert r["gemma_1"] == "Yes because gemma" and r["llama3_1"] == "Yes because llama"
    assert r["gemma_2"] == "gemma feedback text" and r["llama3_2"] == "llama feedback text"
    assert r["gemma_final"] == "Yes" and r["llama3_final"] == "Yes"
    assert "RULE_TEXT_UNIQUE" in first["initial"]
    recs2, first2 = debate_chunk(FakeGen("g"), FakeGen("l"), rows, "released", 64)
    assert "RULE_TEXT_UNIQUE" not in first2["initial"]


def test_parsers_verbatim_behaviour():
    assert parse_final_answer("Answer (Yes, No or Neither): No<end_of_turn>junk") == "No"
    assert parse_response("x Answer: foo bar", "Answer:") == "foo bar"


def test_judge_routing_and_prompt():
    assert agreed_label("Yes", " yes.") == "yes"
    assert agreed_label("No", "Yes") is None
    assert agreed_label("Neither", "Neither, no rule") is None       # 2nd contains 'no' -> 'no' (paper scorer quirk)
    assert agreed_label("garbage", "garbage") is None                # unparseable never counts as agreement
    p = prompt_judge(ROW, "M1R", "M2R", "M1F", "M2F", "M1D", "M2D")
    assert not PLACEHOLDER.search(p)
    assert "Model1 opinion: M1R" in p and "Model2 final decision: M2D" in p and "Rule: RULE_TEXT_UNIQUE" in p


def test_score_debate_end_to_end():
    deb = [
        {"ID": 0, "Country": "x", "Gold Label": "yes", "variant": "fixed", "gemma_final": "Yes", "llama3_final": "Yes",
         "gemma_1": "", "llama3_1": "", "gemma_2": "", "llama3_2": ""},
        {"ID": 1, "Country": "x", "Gold Label": "no", "variant": "fixed", "gemma_final": "No", "llama3_final": "Yes",
         "gemma_1": "", "llama3_1": "", "gemma_2": "", "llama3_2": ""},
        {"ID": 2, "Country": "x", "Gold Label": "neutral", "variant": "fixed", "gemma_final": "Yes", "llama3_final": "No",
         "gemma_1": "", "llama3_1": "", "gemma_2": "", "llama3_2": ""},
    ]
    judge = [{"ID": 1, "judge_generation": "No"}, {"ID": 2, "judge_generation": "Yes"}]
    with tempfile.TemporaryDirectory() as d:
        dp, jp = os.path.join(d, "d.jsonl"), os.path.join(d, "d.jsonl.judge.jsonl")
        open(dp, "w").write("\n".join(json.dumps(x) for x in deb))
        open(jp, "w").write("\n".join(json.dumps(x) for x in judge))
        out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_debate.py"), dp], capture_output=True, text=True, check=True).stdout
    assert "D adjudicated" in out
    # item0 agree+correct; item1 judged 'No' correct; item2 judged 'Yes' wrong -> 2/3 = 66.7
    assert "66.7" in out, out
    assert "judged: 2" in out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all tests passed")
