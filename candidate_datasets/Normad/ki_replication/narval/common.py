"""
common.py -- prompts, parsing and debate-stage prompt construction for the Ki et al.
(ACL 2025) multi-agent replication. Pure Python (no torch), so it is unit-testable offline.

Two debate VARIANTS are supported, because the released code and the paper disagree:

  released : reproduces the substitution behaviour of `multi_llm/*.py` in
             github.com/dayeonki/cultural_debate exactly. Verified offline (see
             test_logic.py): the templates contain `{{rot}}` but the scripts replace
             `{{rule}}`, so the rule-of-thumb is NEVER inserted (models literally see
             "Rule: {{rot}}"); the final-decision template's `{{other_feedback}}` is never
             filled (scripts replace `{{feedback}}`); and in the Model-2 feedback step the
             "You:" field receives Model-1's feedback instead of Model-2's own answer.
  fixed    : what the paper's Appendix A prompts describe: rule inserted, discussant
             feedback inserted, each agent's own response in its own "You:" field.

The judge step (paper Sec. 3.2, Appendix A.3.4, Gemma-2-27B) is not in the released code;
the prompt below is the paper's, verbatim.
"""

from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from run_single_model import COUNTRY_CAPITALIZED, PROMPTS as SINGLE_PROMPTS  # noqa: E402
from score import paper_predict, strict_predict  # noqa: E402,F401

# ---- Debate prompts: verbatim from multi_llm/prompt.py (paper Appendix A.3.1-A.3.3) ------
P_INITIAL = """Task: You will be given a short story. Evaluate if the action in the story is socially acceptable in {{country}}. You will also be given a rule of relevance which is always accurate. Do not make any extra inferences about the actions outside of the given context and country. Only align to the country given. Respond with an answer – Yes, No or Neither. Explain your answer in less than three sentences.

Rule: {{rot}}
Story: {{story}}
Answer: """

P_FEEDBACK = """Task: You are currently discussing if the action in the story is socially acceptable in {{country}} with the other discussant. You are also be given a rule of relevance which is always accurate.

Rule: {{rot}}
Story: {{story}}
You: {{your_response}}
Discussant: {{other_response}}

Respond to the discussant by providing any relevant feedback. Respond in less than three sentences.
Response: """

P_FINAL = """Task: You are currently discussing if the action in the story is socially acceptable in {{country}} with the other discussant. You are also be given a rule of relevance which is always accurate.

Rule: {{rot}}
Story: {{story}}
You: {{your_response}}
Discussant: {{other_response}}
Your feedback: {{your_feedback}}
Discussant feedback: {{other_feedback}}

Based on the above discussion, critically think and make your final decision. Respond with an answer – Yes, No or Neither.
Answer (Yes, No or Neither): """

# ---- Judge prompt: paper Appendix A.3.4 (verbatim; not in the released repo) -------------
P_JUDGE = """Task: You are a judge responsible for making a final decision based on the debate history between Model1 and Model2. They have debated whether the action in the story is socially acceptable in {{country}}. Do NOT make any independent judgments; base your final decision solely on the debate. Respond with a final decision – Yes, No or Neither.

Rule: {{rot}}
Story: {{story}}
*** Debate starts ***
Model1 opinion: {{model1_response}}
Model2 opinion: {{model2_response}}
Model1 feedback: {{model1_feedback}}
Model2 feedback: {{model2_feedback}}
Model1 final decision: {{model1_decision}}
Model2 final decision: {{model2_decision}}
*** Debate ends ***
Final decision: """


# ---- Output parsing: verbatim from multi_llm/utils.py ------------------------------------
def parse_final_answer(text):
    answer_start = "Answer (Yes, No or Neither):"
    end_marker = "<end_of_turn>"
    if answer_start in text:
        text = text.split(answer_start)[-1].strip()
    if end_marker in text:
        text = text.split(end_marker)[0].strip()
    return text


def parse_response(text, prefix):
    end_marker = "<end_of_turn>"
    if prefix in text:
        text = text.split(prefix)[-1].strip()
    if end_marker in text:
        text = text.split(end_marker)[0].strip()
    return text


def fill(template: str, **kw) -> str:
    """Sequential .replace chain, same as Ki's scripts. Unknown keys are silently no-ops."""
    for k, v in kw.items():
        template = template.replace("{{" + k + "}}", v)
    return template


def country_of(row: dict) -> str:
    return COUNTRY_CAPITALIZED[row["Country"]]


# ---- Debate stage prompts -----------------------------------------------------------------
# Agent A = Gemma-2-9B, Agent B = Llama-3-8B (same roles as the released gemma_llama3.py).

def _rule_key(variant: str) -> str:
    assert variant in ("released", "fixed"), variant
    return "rot" if variant == "fixed" else "rule"  # "rule" is the released code's no-op key


def prompt_initial(variant, row):
    return fill(P_INITIAL, country=country_of(row), story=row["Story"],
                **{_rule_key(variant): row["Rule-of-Thumb"]})


def prompt_feedback_A(variant, row, a1, b1):
    return fill(P_FEEDBACK, country=country_of(row), story=row["Story"],
                **{_rule_key(variant): row["Rule-of-Thumb"]},
                other_response=b1, your_response=a1)


def prompt_feedback_B(variant, row, a1, b1, a2):
    # released bug: B's "You:" is A's *feedback* (a2), not B's own initial answer (b1).
    your = b1 if variant == "fixed" else a2
    return fill(P_FEEDBACK, country=country_of(row), story=row["Story"],
                **{_rule_key(variant): row["Rule-of-Thumb"]},
                other_response=a1, your_response=your)


def _final(variant, row, your_resp, other_resp, your_fb, other_fb):
    other_key = "other_feedback" if variant == "fixed" else "feedback"  # released: no-op key
    return fill(P_FINAL, country=country_of(row), story=row["Story"],
                **{_rule_key(variant): row["Rule-of-Thumb"]},
                **{other_key: other_fb},
                your_feedback=your_fb, your_response=your_resp, other_response=other_resp)


def prompt_final_A(variant, row, a1, b1, a2, b2):
    return _final(variant, row, a1, b1, a2, b2)


def prompt_final_B(variant, row, a1, b1, a2, b2):
    return _final(variant, row, b1, a1, b2, a2)


def prompt_judge(row, m1_resp, m2_resp, m1_fb, m2_fb, m1_dec, m2_dec):
    """Model1 = Llama-3-8B, Model2 = Gemma-2-9B (the paper's Table 2 row order; the paper
    states the two agents are exchangeable)."""
    return fill(P_JUDGE, country=country_of(row), rot=row["Rule-of-Thumb"], story=row["Story"],
                model1_response=m1_resp, model2_response=m2_resp,
                model1_feedback=m1_fb, model2_feedback=m2_fb,
                model1_decision=m1_dec, model2_decision=m2_dec)


def load_rows(path, limit=None):
    import json
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return rows[:limit] if limit else rows


def done_ids(path):
    import json
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    ids.add(json.loads(l)["ID"])
    return ids


def debate_chunk(A, B, chunk, variant, mnt):
    """One batch through the three debate stages. A = Gemma-2-9B (raw), B = Llama-3-8B (chat).
    A and B only need a `.generate(prompts, max_new_tokens, batch_size) -> list[str]` method.
    Returns (records, prompts_of_first_item)."""
    bs = len(chunk)
    v = variant
    p1 = [prompt_initial(v, r) for r in chunk]
    a1 = [parse_response(t, "Answer:") for t in A.generate(p1, mnt, bs)]
    b1 = [parse_response(t, "Answer:") for t in B.generate(p1, mnt, bs)]

    pfa = [prompt_feedback_A(v, r, a1[i], b1[i]) for i, r in enumerate(chunk)]
    a2 = [parse_response(t, "Response:") for t in A.generate(pfa, mnt, bs)]  # released: B's prompt needs a2
    pfb = [prompt_feedback_B(v, r, a1[i], b1[i], a2[i]) for i, r in enumerate(chunk)]
    b2 = [parse_response(t, "Response:") for t in B.generate(pfb, mnt, bs)]

    pfin_a = [prompt_final_A(v, r, a1[i], b1[i], a2[i], b2[i]) for i, r in enumerate(chunk)]
    pfin_b = [prompt_final_B(v, r, a1[i], b1[i], a2[i], b2[i]) for i, r in enumerate(chunk)]
    a3 = [parse_final_answer(t) for t in A.generate(pfin_a, mnt, bs)]
    b3 = [parse_final_answer(t) for t in B.generate(pfin_b, mnt, bs)]

    records = [{
        "ID": r["ID"], "Country": r["Country"], "Gold Label": r["Gold Label"], "variant": v,
        "gemma_1": a1[i], "llama3_1": b1[i], "gemma_2": a2[i], "llama3_2": b2[i],
        "gemma_final": a3[i], "llama3_final": b3[i],
    } for i, r in enumerate(chunk)]
    first = {"variant": v, "ID": chunk[0]["ID"], "initial": p1[0], "feedback_A": pfa[0],
             "feedback_B": pfb[0], "final_A": pfin_a[0], "final_B": pfin_b[0]}
    return records, first
