"""What the judge spec tests and the judge run tests share: a template, a spec, a trace, a
label and an answer line that gives the verdict first."""

from axial.analysis.dataset import Label
from axial.core.traces import Trace
from axial.judge.spec import JudgeSpec

TEMPLATE = "Does this trace show the failure?\n{few_shot}\nINPUT: {input}\nOUTPUT: {output}"


def spec(**kw) -> JudgeSpec:
    base = {
        "mode": "merged",
        "prompt_template": TEMPLATE,
        "model": "fake",
        "few_shot_ids": [],
        "version": 1,
    }
    base.update(kw)
    return JudgeSpec(**base)


def trace(i: int) -> Trace:
    return Trace(trace_id=f"t{i}", input=f"in{i}", output=f"out{i}")


def label(i: int, value: bool) -> Label:
    return Label(trace_id=f"t{i}", label=value, critique="because", annotator="andres")


# The answer line axial's default template ended with before it asked for the reason
# first. A judge.yaml written then keeps its own copy.
VERDICT_FIRST = TEMPLATE + '\nAnswer with JSON only: {{"verdict": "fail", "reason": "one sentence"}}'
