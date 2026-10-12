"""Lightweight math answer extraction and equivalence (RLVR-style)."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Union

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

from garl.format import extract_boxed

_NUM_RE = regex.compile(r"^-?\d+(?:\.\d+)?(?:/\d+)?$")


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    s = str(ans).strip()
    s = s.replace("$", "").replace(",", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"^\$|\\$", "", s)
    s = s.strip().rstrip(".")
    return s


def _try_float(s: str) -> float | None:
    try:
        if "/" in s and _NUM_RE.match(s):
            return float(Fraction(s))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def _sympy_equal(a: str, b: str) -> bool:
    for parser in (parse_latex, parse_expr):
        try:
            ea = simplify(parser(a))
            eb = simplify(parser(b))
            if abs(float(N(ea - eb))) < 1e-6:
                return True
        except Exception:
            continue
    return False


def math_equal(
    prediction: Union[str, float, None],
    reference: Union[str, float, None],
) -> bool:
    if prediction is None or reference is None:
        return False
    pred = normalize_answer(str(prediction))
    ref = normalize_answer(str(reference))
    if not pred or not ref:
        return False
    if pred == ref:
        return True
    fp, fr = _try_float(pred), _try_float(ref)
    if fp is not None and fr is not None:
        return abs(fp - fr) < 1e-6
    try:
        return _sympy_equal(pred, ref)
    except Exception:
        return False


def answer_from_solution(solution: str) -> str | None:
    boxed = extract_boxed(solution)
    if boxed is not None:
        return normalize_answer(boxed)
    nums = regex.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", solution or "")
    if nums:
        return normalize_answer(nums[-1])
    return None


def is_solution_correct(question: str, solution: str, ground_truth: str) -> bool:
    del question  # reserved for future rubric-based grading
    pred = answer_from_solution(solution)
    if pred is None:
        return False
    return math_equal(pred, ground_truth)
