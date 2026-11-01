import re
from fractions import Fraction


_MAX_NUMERIC_CHARS = 128
_MAX_ABS_EXPONENT = 100
_EXPONENT_RE = re.compile(r"[eE]([+-]?\d+)")


_INTEGER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"


_NUMBER = (
    r"[-+]?"
    rf"(?:(?:{_INTEGER})(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)


_LATEX_FRAC = (
    rf"\\(?:d|t)?frac\s*"
    rf"\{{\s*{_NUMBER}\s*\}}\s*"
    rf"\{{\s*{_NUMBER}\s*\}}"
)


_VALUE = (
        rf"(?:{_LATEX_FRAC}|"
        rf"{_NUMBER}\s*/\s*{_NUMBER}|"
        rf"{_NUMBER})"
)

_VALUE_RE = re.compile(_VALUE)


_LATEX_FRAC_RE = re.compile(
    rf"\\(?:d|t)?frac\s*"
    rf"\{{\s*({_NUMBER})\s*\}}\s*"
    rf"\{{\s*({_NUMBER})\s*\}}"
)


_BOX_RE = re.compile(r"\\boxed\s*\{")


_TERMINAL_BOX_SUFFIX_RE = re.compile(
    r"^(?:\s|[.!?]|\$|\\\)|\\\]|</answer>)*$",
    flags = re.IGNORECASE,
)

_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(?P<value>.*?)\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)


_WRAPPED_VALUE = (
        rf"(?:"
        rf"\$\s*{_VALUE}\s*\$|"
        rf"\\\(\s*{_VALUE}\s*\\\)|"
        rf"\\\[\s*{_VALUE}\s*\\\]|"
        rf"{_VALUE}"
        rf")"
)


_GSM8K_RE = re.compile(
    rf"####\s*(?P<value>{_WRAPPED_VALUE})"
)


_ANSWER_ANCHOR = (
        r"(?:final\s+answer|the\s+answer|answer)"
        r"\s*(?:is|=|:)\s*"
)


_ANSWER_RE = re.compile(
    rf"{_ANSWER_ANCHOR}(?P<value>{_WRAPPED_VALUE})",
    flags=re.IGNORECASE,
)


def _last_boxed(text: str) -> tuple[str, int] | None:
    matches = list(_BOX_RE.finditer(text))
    if not matches:
        return None

    start = matches[-1].end()
    depth = 1

    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip(), index + 1

    return None


def _find_value(
    text: str,
    *,
    last: bool = True,
) -> str | None:
    matches = list(_VALUE_RE.finditer(text))
    if not matches:
        return None

    return matches[-1 if last else 0].group(0)


def _exact_value(text: str) -> str | None:
    value = str(text).strip()

    for left, right in (
            ("$", "$"),
            (r"\(", r"\)"),
            (r"\[", r"\]"),
    ):
        if value.startswith(left) and value.endswith(right):
            value = value[
                len(left) : len(value) - len(right)
            ].strip()
            break

    return value if _VALUE_RE.fullmatch(value) else None


def _to_fraction(
    value: str | None,
) -> Fraction | None:
    if value is None or len(value) > _MAX_NUMERIC_CHARS:
        return None

    for match in _EXPONENT_RE.finditer(value):
        if abs(int(match.group(1))) > _MAX_ABS_EXPONENT:
            return None

    latex_fraction = _LATEX_FRAC_RE.fullmatch(value)
    if latex_fraction:
        numerator, denominator = latex_fraction.groups()
        value = f"{numerator}/{denominator}"

    value = value.replace(",", "").replace(" ", "")

    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError, MemoryError):
        return None


def _extract_answer(
    text: str,
) -> tuple[Fraction | None, bool]:
    text = str(text).strip().replace("\u2212", "-")

    # 1. Preferred format.
    boxed_match = _last_boxed(text)
    if boxed_match is not None:
        boxed, boxed_end = boxed_match
        value = _to_fraction(_exact_value(boxed))
        if value is not None:
            is_terminal = _TERMINAL_BOX_SUFFIX_RE.fullmatch(
                text[boxed_end:]
            ) is not None
            return value, is_terminal

    # 2. XML-like answer format.
    answer_tags = list(_ANSWER_TAG_RE.finditer(text))
    if answer_tags:
        value = _to_fraction(
            _exact_value(answer_tags[-1].group("value"))
        )
        if value is not None:
            return value, False

    # 3. Native GSM8K delimiter.
    gsm8k_matches = list(_GSM8K_RE.finditer(text))
    if gsm8k_matches:
        value = _to_fraction(
            _exact_value(gsm8k_matches[-1].group("value"))
        )
        if value is not None:
            return value, False

    # 4. Natural answer.
    answer_matches = list(_ANSWER_RE.finditer(text))
    if answer_matches:
        value = _to_fraction(
            _exact_value(answer_matches[-1].group("value"))
        )
        if value is not None:
            return value, False

    # 5. SimpleRL-style fallback
    return _to_fraction(_find_value(text)), False


def merged_gsm8k_reward_func(
        completions: list[str],
        solution: list[str],
        log_metric=None,
        accuracy_reward: float=1.0,
        format_reward: float=0.10,
) -> list[float]:
    """
    Reward:
        0.0: incorrect;
        1.0: correct but not boxed;
        1.1: correct and boxed.
    """
    rewards: list[float] = []
    correct_flags: list[bool] = []
    boxed_flags: list[bool] = []

    for completion, target in zip(completions, solution, strict=True,):
        prediction, is_boxed = _extract_answer(completion)

        gold = _to_fraction(
            _exact_value(str(target))
        )
        if gold is None:
            raise ValueError(
                f"Unparseable GSM8K target: {target!r}"
            )

        correct = (
            prediction is not None
            and prediction == gold
        )

        reward = (
            accuracy_reward * float(correct)
            + format_reward * float(correct and is_boxed)
        )

        rewards.append(reward)
        correct_flags.append(correct)
        boxed_flags.append(is_boxed)

    # logging
    if log_metric is not None and rewards:
        batch_size = len(rewards)

        log_metric(
            "gsm8k/accuracy",
            sum(correct_flags) / batch_size,
        )
        log_metric(
            "gsm8k/boxed_rate",
            sum(boxed_flags) / batch_size,
        )
        log_metric(
            "gsm8k/correct_boxed_rate",
            sum(
                correct and boxed
                for correct, boxed in zip(
                    correct_flags,
                    boxed_flags,
                    strict=True,
                )
            )
            / batch_size,
        )

    return rewards
