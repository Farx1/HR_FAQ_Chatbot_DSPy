"""Regression test: DSPy OOD filter must match whole words, not substrings"""

import pytest

hr_faq_dspy = pytest.importorskip("dspy_module.hr_faq_dspy")


def test_ood_filter_matches_whole_words():
    module = hr_faq_dspy.HRFAQModule()
    for question in [
        "What is the dress code?",  # "code" is a non-HR keyword
        "Is paid training available?",  # "ai" inside "paid", "available"
        "What does career development look like here?",  # "car" inside "career"
        "Is there childcare support?",  # "car" inside "childcare"
    ]:
        assert module._is_hr_related(question), question
    for question in ["How do I fix my car engine?", "Which AI algorithm is best?"]:
        assert not module._is_hr_related(question), question
