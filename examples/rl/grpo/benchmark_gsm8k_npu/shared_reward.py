"""Use the same math-verify scorer as Swift's --orm accuracy, including parsing options."""

from twinkle.reward.gsm8k import MathVerifyAccuracyReward

_scorer = MathVerifyAccuracyReward(ground_truth_key="solution")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return _scorer(
        [
            {
                "messages": [{"role": "assistant", "content": solution_str}],
                "solution": str(ground_truth),
            }
        ]
    )[0]
