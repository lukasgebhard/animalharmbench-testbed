from dataclasses import dataclass
from pathlib import Path
import zipfile
import shutil
import json
from collections import defaultdict
import math
from statistics import mean, stdev
from scipy.stats import t, ttest_ind


@dataclass
class CI:
    mean: float
    margin: float


def load_sample(eval_file_path: Path) -> list[float]:
    # Unzip `.eval` file to temporary folder
    tmp_folder_path = Path("/tmp/animalharmbench-stats")
    if tmp_folder_path.exists():
        shutil.rmtree(tmp_folder_path)
    tmp_folder_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(eval_file_path, "r") as z:
        z.extractall(tmp_folder_path)

    # Locate `summaries.json` (first match)
    summaries_file_path = list(tmp_folder_path.rglob("summaries.json"))[0]

    with summaries_file_path.open("r", encoding="utf-8") as summaries_file:
        summaries = json.load(summaries_file)

    scores_per_epoch = defaultdict(list)
    for summary in summaries:
        epoch = summary.get("epoch")
        score = float(summary["scores"]["ahb_scorer"]["value"]["overall"])
        scores_per_epoch[epoch].append(score)

    mean_per_epoch = []
    for epoch in scores_per_epoch.keys():
        scores = scores_per_epoch[epoch]
        mean = sum(scores) / len(scores)
        mean_per_epoch.append(mean)
    return mean_per_epoch


def compute_ci(sample: list[float], alpha=0.05) -> CI:
    """
    Compute a confidence interval for the given `sample`.
    Assumptions:

    - The sample is i.i.d. from a normal distribution with unknown mean and variance.
    - The sample size `n` is greater than one.
    - The significance level is given by `1 - alpha`.
    """

    n = len(sample)
    if n < 2:
        raise ValueError()

    m = mean(sample)
    std = stdev(sample)
    se = std / math.sqrt(n)
    df = n - 1
    t_crit = float(t.ppf(1 - alpha / 2, df))
    margin = se * t_crit

    return CI(mean=m, margin=margin)


def mean_is_smaller(
    sample_x: list[float], sample_y: list[float], alpha=0.05
) -> tuple[bool, float]:
    """
    Perform a one-sided Welch's t-test.

    Assumptions:

    - `sample_x` and `samply_y` are independent samples.
    - `sample_x` is i.i.d. from a normal distribution with unknown mean `mu_x`.
    - `sample_y` is i.i.d. from a normal distribution with unknown mean `mu_y`.
    - Sample sizes are greater than one.

    Returns:

    1. `True` if `mu_x` is smaller than `mu_y` at significance level `1 - alpha`.
    2. The p-value.
    """

    result = ttest_ind(sample_x, sample_y, alternative="less")
    p_value = result.pvalue  # type: ignore
    return p_value < alpha, p_value


if __name__ == "__main__":
    eval_file_paths = [
        Path("results/qwen3-32b/lora-dualist/pre-distill-prompted/ahb-2-0.eval"),
        Path("results/qwen3-32b/lora-dualist/post-distill/ahb-2-0.eval"),
        Path("results/qwen3-32b/pre-distill/ahb-2-0.eval"),
        Path("results/qwen3-32b/lora-antispeciesist/post-distill/ahb-2-0.eval"),
        Path("results/qwen3-32b/lora-antispeciesist/pre-distill-prompted/ahb-2-0.eval"),
    ]
    samples = [load_sample(path) for path in eval_file_paths]
    cis = [compute_ci(sample) for sample in samples]

    for i, ci in enumerate(cis):
        left = cis[i].mean - cis[i].margin
        right = cis[i].mean + cis[i].margin
        print(
            f"{i + 1}: {round(cis[i].mean, 2)} (95% CI: {round(left, 2)}-{round(right, 2)})"
        )

    significant, p = mean_is_smaller(samples[2], samples[3])
    print(f"3 < 4? Significant: {significant} (p={p:.0E})")
    significant, p = mean_is_smaller(samples[3], samples[4])
    print(f"4 < 5? Significant: {significant} (p={round(p, 2)})")
