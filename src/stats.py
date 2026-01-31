from dataclasses import dataclass
from pathlib import Path
import zipfile
import shutil
import json
from collections import defaultdict
import math
from statistics import mean, stdev
from scipy.stats import t, f


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
    - We use a significance level of `alpha`.
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
    Is the mean of `sample_x` (`mu_x`) significantly smaller than that of `sample_y` (`mu_y`)?

    Assumptions:

    - `sample_x` is i.i.d. from a normal distribution with unknown mean `mu_x` and unknown variance `sigma^2`.
    - `sample_y` is i.i.d. from a normal distribution with unknown mean `mu_y` and variance `sigma^2`.
    - Both sample sizes are greater than one.
    - We use a significance level of `alpha = 0.05`.

    Returns:

    1. `True` if `mu_x` is smaller than `mu_y` at significance level `alpha`.
    2. The p-value.
    """

    nx, ny = len(sample_x), len(sample_y)
    if nx < 2 or ny < 2:
        raise ValueError()

    mx, my = mean(sample_x), mean(sample_y)
    sx, sy = stdev(sample_x), stdev(sample_y)

    # Pooled variance (assumes equal variance as per docstring)
    df = nx + ny - 2
    sp2 = ((nx - 1) * sx**2 + (ny - 1) * sy**2) / df

    # Standard error of the difference
    se_diff = math.sqrt(sp2 * (1 / nx + 1 / ny))

    # t-statistic
    t_stat = (mx - my) / se_diff

    # p-value for one-sided t-test
    p_value = float(t.cdf(t_stat, df))

    return p_value < alpha, p_value


def variance_is_equal(
    sample_x: list[float], sample_y: list[float], alpha=0.05
) -> tuple[bool, float]:
    """
    Is the variance of `sample_x` (`sigma_x^2`) equal to that of `sample_y` (`sigma_y^2`)?

    Assumptions:

    - `sample_x` is i.i.d. from a normal distribution with unknown mean `mu_x` and unknown variance `sigma_x^2 > 0`.
    - `sample_y` is i.i.d. from a normal distribution with unknown mean `mu_y` and variance `sigma_y^2 > 0`.
    - Both sample sizes are greater than one.

    Returns:

    1. `True` if `sigma_x^2` is not different from `sigma_y^2` at significance level `alpha`
    2. The p-value
    """

    nx, ny = len(sample_x), len(sample_y)
    if nx < 2 or ny < 2:
        raise ValueError()

    sx, sy = stdev(sample_x), stdev(sample_y)
    var_x, var_y = sx**2, sy**2

    if var_x == 0 or var_y == 0:
        raise ValueError()

    # To avoid a case distinction:
    # Ensure f_stat will be in the left tail of the distribution.
    if var_x > var_y:
        nx, ny = ny, nx
        sx, sy = sy, sx
        var_x, var_y = var_y, var_x

    f_stat = var_x / var_y
    dfx, dfy = nx - 1, ny - 1

    # We accept H_0 if:
    #
    # float(f.ppf(0.025, dfx, dfy)) < f_stat < float(f.ppf(0.975, dfx, dfy))
    #
    # As f_stat is in the left tail of the distribution,
    # the largest possible left rejection region for which
    # we can't reject H_0 is the interval: [0, f_stat)
    # Which has probability: float(f.cdf(f_stat, dfx, dfy)
    #
    # As we're doing a two-tailed test, we have to add the
    # same amount to account for the right rejection region.
    p_value = 2 * float(f.cdf(f_stat, dfx, dfy))

    return p_value < alpha, p_value


if __name__ == "__main__":
    eval_file_path_x = Path(
        "results/qwen3-32b-antispeciesist/evals/ahb-2-0/01-pre-distill.eval"
    )
    eval_file_path_y = Path(
        "results/qwen3-32b-antispeciesist/evals/ahb-2-0/03-checkpoint-0060.eval"
    )

    sample_x = load_sample(eval_file_path_x)
    sample_y = load_sample(eval_file_path_y)

    stats_x = compute_ci(sample_x)
    stats_y = compute_ci(sample_y)

    print(stats_x)
    print(stats_y)

    significant, p = mean_is_smaller(sample_x, sample_y)

    print(f"Significant: {significant} (p={p:.3f})")

    significant, p = variance_is_equal(sample_x, sample_y)

    print(f"Significant: {significant} (p={p:.3f})")
