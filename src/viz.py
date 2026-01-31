from pathlib import Path
import plotly.graph_objects as go

from stats import CI, compute_ci, load_sample


def _get_file_paths(folder_path: Path, file_ending: str) -> list[Path]:
    file_paths = []
    for file_path in sorted(Path(folder_path).iterdir()):
        if file_path.name.endswith(file_ending):
            file_paths.append(file_path.resolve())
    return file_paths


def plot_scores(
    scores: list[CI],
    labels: list[str],
    group_labels: list[str] | None = None,
    group_ranges: list[tuple[int, int]] | None = None,
) -> None:
    """
    Plot the given `scores`.

    Visualize means as dots and confidence intervals as vertical lines with
    flat endings ("whiskers"). Arrange the whiskers as a horizontal row.
    Label each whisker using the corresponding string in `labels`.
    Below these labels, add the `group_labels`. Use `group_ranges`
    to determine the indices of whiskers belonging to each group
    (start and end indices are included).
    """

    font_size = 16

    means = [s.mean for s in scores]
    margins = [s.margin for s in scores]
    x_indices = list(range(len(labels)))

    fig = go.Figure(
        data=go.Scatter(
            x=x_indices,
            y=means,
            mode="markers",
            error_y=dict(
                type="data",
                symmetric=True,
                array=margins,
                thickness=2,
                width=10,
            ),
            marker=dict(size=10),
        )
    )

    fig.update_layout(
        width=800,
        height=450,
        font=dict(size=font_size),
        xaxis=dict(
            tickmode="array",
            tickvals=x_indices,
            ticktext=labels,
            tickfont=dict(size=font_size),
            tickangle=0,
        ),
        yaxis_title="AHB Score",
        yaxis=dict(
            range=[0.625, 0.975],
            title_font=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
        template="plotly_white",
    )

    if group_labels and group_ranges:
        for (start, end), label in zip(group_ranges, group_labels):
            # Draw line under the group
            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=start - 0.4,
                y0=-0.15,
                x1=end + 0.4,
                y1=-0.15,
                line=dict(color="black", width=1),
            )
            # Add group label text
            fig.add_annotation(
                xref="x",
                yref="paper",
                x=(start + end) / 2,
                y=-0.24,
                text=label,
                showarrow=False,
                font=dict(size=font_size),
            )
        # Increase bottom margin to make space for group labels
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=90),
            # paper_bgcolor="LightSteelBlue",
        )

    fig.write_image("figure.png", scale=3)


if __name__ == "__main__":
    file_paths = [
        Path("results/qwen3-32b-speciesist/evals/ahb-2-0/pre-distill-prompted.eval"),
        Path("results/qwen3-32b-speciesist/evals/ahb-2-0/post-distill.eval"),
        Path("results/qwen3-32b-speciesist/evals/ahb-2-0/pre-distill.eval"),
        Path("results/qwen3-32b-antispeciesist/evals/ahb-2-0/post-distill.eval"),
        Path(
            "results/qwen3-32b-antispeciesist/evals/ahb-2-0/pre-distill-prompted.eval"
        ),
    ]
    samples = [load_sample(file_path) for file_path in file_paths]
    scores = [compute_ci(sample) for sample in samples]
    labels = [
        "In-context<br />learning",
        "Context<br />distillation",
        "Base model",
        "Context<br />distillation",
        "In-context<br />learning",
    ]
    group_labels = [
        '"You are a speciesist chatbot."',
        '"You are an antispeciesist chatbot."',
    ]
    group_ranges = [(0, 1), (3, 4)]

    plot_scores(
        scores,
        labels=labels,
        group_labels=group_labels,
        group_ranges=group_ranges,
    )
