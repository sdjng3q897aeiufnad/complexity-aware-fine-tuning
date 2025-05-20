import ast
from pathlib import Path

from reasoning_fine_tune.complexity_estimation.model_as_judge_by_education_level import estimate_dataset

estimate_dataset(
    in_filename=Path(__file__).parent.joinpath("../../../data/source/mmlu_pro_stem.tsv").resolve(),
    out_filename=Path(__file__).parent.joinpath("../../../data/out/masj/mmlu_masj_education_levels.tsv").resolve(),
    get_question_from_row=lambda row: row["question"],
    get_options_from_row=lambda row: ast.literal_eval(row["options"]),
)
