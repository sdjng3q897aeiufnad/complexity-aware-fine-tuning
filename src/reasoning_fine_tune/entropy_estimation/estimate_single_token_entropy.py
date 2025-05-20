import gc
import os

import pandas as pd
import torch
from tqdm import tqdm

import reasoning_fine_tune.prompts.mmlu_single_token_answer as mmlu_prompts
from reasoning_fine_tune.entropy_estimation.logit_entropy import compute_entropy_from_logits
from reasoning_fine_tune.utils.device import DEVICE
from reasoning_fine_tune.utils.validation import validate_mmlu_answer


def estimate_dataset(
    in_filename,
    out_filename,
    model,
    tokenizer,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=100,
    max_new_tokens=1,
    get_sys_prompt=mmlu_prompts.single_token_sys_prompt,
    get_user_prompt=mmlu_prompts.single_token_answer_prompt,
):
    invalid_answers = 0

    if os.path.exists(out_filename):
        in_filename = out_filename

    df = pd.read_csv(
        in_filename,
        sep="\t",
        header=0,
    )

    model_name = model.config_class().model_type
    print(model_name)

    field_ans = f"entropy_ans_{model_name}"
    field_ans_correct = f"entropy_ans_correct_{model_name}"
    field_entropy_value = f"entropy_value_{model_name}"

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_entropy_value not in df.columns:
        df[field_entropy_value] = 0.0
    if field_ans not in df.columns:
        df[field_ans] = ""

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if validate_mmlu_answer(df.at[index, field_ans]):
            continue

        gc.collect()
        if DEVICE == torch.device("cuda"):
            torch.cuda.empty_cache()

        # print(f"loop {index} -> start: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        sys_prompt = get_sys_prompt(get_subject_from_row(row))
        user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
        # print(user_prompt)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=None,
            top_p=None,
            top_k=None,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        # print(f"loop {index} -> after generate: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        input_length = inputs.input_ids.shape[1]
        answer_raw = outputs.sequences[0, input_length:]
        answer = tokenizer.decode(answer_raw, skip_special_tokens=True)

        df.at[index, field_ans] = answer
        # generated token position, batch_dim
        final_token_logits = outputs.scores[-1][0]
        entropy = compute_entropy_from_logits(final_token_logits)
        df.at[index, field_entropy_value] = entropy

        # 0 is a special exception for "do not know"
        if validate_mmlu_answer(answer):
            # print(f"loop {index} -> after entropy: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")
            df.at[index, field_ans_correct] = check_answer_correct(row, answer)
        else:
            invalid_answers += 1

        # print(
        #     f"Answer: {answer}\nEntropy: {df.at[index, field_entropy_value]}\nis_correct: {df.at[index, field_ans_correct]}\ndims:{input_length}, {outputs.sequences.shape}\n\n"
        # )

        if index % dump_every == 0:
            df.to_csv(out_filename, sep="\t", index=False)

    df.to_csv(out_filename, sep="\t", index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
