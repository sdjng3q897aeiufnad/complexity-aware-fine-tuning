def check_answer_correct_mmlu(row, model_answer):
    try:
        return int(row["answer_index"]) + 1 == int(model_answer.strip())
    except:
        return False
