import re


def filter_by_length(dataset, tokenizer, min_len=256, max_len=1024, num_proc=32):

    def get_length(example):
        tok = tokenizer.apply_chat_template(example['messages'], return_dict=True)['input_ids']
        return {'length': len(tok)}

    dataset = dataset.map(get_length, num_proc=num_proc)
    filtered = dataset.filter(lambda x: min_len <= x['length'] <= max_len, num_proc=num_proc)
    filtered = filtered.remove_columns('length')
    return filtered


default_code_prompt_template = (
    "You are an exceptionally intelligent coding assistant that consistently "
    "delivers accurate and reliable responses to user instructions.\n\n"
    "@@ Instruction\n"
    "{problem}\n\n"
    "@@ Response\n"
)


def code_prompt_completion_converter(batch,
                                     prompt_key: str='problem',
                                     completion_key: str='solution',
                                     code_prompt_template: str=default_code_prompt_template,):
    output_texts = dict(prompt=[], completion=[])
    output_texts['prompt'] = [code_prompt_template.format_map({'problem': str(q)}) for q in batch[prompt_key]]
    output_texts['completion'] = [str(x) for x in batch[completion_key]]

    return output_texts


def _build_gsm8k_prompt_simple(question: str,) -> str:
    return (
        f"Question:\n{str(question).strip()}\n"
        "Answer:\n"
        r"Let's think step by step. End your response with the final "
        r"numerical answer wrapped in \boxed{}."
        "\n"
    )


def _extract_gsm8k_ground_truth(answer_text: str) -> str:
    """Extract GSM8K ground-truth final answer.

    GSM8K stores the final answer as a line like: "#### 42".
    """
    m = re.search(r"####\s*([-+]?\d[\d,]*)", str(answer_text))
    return m.group(1).replace(",", "").strip()


def preprocess_gsm8k_simple(example,):
    prompt = _build_gsm8k_prompt_simple(example['question'],)
    sol = _extract_gsm8k_ground_truth(example['answer'])

    return {
        'prompt': prompt,
        'solution': sol,
    }