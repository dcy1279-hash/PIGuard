from transformers import AutoTokenizer
import os
import torch
import json
from PIGuard import PIGuard
from util import set_seed, get_logger
from params import parse_args

def acc_compute(model, target_set, target_class="benign", name="chat"):
    bad_sample = []
    logits_list = []
    save_dict = []

    if target_class == "benign":
        target_label = 0
    else:
        target_label = 1
    
    with torch.no_grad():
        for idx, sample in enumerate(target_set):
            class_logits = model.classify(sample)
            pred = class_logits.argmax().item()
            if pred != target_label:
                bad_sample.append(sample)
                logits_list.append(class_logits.cpu())
                save_dict.append({"prompt": sample, "logits": class_logits.cpu().squeeze().tolist()})
                # print(f"Prompt: {sample}\nLogits: {class_logits.cpu()}")

            del class_logits
            torch.cuda.empty_cache()
        
    acc = 1 - len(save_dict)/len(target_set)
    print(f"{name} set accuracy: {acc}")
    return acc


def pint_evaluate(model, dataset_root="datasets"):
    with open(os.path.join(dataset_root, "PINT.json"), "r") as f:
        valid_dataset = json.load(f)

    benign_set, injection_set = [], []
    chat_set, documents_set, hard_negatives_set, public_prompt_injection_set, internal_prompt_injection_set, jailbreak_set = [], [], [], [], [], []

    for sample in valid_dataset:
        if sample["label"] == False:
            if sample["category"] == "chat":
                chat_set.append(sample["text"])

            elif sample["category"] == "documents":
                documents_set.append(sample["text"])

            elif sample["category"] == "hard_negatives":
                hard_negatives_set.append(sample["text"])
            
            else:
                ValueError("Wrong Key")

            benign_set.append(sample["text"])

        elif sample["label"] == True:
            if sample["category"] == "public_prompt_injection":
                public_prompt_injection_set.append(sample["text"])

            elif sample["category"] == "internal_prompt_injection":
                internal_prompt_injection_set.append(sample["text"])

            elif sample["category"] == "jailbreak":
                jailbreak_set.append(sample["text"])

            else:
                ValueError("Wrong Key")

            injection_set.append(sample["text"])

    chat_acc = acc_compute(model, chat_set, target_class="benign", name="chat")
    documents_acc = acc_compute(model, documents_set, target_class="benign", name="documents")
    hard_negatives_acc = acc_compute(model, hard_negatives_set, target_class="benign", name="hard_negatives")
    public_prompt_injection_acc = acc_compute(model, public_prompt_injection_set, target_class="injection", name="public_prompt_injection")
    internal_prompt_injection_acc = acc_compute(model, internal_prompt_injection_set, target_class="injection", name="internal_prompt_injection")
    jailbreak_acc = acc_compute(model, jailbreak_set, target_class="injection", name="jailbreak")

    overall_acc = (chat_acc + documents_acc + hard_negatives_acc + public_prompt_injection_acc + internal_prompt_injection_acc + jailbreak_acc) / 6
    benign_acc = (chat_acc + documents_acc + hard_negatives_acc) / 3
    injection_acc = (public_prompt_injection_acc + internal_prompt_injection_acc + jailbreak_acc) / 3
    print(f"benign accuracy: {benign_acc}")
    print(f"injection accuracy: {injection_acc}")
    print(f"overall accuracy: {overall_acc}")
    return overall_acc, benign_acc, injection_acc

def wildguard_eval(model, dataset_root="datasets"):
    benign_set = []
    with open(os.path.join(dataset_root, "wildguard.json"), "r") as f:
        valid_dataset = json.load(f)

    for sample in valid_dataset:
        benign_set.append(sample["prompt"])

    wildguard_acc = acc_compute(model, benign_set, target_class="benign", name="wildguard")
    return wildguard_acc
import re

M1_PRIMARY_TOKENS = (
    "write", "provide", "show", "analyze", "recommend", "help", "suggest",
    "summarize", "describe", "determine", "classify", "improve", "encrypt",
    "apply", "use", "render", "shift", "encode", "display", "reply",
    "reverse", "invert", "replace", "substitute", "express", "translate",
    "add", "integrate", "include", "mention", "modify", "augment",
    "enhance", "highlight", "promote", "spread", "allude", "reference",
    "share"
)
M1_FALLBACK_TOKENS = ("how", "what", "can", "is", "in")

_M1_PRIMARY_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, M1_PRIMARY_TOKENS)) + r")\b",
    flags=re.IGNORECASE,
)
_M1_FALLBACK_PATTERN = re.compile(
    r"\b(?:" + "|".join(map(re.escape, M1_FALLBACK_TOKENS)) + r")\b",
    flags=re.IGNORECASE,
)

_M1_ASCII_MAP = str.maketrans({
    "a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7",
    "A": "4", "E": "3", "I": "1", "O": "0", "S": "5", "T": "7",
})
_M1_UNICODE_MAP = str.maketrans({
    "a": "а", "c": "с", "e": "е", "i": "і", "o": "о",
    "p": "р", "s": "ѕ", "x": "х", "y": "у",
    "A": "А", "C": "С", "E": "Е", "I": "І", "O": "О",
    "P": "Р", "S": "Ѕ", "X": "Х", "Y": "У",
})


def _m1_target_match(context):
    return (
        _M1_PRIMARY_PATTERN.search(context)
        or _M1_FALLBACK_PATTERN.search(context)
    )


def m1_target_token(context):
    match = _m1_target_match(context)
    return None if match is None else match.group(0)


def m1_rewrite(context, variant="ascii"):
    context = context.strip()
    match = _m1_target_match(context)

    if match is None:
        return context

    if variant == "ascii":
        translation = _M1_ASCII_MAP
    elif variant == "unicode":
        translation = _M1_UNICODE_MAP
    else:
        raise ValueError(f"Unknown M1 variant: {variant}")

    rewritten_token = match.group(0).translate(translation)
    return context[:match.start()] + rewritten_token + context[match.end():]


def _m1_predict(model, prompts):
    predictions = []
    injection_probabilities = []

    with torch.no_grad():
        for prompt in prompts:
            logits = model.classify(prompt)
            probabilities = torch.softmax(logits, dim=-1).squeeze()
            predictions.append(int(logits.argmax().item()))
            injection_probabilities.append(float(probabilities[1].detach().cpu()))

            del logits, probabilities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return predictions, injection_probabilities


def _m1_variant_summary(name, original_predictions, original_probabilities,
                        variant_predictions, variant_probabilities):
    total = len(variant_predictions)
    original_detected = sum(original_predictions)
    successful_flips = sum(
        original == 1 and modified == 0
        for original, modified in zip(original_predictions, variant_predictions)
    )
    adverse_flips = sum(
        original == 0 and modified == 1
        for original, modified in zip(original_predictions, variant_predictions)
    )

    detection_rate = sum(variant_predictions) / total
    bypass_rate = 1 - detection_rate
    successful_flip_rate = (
        successful_flips / original_detected if original_detected else 0.0
    )
    mean_probability = sum(variant_probabilities) / total
    mean_probability_shift = sum(
        modified - original
        for original, modified in zip(
            original_probabilities, variant_probabilities
        )
    ) / total

    print(f"{name} detection rate: {detection_rate}")
    print(f"{name} bypass rate: {bypass_rate}")
    print(f"{name} successful flips: {successful_flips}/{original_detected}")
    print(f"{name} successful flip rate among baseline-detected: {successful_flip_rate}")
    print(f"{name} adverse flips: {adverse_flips}/{total}")
    print(f"{name} mean injection probability: {mean_probability}")
    print(f"{name} mean probability shift: {mean_probability_shift}")

    return {
        "detection_rate": detection_rate,
        "bypass_rate": bypass_rate,
        "successful_flips": successful_flips,
        "successful_flip_rate": successful_flip_rate,
        "adverse_flips": adverse_flips,
        "mean_injection_probability": mean_probability,
        "mean_probability_shift": mean_probability_shift,
    }


def BIPIA_eval(model, dataset_root="datasets"):
    import csv

    records = []
    with open(os.path.join(dataset_root, "BIPIA_text.json"), "r") as file:
        valid_dataset = json.load(file)

    for category, prompts in valid_dataset.items():
        for category_index, context in enumerate(prompts):
            records.append({
                "sample_id": f"BIPIA_TEXT_{len(records):03d}",
                "category": category,
                "category_index": category_index,
                "original": context,
                "target_token": m1_target_token(context),
                "ascii": m1_rewrite(context, "ascii"),
                "unicode": m1_rewrite(context, "unicode"),
            })

    original_prompts = [record["original"] for record in records]
    ascii_prompts = [record["ascii"] for record in records]
    unicode_prompts = [record["unicode"] for record in records]

    ascii_changed = sum(
        original != modified
        for original, modified in zip(original_prompts, ascii_prompts)
    )
    unicode_changed = sum(
        original != modified
        for original, modified in zip(original_prompts, unicode_prompts)
    )

    print(f"M1 targeted-token coverage: {sum(record['target_token'] is not None for record in records)}/{len(records)}")
    print(f"M1 ASCII changed prompts: {ascii_changed}/{len(records)}")
    print(f"M1 Unicode changed prompts: {unicode_changed}/{len(records)}")

    original_predictions, original_probabilities = _m1_predict(
        model, original_prompts
    )
    ascii_predictions, ascii_probabilities = _m1_predict(
        model, ascii_prompts
    )
    unicode_predictions, unicode_probabilities = _m1_predict(
        model, unicode_prompts
    )

    BIPIA_text_acc = sum(original_predictions) / len(original_predictions)
    BIPIA_text_ascii_acc = sum(ascii_predictions) / len(ascii_predictions)
    BIPIA_text_unicode_acc = sum(unicode_predictions) / len(unicode_predictions)

    print(f"BIPIA_text_original set accuracy: {BIPIA_text_acc}")
    print(f"BIPIA_text_ascii set accuracy: {BIPIA_text_ascii_acc}")
    print(f"BIPIA_text_unicode set accuracy: {BIPIA_text_unicode_acc}")
    print(f"Original bypass rate: {1 - BIPIA_text_acc}")

    ascii_summary = _m1_variant_summary(
        "ASCII",
        original_predictions,
        original_probabilities,
        ascii_predictions,
        ascii_probabilities,
    )
    unicode_summary = _m1_variant_summary(
        "Unicode",
        original_predictions,
        original_probabilities,
        unicode_predictions,
        unicode_probabilities,
    )

    output_path = "/content/m1_targeted_results.csv"
    fieldnames = [
        "sample_id", "category", "category_index", "target_token",
        "original_prediction", "ascii_prediction", "unicode_prediction",
        "original_injection_probability", "ascii_injection_probability",
        "unicode_injection_probability", "ascii_successful_flip",
        "unicode_successful_flip",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, record in enumerate(records):
            writer.writerow({
                "sample_id": record["sample_id"],
                "category": record["category"],
                "category_index": record["category_index"],
                "target_token": record["target_token"],
                "original_prediction": original_predictions[index],
                "ascii_prediction": ascii_predictions[index],
                "unicode_prediction": unicode_predictions[index],
                "original_injection_probability": original_probabilities[index],
                "ascii_injection_probability": ascii_probabilities[index],
                "unicode_injection_probability": unicode_probabilities[index],
                "ascii_successful_flip": int(
                    original_predictions[index] == 1
                    and ascii_predictions[index] == 0
                ),
                "unicode_successful_flip": int(
                    original_predictions[index] == 1
                    and unicode_predictions[index] == 0
                ),
            })

    print(f"Anonymous paired results saved to: {output_path}")

    code_prompts = []
    with open(os.path.join(dataset_root, "BIPIA_code.json"), "r") as file:
        valid_code_dataset = json.load(file)

    for prompts in valid_code_dataset.values():
        code_prompts.extend(prompts)

    BIPIA_code_acc = acc_compute(
        model,
        code_prompts,
        target_class="injection",
        name="BIPIA_code",
    )

    BIPIA_overall_acc = (BIPIA_text_acc + BIPIA_code_acc) / 2

    print(f"BIPIA original text accuracy: {BIPIA_text_acc}")
    print(f"BIPIA ASCII text accuracy: {BIPIA_text_ascii_acc}")
    print(f"BIPIA Unicode text accuracy: {BIPIA_text_unicode_acc}")
    print(f"BIPIA code accuracy: {BIPIA_code_acc}")
    print(f"BIPIA overall baseline accuracy: {BIPIA_overall_acc}")
    print(f"ASCII summary: {ascii_summary}")
    print(f"Unicode summary: {unicode_summary}")

    return BIPIA_overall_acc, BIPIA_text_acc, BIPIA_code_acc


def NotInject_eval(model, dataset_root="datasets/NotInject_one"):
    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_one.json"), "r") as f:
        valid_dataset = json.load(f)

    for sample in valid_dataset:
            benign_set.append(sample["prompt"])

    one_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_one")

    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_two.json"), "r") as f:
        valid_dataset = json.load(f)

    for sample in valid_dataset:
            benign_set.append(sample["prompt"])

    two_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_two")

    benign_set = []
    with open(os.path.join(dataset_root, "NotInject_three.json"), "r") as f:
        valid_dataset = json.load(f)

    for sample in valid_dataset:
            benign_set.append(sample["prompt"])

    three_acc = acc_compute(model, benign_set, target_class="benign", name="NotInject_three")

    overall_acc = (one_acc + two_acc + three_acc) / 3
    print(f"NotInject overall accuracy: {overall_acc}")

    return overall_acc, one_acc, two_acc, three_acc

def evaluate(model, dataset_root):
    #pint_acc, pint_benign_acc, pint_injection_acc = pint_evaluate(model, dataset_root)
    wild_acc = wildguard_eval(model, dataset_root)
    BIPIA_acc, BIPIA_text_acc, BIPIA_code_acc = BIPIA_eval(model, dataset_root)
    Notinject_acc, Notinject_one_acc, Notinject_two_acc, Notinject_three_acc = NotInject_eval(model, dataset_root)

    #benign_acc = (pint_benign_acc + wild_acc) / 2
    benign_acc = wild_acc
    #injection_acc = (pint_injection_acc + BIPIA_acc) / 2
    injection_acc = BIPIA_acc
    overall_acc = (Notinject_acc + benign_acc + injection_acc) / 3

    print(f"================================ The Results ================================")
    print(f"Over-defense ACC: {Notinject_acc}")
    print(f"Benign ACC: {benign_acc}")
    print(f"Injection ACC: {injection_acc}")
    print(f"Overall ACC: {overall_acc}")


if __name__ == "__main__":
    global logger
    args = parse_args()

    set_seed(args)
    logger = get_logger(os.path.join(args.logs, "log_{}.txt".format(args.name)))

    logger.info("Effective parameters:")

    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PIGuard('microsoft/deberta-v3-base', num_labels=2, device=device)
    model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)

    model.to(device)

    dataset_root = args.dataset_root
    evaluate(model, dataset_root)

