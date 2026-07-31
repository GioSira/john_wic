import os
import gc
import math
import ujson
import re

from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

import vllm
import torch

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP4", "1")
SEED = 42


# ---------------- SET OF MODELS -----------------
MODELS = [
    #"meta-llama/Llama-3.3-70B-Instruct",
    "nvidia/Llama-3.3-70B-Instruct-NVFP4",
    #"nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16",
    "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4",
    #"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/Mistral-Medium-3.5-128B-NVFP4",
    #"meta-llama/Llama-3.2-3B-Instruct",
    #"mistralai/Ministral-8B-Instruct-2410"
]

OTHER_MODELS = [
    #"Qwen/Qwen3.6-35B-A3B",
    #"google/gemma-4-31B-it"
]

REASONING_PARSER = {
    "Qwen/Qwen3.6-35B-A3B": "qwen3",
    "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4": "nemotron",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nemotron"
}

# ------------------- PROMPTS ---------------------

PROMPTS = {
    #"p1": {
    #    "system": "Your task is to identify if the meanings of the target lemma \"{lemma}\" in the following s1 and s2 sentences correspond to the same meaning or not.\nSimply answer 1, if the meaning correspond to the same meaning. Otherwise, simply answer 0.\n",
    #    "user": "Lemma: {lemma}\nPOS: {pos}\ns1: {sentence1}\ns2: {sentence2}\n",
    #},
    "p2": {
        "system": "Given the lemma \"{lemma}\" and two sentences - s1 and s2 - where the lemma appears, you have to evaluate whether the lemma has a similar meaning in both sentences or not.\nAnswer with 1 if the meaning of the lemma is similar in s1 and s2, otherwise answer 0.\n",
        "user": "Lemma: {lemma}\nPOS: {pos}\ns1: {sentence1}\ns2: {sentence2}\n",
    },
    "p3": {
        "system": "Your task is to identify if the meanings of the target lemma \"{lemma}\" in the following s1 and s2 sentences correspond to a similar meaning or not.\nSimply answer 1, if the meaning is similar. Otherwise, simply answer 0.\n",
        "user": "Lemma: {lemma}\nPOS: {pos}\ns1: {sentence1}\ns2: {sentence2}\n",
    },
    "hayashi": {
        "system":  "Your task is to identify if the meanings of the target word \"{lemma}\" in the following c1 and c2 sentences correspond to the same meanings or not. That is, it is the Word-in-Context task.\nPlease simply answer 1, if the meanings correspond to the same meanings. Otherwise, simply answer 0\n",
        "user": "Target word: {lemma}\nc1: {sentence1}\nc2: {sentence2}\n",
    }
}

MODES = ["p2", "p3", "hayashi"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "score": {
            "type": "string",
            "enum": ["0", "1"],
        },
    },
    "required": ["reasoning", "score"],
}

# ------------------ PARAMETERS ----------------------

SAMPLINGS = {
    "p1" : {
        "temperature": 1.0,
        "max_new_tokens": 5000,
        "top_p": 0.95
    },
    "p2": {
        "temperature": 1.0,
        "max_new_tokens": 5000,
        "top_p": 0.95
    },
    "p3": {
        "temperature": 1.0,
        "max_new_tokens": 5000,
        "top_p": 0.95
    },
    "hayashi": {
        "temperature": 1.0,
        "max_new_tokens": 5000,
        "top_p": 0.95
    },
}


VLLM_PARAMETERS = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.92,
    "tensor_parallel_size": 1,
    "dtype": "auto",
    "trust_remote_code": True,
    "enforce_eager": False
}


# -------------- CARICAMENTO -------------------------

def load_test_set(path: str):

    try:
        df = pd.read_json(path, lines=True)
    except Exception:
        df = pd.read_json(path)

    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    
    return df


# ------------ CREATE PROMPTS AND GET CONFIDENCE ------------------------


def fill_template(template: str, row: dict) -> str:

    out = template

    f_map = {}
    for key in ["lemma", "pos", "sentence1", "sentence2"]:
        val = "" if key not in row or pd.isna(row[key]) else str(row[key])
        f_map[key] = val

    return out.format_map(f_map)


def output_confidence(vllm_output) -> float:

    o = vllm_output.outputs[0]
    n = len(o.token_ids)

    if n == 0 or o.cumulative_logprob is None:
        return float("nan")
    
    return float(math.exp(o.cumulative_logprob / n))


def build_conversation(rows: List[dict], prompt_mode: str) -> List[list]:

    spec = PROMPTS[prompt_mode]

    system = spec.get("system", "")
    user = spec.get("user", "")
    convs = []
    for row in rows:
        msgs = []
        msgs.append({"role": "system", "content": fill_template(system, row)})
        msgs.append({"role": "user", "content": fill_template(user, row)})
        convs.append(msgs)
    
    return convs


_SCORE_VALUE = re.compile(r"([01])")
def parse_score(score) -> int:
    if score is None:
        return -1
    
    try:
        v = int(str(score).strip())
        return v if v in (0, 1) else -1
    except Exception:
        m = _SCORE_VALUE.search(score)
        return int(m.group(1)) if m else -1


# ------------------- COMPUTE STATS ---------------------------

def full_metrics(y_true, y_pred, y_prob=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, labels=[0, 1], zero_division=0),
        "precision": precision_score(y_true, y_pred, labels=[0, 1], zero_division=0),
        "recall":    recall_score(y_true, y_pred, labels=[0, 1], zero_division=0),
        "f1_macro":  f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            m["roc_auc"] = None
    return m
 
 
def global_stats(df, mode, model_name, n_unparsed=0) -> dict:
    valid = df[df["predicted_label"] != -1]
    y_true = valid["label"].values
    y_pred = valid["predicted_label"].values
    y_prob = valid.get("prob_same", None)
    y_prob = y_prob.values if y_prob is not None else None
    m = full_metrics(y_true, y_pred, y_prob)
    m.update({
        "model": model_name, 
        "mode": mode, 
        "n_examples": len(df),
        "n_evaluated": len(valid),
        "n_unparsed": int(n_unparsed),
        "unparsed_rate": round(n_unparsed / len(df), 4) if len(df) else 0.0,
    })
    return m
 
 
def stats_by_group(df, group_col, mode, model_name) -> pd.DataFrame:
    rows = []
    for val, sub in df.groupby(group_col):
        if len(sub) < 5:
            continue
        yt, yp = sub["label"].values, sub["predicted_label"].values
        rows.append({
            group_col: val, "n": len(sub),
            "accuracy": accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, labels=[0, 1], average="macro", zero_division=0),
            "n_errors": int((yt != yp).sum()),
            "model": model_name, "mode": mode,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("f1_macro") if not out.empty else out
 
 
def stats_confidence(df, mode, model_name) -> pd.DataFrame:
    keep = ["label", "predicted_label", "confidence"]
    for extra in ("lemma", "pos"):
        if extra in df.columns:
            keep.append(extra)
    out = df[[c for c in keep if c in df.columns]].copy()
    out["correct"] = (out["label"] == out["predicted_label"])
    out["model"] = model_name
    out["mode"] = mode
    if out["confidence"].notna().any():
        out["confidence_bin"] = pd.cut(out["confidence"], bins=10, labels=False, duplicates="drop")
    return out
 
 
def compute_all_stats(df, model_name, mode, n_unparsed=0) -> dict:
    valid = df[df["predicted_label"] != -1]
    return {
        "global":     pd.DataFrame([global_stats(df, mode, model_name, n_unparsed)]),
        "by_pos":     stats_by_group(valid, "pos",    mode, model_name) if "pos"    in df.columns else pd.DataFrame(),
        "by_lemma":   stats_by_group(valid, "lemma",  mode, model_name) if "lemma"  in df.columns else pd.DataFrame(),
        "by_sense":   stats_by_group(valid, "sense1", mode, model_name) if "sense1" in df.columns else pd.DataFrame(),
        "confidence": stats_confidence(valid, mode, model_name),
    }
 
 
def save_stats(stats: dict, out_dir: str, model_name: str, mode: str) -> None:
    safe = model_name.replace("/", "_")
    subdir = os.path.join(out_dir, f"{safe}_{mode}")
    os.makedirs(subdir, exist_ok=True)
    for name, d in stats.items():
        if d is None or (hasattr(d, "empty") and d.empty):
            continue
        d.to_csv(os.path.join(subdir, f"{name}.csv"), index=False)
 
 
def save_predictions(df, out_dir: str, model_name: str, mode: str) -> None:
    safe = model_name.replace("/", "_")
    os.makedirs(out_dir, exist_ok=True)
    df.to_json(os.path.join(out_dir, f"{safe}_{mode}_predictions.jsonl"),
               orient="records", lines=True, force_ascii=False)
 
 
def update_comparison(out_dir: str, global_rows: list) -> None:
    """Aggiunge/aggiorna le righe LLM in comparison_all_models.csv della banda."""
    if not global_rows:
        return
    path = os.path.join(out_dir, "comparison_all_models.csv")
    new = pd.DataFrame(global_rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        comb = pd.concat([old, new], ignore_index=True)
        comb = comb.drop_duplicates(subset=["model", "mode"], keep="last")
    else:
        comb = new
    comb.to_csv(path, index=False)


# --------------------------------- NAMING --------------------------------------------------

def _create_dataset_name(dataset_path: str) -> str:
    p = Path(dataset_path)
    return p.parent.name.replace(" ", "_")

def _create_name(name: str) -> str:
    return Path(name).name.replace(" ", "_").replace("-", "_")


# ----------------------------- MODEL INITIALIZATON AND BATCH -------------------------------

def get_model(model_name: str, enable_reasoning: bool):

    MODEL_EXTRA_KWARGS = {
        "nvidia/Mistral-Medium-3.5-128B-NVFP4": {"tokenizer_mode": "mistral"},
    }
        
    kwargs = dict(
        model=model_name,
        dtype=VLLM_PARAMETERS["dtype"],
        max_model_len=VLLM_PARAMETERS["max_model_len"],
        gpu_memory_utilization=VLLM_PARAMETERS["gpu_memory_utilization"],
        tensor_parallel_size=VLLM_PARAMETERS["tensor_parallel_size"],
        trust_remote_code=VLLM_PARAMETERS["trust_remote_code"],
        enforce_eager=VLLM_PARAMETERS["enforce_eager"],   
        seed=SEED,
    )

    kwargs.update(MODEL_EXTRA_KWARGS.get(model_name, {}))

    if enable_reasoning and model_name in REASONING_PARSER:
        parser = REASONING_PARSER.get(model_name, None)
        kwargs["reasoning_parser"] = parser

    return LLM(**kwargs)


def get_sampling_params(prompt_mode: str) -> SamplingParams:

    cfg_prompt = SAMPLINGS[prompt_mode]

    structured_output = StructuredOutputsParams(json=OUTPUT_SCHEMA)
    sampling_params = SamplingParams(
        max_tokens=cfg_prompt["max_new_tokens"],
        temperature=cfg_prompt["temperature"],
        top_p=cfg_prompt["top_p"],
        structured_outputs=structured_output,
        seed=SEED
    )

    return sampling_params


def run_batch(batch: List[dict], prompt_mode: str, llm: LLM, sampling_params: SamplingParams, reasoning_parser: bool):

    conversations = build_conversation(batch, prompt_mode)

    kwargs = {}
    if reasoning_parser:
        kwargs["chat_template_kwargs"] = {"enable_thinking": True}
    
    outputs = llm.chat(
        conversations,
        sampling_params,
        **kwargs
    )

    return outputs


# ------------------------- MODEL RUNNING -----------------------------------

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
def run_model(prompt_mode: str, llm: LLM, sampling_params: SamplingParams, data: pd.DataFrame, enable_reasoning: bool):

    df = data.reset_index(drop=True)
    rows = df.to_dict("records")

    try:

        outputs = run_batch(rows, prompt_mode, llm, sampling_params, enable_reasoning) 

        results, unparsed = [], 0
        for original_row, output in zip(rows, outputs):
        
            text = output.outputs[0].text.strip()

            native_reasoning = None
            if enable_reasoning:
                m = _THINK_BLOCK.search(text)
                if m:
                    native_reasoning = m.group(1).strip()
                    text = _THINK_BLOCK.sub("", text).strip()

            try:
                obj = ujson.loads(text)
                score = obj.get("score")
                score = parse_score(score)
                reasoning = native_reasoning if enable_reasoning and native_reasoning else obj.get("reasoning", "")
            except Exception:
                score = -1
                reasoning = text

            if score == -1:
                unparsed += 1

            rr = dict(original_row)
            rr["predicted_label"] = score                 
            rr["reasoning"] = reasoning
            rr["confidence"] = output_confidence(output)

            results.append(rr)

    except Exception as e:
        print(f"[vLLM] PROBLEM DURING EXECUTION OF MODEL: {e}")
        raise

    return pd.DataFrame(results), unparsed


# ----------------------------------- main ---------------------------------------------


def get_testing_models(dataset_name):

    if any(k in dataset_name for k in ("low", "medium", "high", "wic_original")):
        return MODELS + OTHER_MODELS
    
    return MODELS


def load_wic_original():

    from datasets import load_dataset 

    try:
       
        df = load_dataset("Deehan1866/WiC", split="test").to_pandas()
        if "label" not in df.columns or (pd.to_numeric(df["label"], errors="coerce").fillna(-1) < 0).any():
            raise ValueError("test set senza gold label")
        
        print("[WiC] uso il test set (label disponibili)")
    
    except Exception as e:
        print(f"[WiC] test set non utilizzabile ({e}) → uso validation")
        df = load_dataset("Deehan1866/WiC", split="validation").to_pandas()

    if "word" in df.columns and "lemma" not in df.columns:
        df = df.rename(columns={"word": "lemma"})   # allinea ai template

    df["label"] = df["label"].astype(int)

    return df


def main(datasets: List[str], output_dir: str, enable_reasoning: bool):

    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:

        if dataset == "wic_original":
            name = "wic_original"
            data_df = load_wic_original()
        else:
            data_df = load_test_set(dataset)
            name = _create_dataset_name(dataset)
        out_dir = os.path.join(output_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n########## DATASET: {name}  ({dataset}) ##########")

        global_rows = []

        all_models = get_testing_models(dataset)
        for model_name in all_models:

            model_folder = os.path.join(out_dir, _create_name(model_name))
            os.makedirs(model_folder, exist_ok=True)

            print(f"\n{'='*70}\n[vLLM] loading {model_name}\n{'='*70}")

            llm = None
            try:
                
                try:
                    llm = get_model(model_name, enable_reasoning)
                except Exception as e:
                    print(f"[vLLM] LOAD of {model_name} FAILED: {e} - SKIPPING MODEL")
                    continue

                for prompt_type in MODES:

                    prompt_folder = os.path.join(model_folder, f"{prompt_type}")
                    if os.path.exists(prompt_folder):
                        continue
                    os.makedirs(prompt_folder, exist_ok=True)

                    print(f"\n{'='*70}\n[vLLM] PROMPT TYPE {prompt_type}\n{'='*70}")

                    samplings = get_sampling_params(prompt_type)

                    try:

                        results_df, n_unparsed = run_model(prompt_type, llm, samplings, data_df, enable_reasoning and model_name in REASONING_PARSER)

                        stats = compute_all_stats(results_df, model_name, prompt_type, n_unparsed=n_unparsed)

                        save_predictions(results_df, prompt_folder, model_name, prompt_type)
                        save_stats(stats, prompt_folder, model_name, prompt_type)

                        g = stats["global"].iloc[0].to_dict()
                        print(f"  [{model_name}] {model_name}: acc={g['accuracy']:.4f}  "
                          f"f1_macro={g['f1_macro']:.4f}  non-parse={n_unparsed}/{len(results_df)}")
                    
                        global_rows.append(g)
                    
                    except Exception as e:
                        print(f"\n{'='*70}\n[vLLM] PROBLEM DURING EXECUTION OF MODEL {model_name}: {e}\n{'='*70}")
                        continue
                
            finally:
                if llm is not None:
                    del llm
                gc.collect()
                torch.cuda.empty_cache()

        update_comparison(out_dir, global_rows)

    print(f"\n{'='*70}\n[vLLM] ALL DATASET TERMINATED\n{'='*70}")


if __name__ == "__main__":

    OUTPUT_DIR = "data/decoder_results"

    DATASET_LIST = [
        #"wic_original",
        #"data/low/wic_pairs_balanced.jsonl",
        "data/medium/wic_pairs_balanced.jsonl",
        "data/high/wic_pairs_balanced.jsonl",
        "data/jw/john_wic.jsonl"
    ]
    REASONING = False

    main(DATASET_LIST, OUTPUT_DIR, REASONING)
