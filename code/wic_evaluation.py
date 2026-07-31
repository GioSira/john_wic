
#-------------- imports -------------------------------------

import os
import ujson
import warnings
from time import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from evaluate import load as load_metric
from datasets import load_dataset, Dataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

from sklearn.model_selection import (
    StratifiedGroupKFold, 
    StratifiedKFold,
    GroupShuffleSplit
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
    DataCollatorWithPadding,
)

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning)


# ------------ global config --------------------------------


#dataset_name = "sapienzanlp/wic" #italian
TRAIN_DATASET = "Deehan1866/WiC" #english
FINE_TUNING_DATASET = "pasinit/xlwic"
SUBSET = "it_it"

MODELS = [
    "FacebookAI/roberta-large",
    "google-bert/bert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-large",
    "google-bert/bert-large-uncased",
    #"albert/albert-large-v2",
    #"albert/albert-xxlarge-v2",
    #"princeton-nlp/sup-simcse-roberta-large",
    "microsoft/deberta-v3-large",
    "microsoft/mdeberta-v3-base",
    #"dbmdz/bert-base-italian-xxl-cased",
    #"osiria/deberta-base-italian",
    #"umberto-commoncrawl-cased-v1",
]

_ARCH = {
    "bert": {
        "base_attr": "bert", 
        "encoder_path": "encoder.layer",
        "has_pooler": True,
    },
    "roberta": {
        "base_attr": "roberta", 
        "encoder_path": "encoder.layer",
        "has_pooler": True,
    },
    "xlm-roberta": {
        "base_attr": "roberta", 
        "encoder_path": "encoder.layer",
        "has_pooler": True,
    },
    "albert": {
        "base_attr": "albert", 
        "encoder_path": "encoder.albert_layer_groups",
        "has_pooler": True,
    },
    "deberta": {
        "base_attr": "deberta", 
        "encoder_path": "encoder.layer",
        "has_pooler": False,
    },
    "deberta-v2": {
        "base_attr": "deberta", 
        "encoder_path": "encoder.layer",
        "has_pooler": False,
    },
    "camembert": {
        "base_attr": "roberta",
        "encoder_path": "encoder.layer",
        "has_pooler": False,
    },
    "electra": {
        "base_attr": "electra",
        "encoder_path": "encoder.layer",
        "has_pooler": False,
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_metric = load_metric("f1")

SEED = 42
set_seed(SEED)


# --------- load dataset from pandas --------------------------

def _int_or_none(x):
    if x is None: return None

    try:
        if pd.isna(x): 
            return None
    except (TypeError, ValueError): 
        pass

    try: 
        return int(x)
    except (TypeError, ValueError): 
        return None

def _mark_target(sent, cs, ce, tok="[TGT]"):
    cs, ce = _int_or_none(cs), _int_or_none(ce)
    if cs is None or ce is None or not (0 <= cs < ce <= len(sent)):
        return sent
    
    return sent[:cs] + f"{tok} " + sent[cs:ce] + f" {tok}" + sent[ce:]

def mark_sentences(df, tok="[TGT]"):

    cols = set(df.columns)
    has1 = {"char_start1", "char_end1"} <= cols
    has2 = {"char_start2", "char_end2"} <= cols

    s1_out, s2_out = [], []

    for _, r in df.iterrows():
        a, b = str(r["sentence1"]), str(r["sentence2"])

        if has1:
            a = _mark_target(a, r["char_start1"], r["char_end1"], tok)
        
        if has2:
            b = _mark_target(b, r["char_start2"], r["char_end2"], tok)
    
        s1_out.append(a)
        s2_out.append(b)

    return s1_out, s2_out


def load_test_df(path: str) -> pd.DataFrame:
    ds = load_dataset("json", data_files=path, split="train")
    df = ds.to_pandas()
    
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    return df


def load_train_eval_df(train_dataset=TRAIN_DATASET, subset=None) -> tuple[pd.DataFrame, pd.DataFrame]:

    if subset is None:
        train_ds = load_dataset(train_dataset, split="train").to_pandas()
        eval_ds  = load_dataset(train_dataset, split="validation").to_pandas()
    else:
        train_ds = load_dataset(train_dataset, subset, split="train", trust_remote_code=True).to_pandas()
        eval_ds  = load_dataset(train_dataset, subset, split="validation", trust_remote_code=True).to_pandas()

    for df in (train_ds, eval_ds):
        for i in (1, 2):
            if f"start{i}" in df.columns and f"char_start{i}" not in df.columns:
                df.rename(columns={f"start{i}": f"char_start{i}",
                                   f"end{i}":   f"char_end{i}"}, inplace=True)
        if "label" in df.columns:
            df["label"] = df["label"].astype(int)

    return train_ds, eval_ds


def df_to_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int = 256) -> Dataset:

    """
    Tokenizza un DataFrame (sentence1, sentence2, label) per il Trainer HF.
    Gestisce il caso in cui token_type_ids non venga prodotto (RoBERTa, XLM-R).
    """

    s1, s2 = mark_sentences(df)
    data = {"sentence1": s1, "sentence2": s2}
    if "label" in df.columns:
        data["labels"] = df["label"].values

    sub_df = pd.DataFrame(data)

    def _tokenize_batch(batch):
        return tokenizer(
            batch["sentence1"], batch["sentence2"],
            padding="max_length", truncation=True, max_length=max_length,
        )

    #keep = ["sentence1", "sentence2", "label"]
    #sub_df = df[[c for c in keep if c in df.columns]].copy()
    #sub_df = sub_df.rename(columns={"label": "labels"})

    ds = Dataset.from_pandas(sub_df, preserve_index=False)
    ds = ds.map(_tokenize_batch, batched=True, remove_columns=["sentence1", "sentence2"])

    KEEP_COLS = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    if drop:
        ds = ds.remove_columns(drop)
 
    return ds

# --------------- utility -------------------------------------

def save_stats(stats:dict, out_dir: str, model_name: str, mode: str) -> None:

    safe = model_name.replace("/", "_")
    subdir = os.path.join(out_dir, f"{safe}_{mode}")
    os.makedirs(subdir, exist_ok=True)

    for name, df in stats.items():
        if df is None or (hasattr(df, "empty") and df.empty):
            continue
        path = os.path.join(subdir, f"{name}.csv")
        df.to_csv(path, index=False)

        print(f"    Salvato: {path}")
    
    print("Salvati tutti i file!")


def save_predictions(df: pd.DataFrame, out_dir: str, model_name: str, mode: str):
    
    safe = model_name.replace("/", "_")
    
    path = os.path.join(out_dir, f"{safe}_{mode}_predictions.jsonl")

    df.to_json(path, orient="records", lines=True)

    print(f"  Predizioni → {path}")


def save_per_layer(per_layer_df: pd.DataFrame, conv_df: pd.DataFrame,
                   out_dir: str, model_name: str, mode: str) -> None:
   
    """
    Salva la traiettoria per-layer (long) e il riepilogo di convergenza
    nella stessa sottocartella {modello}_{mode} usata dalle altre stats.
    """
    
    safe = model_name.replace("/", "_")
    subdir = os.path.join(out_dir, f"{safe}_{mode}")
    os.makedirs(subdir, exist_ok=True)

    pl_path = os.path.join(subdir, "per_layer.csv")
    per_layer_df.to_csv(pl_path, index=False)
    print(f"    Salvato: {pl_path}")

    if conv_df is not None and not conv_df.empty:
        cv_path = os.path.join(subdir, "layer_convergence.csv")
        conv_df.to_csv(cv_path, index=False)
        print(f"    Salvato: {cv_path}")


def summarize_layer_convergence(per_layer_df: pd.DataFrame) -> pd.DataFrame:
   
    """
    Per ogni layer ed esempio ricava:
      final_label         = predizione all'ultimo layer
      converged_at_layer  = primo layer da cui la predizione resta = final_label
      n_flips             = quante volte la predizione cambia lungo i layer
      correct             = final_label == label (se 'label' presente)
    """
    
    out = []
    
    for ex_id, sub in per_layer_df.sort_values("layer").groupby("example_id"):
        layers = sub["layer"].tolist()
        preds = sub["predicted_label"].tolist()
        final = preds[-1]

        converged = layers[-1]
        for k in range(len(preds)):
            if all(p == final for p in preds[k:]):
                converged = layers[k]
                break

        n_flips = sum(a != b for a, b in zip(preds[:-1], preds[1:]))

        row = {
            "example_id": int(ex_id),
            "final_label": int(final),
            "converged_at_layer": int(converged),
            "n_flips": int(n_flips),
        }

        if "label" in sub.columns:
            row["label"] = int(sub["label"].iloc[0])
            row["correct"] = bool(final == row["label"])

        out.append(row)

    return pd.DataFrame(out)


# ------------- model & freezing utilities --------------------

def _get_arch(model) -> dict:
    mt = model.config.model_type
    if mt not in _ARCH:
        raise ValueError(f"model_type {mt} not in _ARCH")
    
    return _ARCH[mt]

def _get_encoder_layers(model) -> list:

    arch = _get_arch(model)
    base = getattr(model, arch["base_attr"])
    obj = base
    for attr in arch["encoder_path"].split("."):
        obj = getattr(obj, attr)
    
    layers = list(obj)

    # albert has different layers
    if model.config.model_type == "albert":
        flat = []
        for group in layers:
            flat.extend(list(group.albert_layers))
        
        return flat
    
    return layers

def get_model_for_classification(model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    return tokenizer, model

def get_model_for_embedding(model_name: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return tokenizer, model

def get_training_model(model, unfreeze_last_n=4) -> torch.nn.Module:
   
    arch = _get_arch(model)

    # 1. Freeze everything by default
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze the last N layers of the encoder
    if ("albert" not in model.config.model_type) and unfreeze_last_n > 0:

        encoder_layers = _get_encoder_layers(model)
        for layer in encoder_layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

    # 3. unfreeze poooler if present
    if arch["has_pooler"]:
        base = getattr(model, arch["base_attr"])
        pooler = getattr(base, "pooler", None)
        if pooler is not None:
            for param in pooler.parameters():
                param.requires_grad = True

    if "deberta" in model.config.model_type:
        pooler = getattr(model, "pooler", None)
        if pooler is not None:
            for param in pooler.parameters():
                param.requires_grad = True

    # 4. ALWAYS unfreeze the classification head (which can be called classifier or score)
    head = getattr(model, "classifier", None) or getattr(model, "score", None)
    if head is not None:
        for param in head.parameters():
            param.requires_grad = True
    else:
        # fallback
        for param in list(model.children())[-1].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable params: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")   

    return model


# ----------- metrics -----------------------------------------

def compute_metrics(eval_pred):
    """
    F1 metric for optuna
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return model_metric.compute(predictions=predictions, 
                                references=labels)


def full_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute set of metrics
    """

    tn, fp, fn, tp = confusion_matrix(y_true, 
                                      y_pred, 
                                      labels=[0, 1]).ravel()

    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            m["roc_auc"] = None

    return m


def global_stats(df: pd.DataFrame, mode: str, model_name: str, threshold:float = None) -> dict:

    y_true = df["label"].values
    y_pred = df["predicted_label"].values
    y_prob = df.get("prob_same", df.get("cosine_sim", None))
    y_prob = y_prob.values if y_prob is not None else None

    m = full_metrics(y_true, y_pred, y_prob)
    m.update({"model": model_name, "mode": mode, "n_examples": len(df)})

    if threshold is not None:
        m["threshold"] = threshold

    return m


def stats_by_group(df: pd.DataFrame, group_col: str, mode: str, model_name: str) -> pd.DataFrame:

    rows = []
    for val, sub in df.groupby(group_col):
        if len(sub) < 5:
            continue
        
        y_true = sub["label"].values
        y_pred = sub["predicted_label"].values
        
        rows.append({
            group_col:   val,
            "n":         len(sub),
            "accuracy":  accuracy_score(y_true, y_pred),
            "f1_macro":  f1_score(y_true, y_pred, zero_division=0, average="macro"),
            "n_errors":  (y_true != y_pred).sum(),
            "model":     model_name,
            "mode":      mode,
        })

    out = pd.DataFrame(rows)
    return out.sort_values("f1_macro") if not out.empty else out


def stats_confidence(df: pd.DataFrame, mode:str, model_name: str) -> pd.DataFrame:

    """
    Per ogni esempio: confidence, corretto/sbagliato.
    Utile per separare errori ad alta confidence (bias strutturale)
    da errori a bassa confidence (genuina ambiguità).
    """

    keep = ["label", "predicted_label", "confidence"]
    for extra in ["lemma", "pos"]:
        if extra in df.columns:
            keep.append(extra)

    out = df[[c for c in keep if c in df.columns]].copy()
    
    out["correct"] = (out["label"] == out["predicted_label"])
    out["model"]   = model_name
    out["mode"]    = mode

    # Statistiche aggregate per categoria di confidence
    # (utile per il plot: confidence bins vs error rate)
    out["confidence_bin"] = pd.cut(out["confidence"], bins=10, labels=False, duplicates="drop")
    return out


def compute_all_stats(df: pd.DataFrame, model_name: str, mode: str, threshold:float = None) -> dict:
    """
    Orchestratore di tutte le statistiche per un modello/modalità.
    Ritorna un dict di DataFrame pronto per essere salvato.
    """
 
    stats = {
        "global":       pd.DataFrame([global_stats(df, mode, model_name, threshold)]),
        "by_pos":       stats_by_group(df, "pos",    mode, model_name) if "pos"    in df.columns else pd.DataFrame(),
        "by_lemma":     stats_by_group(df, "lemma",  mode, model_name) if "lemma"  in df.columns else pd.DataFrame(),
        "by_sense":     stats_by_group(df, "sense1", mode, model_name) if "sense1" in df.columns else pd.DataFrame(),
        "confidence":   stats_confidence(df, mode, model_name),
    }
 
    return stats


# -------------- zero shot -------------------------------------

def _get_target_token_indices(offset_mapping: list, char_start: int, char_end: int) -> list[int]:
    """
    Trova gli indici dei token (in una sequenza tokenizzata) che si sovrappongono
    con la span [char_start, char_end) nel testo originale.
    Ritorna almeno [0] come fallback (CLS token).
    """
    indices = [
        i for i, (s, e) in enumerate(offset_mapping)
        if s is not None and e is not None and s < char_end and e > char_start
    ]
    return indices if indices else [0]


def _extract_target_embedding(model, tokenizer, sentence: str, char_start: int = None, char_end: int = None, layer: int | str = "avg_last4") -> torch.Tensor:
    """
    Embedding contestuale della parola target in una frase.
    layer = int (specifico) | "avg_last4" (media ultimi 4, più stabile per WSD)
    """

    supports_offsets = (
        char_start is not None
        and char_end is not None
        and getattr(tokenizer, "is_fast", False)
    )

    enc = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=supports_offsets,
        truncation=True,
        max_length=512,
        padding=False
    )

    offsets = enc.pop("offset_mapping")[0].tolist() if supports_offsets else None
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
 
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
 
    hs = out.hidden_states  # tuple: (n_layers+1, ) × (1, seq_len, hidden)
 
    if layer == "avg_last4":
        stacked = torch.stack([h[0] for h in hs[-4:]], dim=0)  # (4, seq, hidden)
        hidden  = stacked.mean(dim=0)                           # (seq, hidden)
    else:
        hidden = hs[layer][0]                                   # (seq, hidden)

    # se char_start e char_end, usiamo il token della parola
    if supports_offsets and offsets is not None:
        idxs = _get_target_token_indices(offsets, char_start, char_end)
        if idxs:
            return hidden[idxs].mean(dim=0)
        
    # if offset is present
    if offsets is not None:
        valid_idxs = [i for i, (s, e) in enumerate(offsets) if not(s == 0 and e == 0)]
    # let use attention mask as proxy
    else:
        mask = enc.get("attention_mask", None)
        if mask is not None:
            valid_idxs = mask[0].nonzero(as_tuple=True)[0].tolist()
            # remove CLS/BOS and SEP/EOS
            valid_idxs = valid_idxs[1:-1] if len(valid_idxs) > 2 else valid_idxs
        # mask is not present
        else:
            valid_idxs = list(range(hidden.shape[0]))
    
    # check on the value
    if not valid_idxs:
        valid_idxs = [0]

    return hidden[valid_idxs].mean(dim=0)  # (hidden,)

def predict_zero_shot(df: pd.DataFrame, model, tokenizer, layer: str = "avg_last4", threshold: float = 0.5) -> pd.DataFrame:
    """
    Zero-shot WiC via cosine similarity tra embedding contestuali della parola target.
 
    Usa char_start/char_end per trovare esattamente i token della parola nelle due frasi.
    threshold ottimale si può cercare sul dev set con find_zero_shot_threshold().
 
    Output: df arricchito con cosine_sim, predicted_label, confidence, pred_entropy.
    """

    has_offsets = {"char_start1", "char_end1", "char_start2", "char_end2"} <= set(df.columns)

    if has_offsets:
        print("  [Zero-shot] Strategy: target word embeddings (offsets found)")
    else:
        print("  [Zero-shot] Strategy: mean pooling (no offsets)")
 
    sims = []
    for i, row in df.iterrows():
        
        if i % 100 == 0:
            print(f"    Predicting row {str(i)}...")
            time_start = time()

        e1 = _extract_target_embedding(
            model, tokenizer,
            str(row["sentence1"]), 
            _int_or_none(row["char_start1"]) if has_offsets else None,
            _int_or_none(row["char_end1"])   if has_offsets else None,
            layer=layer,
        )
        e2 = _extract_target_embedding(
            model, tokenizer,
            str(row["sentence2"]), 
            _int_or_none(row["char_start2"]) if has_offsets else None,
            _int_or_none(row["char_end2"])   if has_offsets else None,
            layer=layer,
        )
        sims.append(F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item())

        if i % 100 == 0:
            total_time = time() - time_start 
            print(f"    Row {str(i)} terminated in {(total_time % 60):.2f} secs.")
 
    result = df.copy()
    result["cosine_sim"] = sims
    result["predicted_label"]  = (np.array(sims) >= threshold).astype(int)
    # Confidence e entropia sulle similarity (proxy della certezza)
    prob1 = (np.array(sims) + 1) / 2   # map [-1,1] → [0,1]
    prob0 = 1 - prob1
    result["confidence"] = np.maximum(prob1, prob0)
    eps = 1e-9
    result["pred_entropy"] = -(prob1 * np.log2(prob1 + eps) + prob0 * np.log2(prob0 + eps))

    return result

def find_zero_shot_threshold(df_dev: pd.DataFrame, model, tokenizer, layer: str = "avg_last4", n_thresholds: int = 50) -> tuple[float, float]:
    """
    Cerca il threshold ottimale per la cosine similarity sul dev set,
    massimizzando F1. Ritorna (best_threshold, best_f1).
    """
    res = predict_zero_shot(df_dev, model, tokenizer, layer=layer, threshold=0.0)
    sims   = res["cosine_sim"].values
    labels = df_dev["label"].values
 
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(sims.min(), sims.max(), n_thresholds):
        preds = (sims >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
 
    print(f"  [Threshold search] Best threshold={best_t:.4f}  F1={best_f1:.4f}")
    return best_t, best_f1

# -------------- fine tuning -----------------------------------

def build_training_arguments(output_dir, params: dict, disable_tqdm=True) -> TrainingArguments:

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        num_train_epochs=params["num_train_epochs"],
        gradient_accumulation_steps=1,
        weight_decay=params["weight_decay"],
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        optim="adamw_torch",
        fp16=False if use_bf16 else True,
        bf16=use_bf16,
        tf32=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        disable_tqdm=disable_tqdm,
        group_by_length=True,
        seed=SEED,
    )


def find_hyperparameters(model_name, num_trials=10):

    """
    HPO con Optuna su learning_rate, batch_size, num_train_epochs,
    weight_decay e unfreeze_last_n. Usa il validation split di TRAIN_DATASET.
    """

    print(f"  [HPO] {model_name}  ({num_trials} trials)")
    tokenizer, _ = get_model_for_classification(model_name)
    train_df, eval_df = load_train_eval_df()

    def objective(trial):
        _, model = get_model_for_classification(model_name)

        params = {
            "learning_rate":    trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True),
            #"batch_size":       trial.suggest_categorical("batch_size", [64]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
            "weight_decay":     trial.suggest_float("weight_decay", 0.0, 0.1),
            #"unfreeze_last_n":  trial.suggest_int("unfreeze_last_n", 1, 4),
        }

        model = get_training_model(model, unfreeze_last_n=4) #params["unfreeze_last_n"])

        train_ds = df_to_hf_dataset(train_df, tokenizer)
        eval_ds  = df_to_hf_dataset(eval_df,  tokenizer)

        f_outdir = os.path.join("./hpo_tmp", f"trial_{str(trial.number)}")
        args = build_training_arguments(f_outdir, params)
        trainer = Trainer(
            model=model, 
            args=args,
            train_dataset=train_ds, 
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        result = trainer.evaluate()

        del model
        gc.collect()
        torch.cuda.empty_cache()

        return result["eval_f1"]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED, n_startup_trials=5))
    study.optimize(objective, n_trials=num_trials)
 
    print(f"  [HPO] Best params: {study.best_params}  F1={study.best_value:.4f}")

    return study.best_params


def train_model(model_name: str, output_path: str, params: dict) -> tuple:
    """
    Fine-tuning completo. Ritorna (tokenizer, trained_model, eval_metrics).
    """
    
    print(f"  [Train] {model_name}")
    
    tokenizer, model = get_model_for_classification(model_name)
    
    train_df, eval_df = load_train_eval_df()

    model = get_training_model(model, unfreeze_last_n=4) #params.get("unfreeze_last_n", 2))
    train_ds = df_to_hf_dataset(train_df, tokenizer)
    eval_ds  = df_to_hf_dataset(eval_df,  tokenizer)
 
    args = build_training_arguments(output_path, params, disable_tqdm=False)
    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=train_ds, 
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    print(f"  [Train] Val F1={eval_metrics.get('eval_f1', '?'):.4f}")

    return tokenizer, model, eval_metrics


def fine_tuning(tokenizer, model, output_path, params):

    train_df, eval_df = load_train_eval_df(FINE_TUNING_DATASET, SUBSET)

    train_ds = df_to_hf_dataset(train_df, tokenizer)
    eval_ds  = df_to_hf_dataset(eval_df,  tokenizer)

    args = build_training_arguments(output_path, params, disable_tqdm=False)
    trainer = Trainer(
        model=model, 
        args=args,
        train_dataset=train_ds, 
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    print(f"  [Fine-tuning] Val F1={eval_metrics.get('eval_f1', '?'):.4f}")

    return tokenizer, model, eval_metrics


def predict_finetuned(df: pd.DataFrame, model, tokenizer, batch_size: int = 128) -> pd.DataFrame:
    """
    Inferenza con modello fine-tuned. Restituisce il df con:
    predicted_label, prob_same, prob_diff, confidence, pred_entropy.
    """

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    ds = df_to_hf_dataset(
        df.assign(label=df.get("label", [0] * len(df))), tokenizer
    )
    
    # Rimuoviamo labels per non confondere il Trainer durante il predict
    if "labels" in ds.column_names:
        ds = ds.remove_columns(["labels"])
 
    inf_args = TrainingArguments(
        output_dir="./predictions_tmp",
        per_device_eval_batch_size=batch_size,
        disable_tqdm=False,
        bf16=use_bf16,
        fp16=False if use_bf16 else True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        tf32=True,
        seed=SEED,
    )

    trainer = Trainer(model=model, args=inf_args)
    out     = trainer.predict(ds)
    logits  = out.predictions                               # (N, 2)
 
    probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    prob0, prob1 = probs[:, 0], probs[:, 1]
    eps = 1e-9
 
    result = df.copy()
    result["predicted_label"] = probs.argmax(axis=1)
    result["prob_same"]       = prob1
    result["prob_diff"]       = prob0
    result["confidence"]      = probs.max(axis=1)
    result["pred_entropy"]    = -(
        prob1 * np.log2(prob1 + eps) + prob0 * np.log2(prob0 + eps)
    )
    return result


def _classifier_logits_from_hidden(model, layer_hidden: torch.Tensor) -> torch.Tensor:
    """
    Dato l'hidden state di UN layer (B, seq, hidden), applica la testa di
    classificazione del modello fine-tuned (pooler + classifier) al token CLS,
    riproducendo il percorso forward usato sull'ultimo layer.
    Ritorna i logits (B, num_labels). Architecture-aware.
    """

    mt = model.config.model_type
    dropout = getattr(model, "dropout", None)

    # RoBERTa / XLM-R / SimCSE: RobertaClassificationHead slicea [:,0] internamente
    if mt in ("roberta", "xlm-roberta"):
        return model.classifier(layer_hidden)

    # BERT: BertPooler slicea [:,0] internamente -> dropout -> classifier
    if mt == "bert":
        pooled = model.bert.pooler(layer_hidden)
        if dropout is not None:
            pooled = dropout(pooled)
        return model.classifier(pooled)

    # ALBERT: pooler (Linear) + tanh sul CLS già sliceato -> dropout -> classifier
    if mt == "albert":
        cls = layer_hidden[:, 0]
        pooled = model.albert.pooler_activation(model.albert.pooler(cls))
        if dropout is not None:
            pooled = dropout(pooled)
        return model.classifier(pooled)

    # DeBERTa v2/v3: ContextPooler (self.pooler) slicea [:,0] -> dropout -> classifier
    if mt in ("deberta", "deberta-v2"):
        pooled = model.pooler(layer_hidden)
        if dropout is not None:
            pooled = dropout(pooled)
        return model.classifier(pooled)

    raise ValueError(f"logit lens non configurato per model_type={mt}")


def predict_finetuned_per_layer(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 256,
) -> pd.DataFrame:

    print(f"  [Per-layer FT] logit lens su {model.config.model_type}")
    model = model.to(DEVICE).eval()
    eps = 1e-9
    rows = []
    n = len(df)

    for start in range(0, n, batch_size):
        if start % (batch_size * 10) == 0:
            print(f"    Per-layer FT rows {start}/{n}...")

        chunk = df.iloc[start:start + batch_size]
        s1m, s2m = mark_sentences(chunk)
        enc = tokenizer(
            s1m,
            s2m,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        hs = out.hidden_states
        labels = chunk["label"].tolist() if "label" in chunk.columns else None
        ids = chunk.index.tolist()

        for L, layer_hidden in enumerate(hs):
            with torch.no_grad():
                logits = _classifier_logits_from_hidden(model, layer_hidden.detach())

            logits_np = logits.detach().cpu().numpy()
            probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
            prob_diff, prob_same = probs[:, 0], probs[:, 1]
            preds = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            entropy = -(prob_same * np.log2(prob_same + eps) +
                        prob_diff * np.log2(prob_diff + eps))

            for j in range(len(chunk)):
                r = {
                    "example_id":      int(ids[j]),
                    "layer":           L,
                    "logit_diff":      float(logits_np[j, 0]),
                    "logit_same":      float(logits_np[j, 1]),
                    "prob_same":       float(prob_same[j]),
                    "prob_diff":       float(prob_diff[j]),
                    "predicted_label": int(preds[j]),
                    "confidence":      float(conf[j]),
                    "pred_entropy":    float(entropy[j]),
                }
                if labels is not None:
                    r["label"] = int(labels[j])
                rows.append(r)

    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ------------------ k-fold evaluation --------------------------------------


def _tokenize_kfold(
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 256
) -> Dataset:
    
    s1, s2 = mark_sentences(df)
    data = {"sentence1": s1, "sentence2": s2}
    if "label" in df.columns:
            data["labels"] = df["label"].values
    
    sub = pd.DataFrame(data)
    
    #sub = df[["sentence1", "sentence2", "label"]].copy()
    #sub = sub.rename(columns={"label": "labels"})

    ds = Dataset.from_pandas(sub, preserve_index=False)
    ds = ds.map(
        lambda b: tokenizer(b["sentence1"], b["sentence2"],
                            truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["sentence1", "sentence2"],
    )

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    drop = [c for c in ds.column_names if c not in keep]

    if drop:
        ds = ds.remove_columns(drop)

    return ds


def _hpo_training_arguments(output_dir, lr, max_epochs, batch_size=64):

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch", 
        save_strategy="no", 
        logging_strategy="epoch",
        report_to=[], 
        learning_rate=lr,
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=128,
        num_train_epochs=max_epochs, 
        gradient_accumulation_steps=1,
        weight_decay=0.01, 
        lr_scheduler_type="linear", 
        optim="adamw_torch",
        fp16=False if use_bf16 else True, 
        bf16=use_bf16, 
        tf32=True,
        group_by_length=True, 
        dataloader_num_workers=8, 
        dataloader_pin_memory=True,
        seed=SEED, 
        disable_tqdm=True,
    )


def quick_hpo(model_name, 
              band_df, 
              group_col="lemma", 
              lr_grid=[1e-5, 2e-5, 3e-5, 5e-5], 
              max_epochs: int = 5, 
              subsample: int = 10000,
              dev_cap: int = 4000) -> dict:

    print(f"  [HPO] {model_name}  LR grid={list(lr_grid)}  max_epochs={max_epochs}")
    
    df = band_df.reset_index(drop=True).copy()
    df["label"] = df["label"].astype(int)

    if group_col in df.columns:
        groups = df[group_col].astype(str).values
    else:
        groups = np.arange(len(df))  # fallback: nessun grouping
    
    tr_i, dv_i = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        .split(df, df["label"], groups)
    )

    tr, dv = df.iloc[tr_i].copy(), df.iloc[dv_i].copy()

    if subsample and len(tr) > subsample:
        tr = tr.sample(subsample, random_state=SEED)

    if dev_cap and len(dv) > dev_cap:
        dv = dv.sample(dev_cap, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tr_ds = df_to_hf_dataset(tr, tokenizer)
    dv_ds = df_to_hf_dataset(dv, tokenizer)

    print(f"        train={len(tr)}  dev={len(dv)} (lemma-disjoint)")

    best = {"learning_rate": lr_grid[0], "num_train_epochs": max_epochs,
            "batch_size": 64, "weight_decay": 0.01, "eval_f1_val": -1.0}
    
    for lr in lr_grid:
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )

        model = get_training_model(model, unfreeze_last_n=4)
 
        args = _hpo_training_arguments(f"./hpo_tmp_lr_{lr:.0e}", lr, max_epochs)
        
        trainer = Trainer(model=model, args=args, train_dataset=tr_ds,
                          eval_dataset=dv_ds, compute_metrics=compute_metrics)
        trainer.train()

        evals = [h for h in trainer.state.log_history if "eval_f1" in h]
        if evals:
            top = max(evals, key=lambda h: h["eval_f1"])
            f1, ep = float(top["eval_f1"]), int(round(top.get("epoch", max_epochs)))

            print(f"        LR={lr:.1e}: best eval_f1={f1:.4f} @ epoch {ep}")

            if f1 > best["eval_f1_val"]:
                best = {"learning_rate": float(lr), "num_train_epochs": max(ep, 1),
                        "batch_size": 64, "weight_decay": 0.01, "eval_f1_val": f1}
 
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
 
    print(f"  [HPO] best: LR={best['learning_rate']:.1e}  epochs={best['num_train_epochs']}"
          f"  (val F1={best['eval_f1_val']:.4f})")
    
    return best
    

def _kfold_training_arguments(
        output_folder: str,
        params: dict,
) -> TrainingArguments:
    
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return TrainingArguments(
        output_dir = output_folder,
        eval_strategy= "no",
        save_strategy= "no",
        logging_strategy= "no",
        report_to=[],
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params.get("batch_size", 64),
        per_device_eval_batch_size=128,
        num_train_epochs=params["num_train_epochs"],
        gradient_accumulation_steps=1,
        weight_decay=params["weight_decay"],
        lr_scheduler_type="linear",
        optim="adamw_torch",
        fp16=False if use_bf16 else True,
        bf16=use_bf16,
        tf32=True,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        seed=SEED,
        disable_tqdm=True,
        group_by_length=True,
    )
    

def run_kfold_evaluation(
    test_df,
    model_name: str,
    params: dict,
    output_folder: str,
    k: int = 5,
    unfreeze_last_n: int = 4,
    group_col: str = "lemma",
    max_length: int = 256,
    mode: str = "k-fold",
):
    
    df = test_df.reset_index(drop=True).copy()
    n = len(df)

    assert "label" in df.columns, "test_df should contain column 'label'"

    y = df["label"].astype(int).values
    X = np.zeros((n, 1))

    if group_col and group_col in df.columns:

        groups = df[group_col].astype(str).values
        n_groups = len(np.unique(groups))

        if n_groups < k:
            print(f"    [warn] {n_groups} gruppi < k={k}: using StratifiedKFold (no grouping)")
            split_iter = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, y)
        else:
            split_iter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, y, groups)
    
    else:
        print(f"    [warn] group_col '{group_col}' not found: using StratifiedKFold (no grouping)")
        split_iter = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED).split(X, y)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    full_ds = _tokenize_kfold(df, tokenizer, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer)

    eps = 1e-9
    fold_pred_frames = []
    per_fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(split_iter):
 
        print(f"    -- fold {fold_id + 1}/{k}: train={len(train_idx)}  test={len(test_idx)}")
 
        # modello fresco a ogni fold: nessun carryover di pesi dal fold precedente
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )
        model = get_training_model(model, unfreeze_last_n=unfreeze_last_n)

        train_ds = full_ds.select(train_idx.tolist())
        test_ds  = full_ds.select(test_idx.tolist())

        args = _kfold_training_arguments(os.path.join("./kfold_tmp", f"fold_{fold_id}"), params)
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
        trainer.train()

        out = trainer.predict(test_ds)
        probs = torch.softmax(torch.tensor(out.predictions), dim=-1).numpy()
        prob_diff, prob_same = probs[:, 0], probs[:, 1]
        preds = probs.argmax(axis=1)

        # ricostruisci le righe originali del fold (con lemma/pos/sense/...)
        chunk                    = df.iloc[test_idx].copy()
        chunk["fold"]            = fold_id
        chunk["predicted_label"] = preds
        chunk["prob_same"]       = prob_same
        chunk["prob_diff"]       = prob_diff
        chunk["confidence"]      = probs.max(axis=1)
        chunk["pred_entropy"]    = -(prob_same * np.log2(prob_same + eps) +
                                     prob_diff * np.log2(prob_diff + eps))
        fold_pred_frames.append(chunk)

        yt = chunk["label"].values
        
        try:
            auc = roc_auc_score(yt, prob_same)
        except Exception:
            auc = np.nan
        
        per_fold_rows.append({
            "fold": fold_id, "n": int(len(chunk)),
            "accuracy": accuracy_score(yt, preds),
            "f1_macro": f1_score(yt, preds, average="macro", zero_division=0),
            "roc_auc": auc, "model": model_name, "mode": mode,
        })
        
        print(f"       fold {fold_id + 1}: acc={per_fold_rows[-1]['accuracy']:.4f}  "
              f"f1_macro={per_fold_rows[-1]['f1_macro']:.4f}  auc={auc:.4f}")
 
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    # --- predizioni out-of-fold complete (ogni esempio una volta, ordine originale) ---
    oof = pd.concat(fold_pred_frames, axis=0).sort_index()
    save_predictions(oof, output_folder, model_name, mode)

    # statistiche standard (global/by_pos/by_lemma/by_sense/confidence) con label "k-fold"
    stats = compute_all_stats(oof, model_name, mode)

    # statistiche per-fold (varianza della CV)
    stats["by_fold"] = pd.DataFrame(per_fold_rows)

    save_stats(stats, output_folder, model_name, mode)

    m = stats["by_fold"][["accuracy", "f1_macro", "roc_auc"]].agg(["mean", "std"])
    print(f"  [k-fold] {model_name} — CV {k}-fold (mean ± std):")
    print(f"       accuracy = {m.loc['mean','accuracy']:.4f} ± {m.loc['std','accuracy']:.4f}")
    print(f"       f1_macro = {m.loc['mean','f1_macro']:.4f} ± {m.loc['std','f1_macro']:.4f}")
    print(f"       roc_auc  = {m.loc['mean','roc_auc']:.4f} ± {m.loc['std','roc_auc']:.4f}")

    return stats["global"]

# -------------------- main -------------------------------------------------

def main(test_files: list, output_folder: str, 
         use_best_threshold=True, 
         use_finetuning=False, 
         num_trials=10,
         skip_zeroshot=False,
         skip_finetuning=False,
         skip_kfold=False):

    os.makedirs(output_folder, exist_ok=True)

    # load train and validation
    _, valid_df = load_train_eval_df()
    print("Caricato train e dev set")

    for test_file in test_files:

        file_name = test_file.split("/")[-2]

        current_output_folder = os.path.join(output_folder, file_name)
        os.makedirs(current_output_folder, exist_ok=True)

        # load test
        test_df = load_test_df(test_file)
        print("Caricato test set")

        # raccoglitore di metriche
        all_global_stats = []

        # esecuzione dei modelli
        print("Inizio esecuzione modelli...")
    
        for model_name in MODELS:
            
            print(f"\n{'='*60}")
            print(f"MODELLO: {model_name}")
            print(f"{'='*60}")

            if not skip_zeroshot:

                print("\n[1/3] Zero-shot evaluation")

                tokenizer_zs, model_zs = get_model_for_embedding(model_name)
                model_zs = model_zs.to(DEVICE).eval()

                if use_best_threshold:
                    best_t, _ = find_zero_shot_threshold(valid_df, model_zs, tokenizer_zs)
                else:
                    best_t = 0.5

                #zs_test = predict_zero_shot(test_df, model_name, threshold=best_t)

                zs_test = predict_zero_shot(
                    test_df, model_zs, tokenizer_zs, threshold=best_t
                )
                save_predictions(zs_test, current_output_folder, model_name, f"zeroshot")
                
                zs_stats = compute_all_stats(zs_test, model_name, f"zeroshot", best_t)
                save_stats(zs_stats, current_output_folder, model_name, f"zeroshot")
                
                all_global_stats.append(zs_stats["global"])

                del model_zs, tokenizer_zs
                gc.collect()
                torch.cuda.empty_cache()


            if not skip_finetuning:

                print("\n[2/3] Fine-tuning")

                best_params = find_hyperparameters(model_name, num_trials=num_trials)

                # training
                tokenizer, trained_model, eval_metrics = train_model(
                    model_name, current_output_folder, best_params
                )

                #if use_finetuning:
                #    # fine_tuning
                #    tokenizer, trained_model, eval_metrics = fine_tuning(
                #        tokenizer, trained_model, current_output_folder, best_params
                #    )

                ft_test = predict_finetuned(test_df, trained_model, tokenizer)
                save_predictions(ft_test, current_output_folder, model_name, "finetuned")
                ft_stats = compute_all_stats(ft_test, model_name, "finetuned")
                save_stats(ft_stats, current_output_folder, model_name, "finetuned")

                try:
                    ft_per_layer = predict_finetuned_per_layer(test_df, trained_model, tokenizer)
                    ft_conv = summarize_layer_convergence(ft_per_layer)
                    save_per_layer(ft_per_layer, ft_conv, current_output_folder, model_name, "finetuned")
                except Exception as e:
                    print(f"Non sono riuscito a fare la prediction per layer. Errore: {e}")

                print("Salvataggio parametri usati per finetuning")

                safe_name = model_name.replace("/", "_")
                hfo_path = os.path.join(current_output_folder, f"{safe_name}_best_parameters.json")

                with open(hfo_path, 'w') as f:
                    ujson.dump({**best_params, "eval_f1_val": eval_metrics.get("eval_f1")}, f, indent=2)
                
                all_global_stats.append(ft_stats["global"])

                del trained_model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            if not skip_kfold:

                print("[3/3] k-fold evaluation")

                # find best hp
                best_hp = quick_hpo(model_name, test_df)

                kfold_global = run_kfold_evaluation(
                    test_df=test_df,
                    model_name=model_name,
                    params=best_hp, 
                    output_folder=current_output_folder,
                    k=10,
                    unfreeze_last_n=4,
                    group_col="lemma",
                    max_length=256,
                    mode="k-fold"
                )
                
                all_global_stats.append(kfold_global)

                print("Computazione statistiche")

                print(f"Fine esecuzione modello {model_name}. Pulizia memoria")

        print(f"\n{'='*60}")
        print("Esecuzione modelli terminata. Confronto i risultati")
        print(f"{'='*60}")

        if all_global_stats:
            
            comparison = pd.concat(all_global_stats, ignore_index=True)
        
            comp_path  = os.path.join(current_output_folder, "comparison_all_models.csv")
            comparison.to_csv(comp_path, index=False)
            
            print(f"\nConfronto globale → {comp_path}")
            print(comparison[["model", "mode", "f1", "accuracy", "roc_auc", "n_examples"]].to_string(index=False))

    print("Fine computazione di tutti i modelli")


if __name__ == "__main__":


    #TEST_FILES_IT = ["data/xl_wic/it/pairs.jsonl",
    #              "data/xl_wic/it/pairs_low.jsonl",
    #              "data/it_he/wic_pairs.jsonl",
    #              "data/it_le/wic_pairs.jsonl"
    #            ]

    TEST_FILES_ENG = [
        "data/high/wic_pairs_balanced.jsonl",
        "data/medium/wic_pairs_balanced.jsonl",
        "data/low/wic_pairs_balanced.jsonl",
        "data/john_wic/john_wic.jsonl"
    ]
    
    OUTPUT_FOLDER = "data/10_k_fold"
    
    main(TEST_FILES_ENG, OUTPUT_FOLDER, 
         use_best_threshold=True, 
         use_finetuning=False, 
         num_trials=10, 
         skip_zeroshot=True, 
         skip_finetuning=True, 
         skip_kfold=False)
