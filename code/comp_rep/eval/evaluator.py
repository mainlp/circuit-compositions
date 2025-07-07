"""
Modules to evaluate models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from comp_rep.data_prep.dataset import (
    PAD_TOKEN,
    CollateFunctor,
    CollateFunctorWithProbabilities,
    SequenceDataset,
    SequenceDatasetWithProbabilities,
)
from comp_rep.eval.cross_task_evaluations import compute_str_match_mask
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.metrics import jsd_faithfulness, kl_divergence


@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    searcher: GreedySearch,
    test_loader: DataLoader,
    predictions_path: Optional[Path] = None,
    device: Optional[str] = "cuda:0",
):
    """
    Generates predictions and evaluates them on the provided test loader.

    Args:
        model (nn.Module): The model to use for generation.
        searcher (GreedySearch): The search method for generating predictions.
        test_loader (DataLoader): The data loader for the test data.
        predictions_path (Path): The path to save the predictions.
        device (str): The device to run the model on.

    Returns:
        float: The accuracy of the predictions.
    """
    corrects = 0
    n = 0
    targets_l = []
    predictions_l = []
    outs = []
    model.to(device)
    for batch in tqdm(test_loader):
        source_ids = batch[0].to(
            device
        )  # With indexing we do not have to do a type check on the dataset type
        source_mask = batch[1].to(device)
        target_str = batch[-1]
        sentences, _ = searcher(source_ids, source_mask)
        for t, p in zip(target_str, sentences):
            t = t.strip()
            p = p.strip()
            c = t == p
            if c:
                corrects += 1
            n += 1
            targets_l.append(t)
            predictions_l.append(p)
            outs.append(t + "," + p + "," + str(c))

    if predictions_path is not None:
        try:
            Path(predictions_path).mkdir(parents=True, exist_ok=True)
            with open(predictions_path / "predictions.csv", "w") as f:
                f.write("\n".join(outs))
        except IOError as e:
            print(f"Failed to save predictions to file.., {e}")
        finally:
            return corrects / n

    return corrects / n


@torch.no_grad()
def evaluate_task_faithfulness(
    model: nn.Module,
    test_loader: DataLoader,
    device: Union[str, torch.device] = "cuda:0",
    mask_func_equivalence: bool = False,
    circuit_name: str = "copy",
    eval_task_name: str = "copy",
) -> Tuple[float, float]:
    """
    Evaluates the faithfulness of a given model on a test dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): The test dataset loader.
        device (Union[str, torch.device], optional): The device to be used for evaluation. Defaults to "cuda:0".

    Returns:
        Tuple[float, float]: The average faithfulness score of the model over the test set wrt jsd and kl-divergence.
    """
    model = model.to(device)
    model.eval()

    total_jsd_faithfulness_score: float = 0.0
    total_kl_div_faithfulness_score: float = 0.0
    n_batches: int = 0

    for batch in tqdm(test_loader):
        (
            source_ids,
            source_mask,
            target_ids,
            target_mask,
            target_probabilities,
            source_str,
            target_str,
        ) = batch

        # Move tensors to device
        source_ids = source_ids.to(device)
        source_mask = source_mask.to(device)
        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)
        target_probabilities = target_probabilities.to(device)

        # Mask pad tokens
        pad_mask = (target_ids[:, 1:] != PAD_TOKEN).float()

        # Mask equivalent tokens for cross-task-faithfulness
        if mask_func_equivalence:
            func_mask = compute_str_match_mask(
                source_str=source_str,
                target_str=target_str,
                max_seq_len=target_probabilities.shape[-2],
                circuit_name=circuit_name,
                eval_task_name=eval_task_name,
            ).to(device)
            pad_mask = pad_mask * func_mask

        # Skip if all tokens would be padded in batch
        if pad_mask.sum() == 0:
            logging.warning("Pad mask was zero for all entries!")
            continue

        # Left shift the targets so that the last token predicts the EOS
        model_logits = model(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        )  # [batch size, max seq len, vocab]

        # jsd
        jsd_faithfulness_score = jsd_faithfulness(
            p_logits=model_logits,
            q_probs=target_probabilities,
            eps=1e-10,
            mask=pad_mask,
            sqrt=False,
        )
        total_jsd_faithfulness_score += jsd_faithfulness_score

        # kl divergence
        kl_div_faithfulness_score = kl_divergence(
            p_logits=model_logits,
            q_probs=target_probabilities,
            eps=1e-10,
            mask=pad_mask,
        )
        total_kl_div_faithfulness_score += kl_div_faithfulness_score
        n_batches += 1

    if n_batches == 0:
        logging.warning("No batches have been processed!")
        return float("nan"), float("nan")

    batch_jsd_faithfulness_score = total_jsd_faithfulness_score / n_batches
    batch_kl_div_faithfulness_score = total_kl_div_faithfulness_score / n_batches

    return batch_jsd_faithfulness_score, batch_kl_div_faithfulness_score


def eval_task(
    task_name: str,
    model: nn.Module,
    tokenizer: Dict,
    device: str,
    output_dir: Path,
    eval_data_path: Path,
    cached_probabilities_path: Path = Path(),
    eval_acc: bool = True,
    eval_faithfulness: bool = False,
    mask_func_equivalence: bool = False,
    circuit_name: str = "copy",
    eval_task_name: str = "copy",
) -> Dict[str, float]:
    """
    Evaluates the performance of a given model on a specific task.

    Args:
        task_name (str): The name of the task to be evaluated.
        model (nn.Module): The model to be evaluated.
        tokenizer (Dict): The tokenizer used for the task.
        device (str): The device to be used for evaluation.
        eval_data_path (Path): The path to the evaluation data.
        output_dir (Path): The directory where the evaluation results will be saved.

    Returns:
        Dict[str, float]: The accuracy and/or faithfulness of the model on the given task.
    """
    logging.info(f"Evaluating function: {task_name}")

    eval_dict: Dict[str, float] = {}
    if not eval_acc and not eval_faithfulness:
        logging.warning(
            "task evaluation will be skipped as both 'eval_acc' and 'eval_faithfulness' are False!"
        )
        return eval_dict

    if eval_faithfulness:
        eval_dataset: SequenceDataset | SequenceDatasetWithProbabilities = (
            SequenceDatasetWithProbabilities(
                path=eval_data_path,
                probabilities_path=cached_probabilities_path,
                tokenizer=tokenizer,
            )
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            collate_fn=CollateFunctorWithProbabilities(),
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
        )

        jsd_faithfulness, kl_div_faithfulness = evaluate_task_faithfulness(
            model=model,
            test_loader=eval_loader,
            device=device,
            mask_func_equivalence=mask_func_equivalence,
            circuit_name=circuit_name,
            eval_task_name=eval_task_name,
        )
        eval_dict["jsd_faithfulness"] = jsd_faithfulness
        eval_dict["kl_div_faithfulness"] = kl_div_faithfulness
    else:
        eval_dataset = SequenceDataset(eval_data_path, tokenizer=tokenizer)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            collate_fn=CollateFunctor(),
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
        )

    if eval_acc:
        searcher = GreedySearch(model, eval_dataset.output_language)
        accuracy = evaluate_generation(model, searcher, eval_loader, output_dir, device)
        eval_dict["accuracy"] = accuracy

    return eval_dict
