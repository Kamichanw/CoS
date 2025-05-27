import os
import importlib
from typing import Callable, Literal, Optional
import warnings
import torch
import torch_npu
from pprint import pprint
from tqdm import tqdm
import hydra
import numpy as np
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from omegaconf import DictConfig, OmegaConf
from src.myutils.file import write_json

draft_model: PreTrainedModel
target_model: PreTrainedModel
tokenizer: PreTrainedTokenizer
device = "npu:0" if torch.npu.is_available() else "cpu"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def propose(
    input_ids: torch.Tensor,
    draft_model: PreTrainedModel,
    gamma: int,
    return_logits: bool = False,
):
    """
    Propose a sequence of tokens using the draft model with greedy decoding.
    Args:
        input_ids (torch.Tensor): Input tensor of shape (batch, seq_len).
        draft_model (PreTrainedModel): The draft model to use for token generation.
        gamma (int): Number of tokens to draft.
        return_logits (bool): Whether to return logits besides the drafted tokens.
    """
    draft_tokens_collector = []
    temp_draft_input_ids = input_ids.clone()

    for _ in range(gamma):
        if temp_draft_input_ids.shape[1] >= tokenizer.model_max_length:
            print("Warning: Draft phase reached model_max_length.")
            break
        draft_outputs = draft_model(temp_draft_input_ids)
        next_token_logits = draft_outputs.logits[:, -1, :]
        next_draft_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        draft_tokens_collector.append(next_draft_token)
        temp_draft_input_ids = torch.cat(
            [temp_draft_input_ids, next_draft_token], dim=1
        )

        if next_draft_token.item() == tokenizer.eos_token_id:
            break
    if return_logits:
        return torch.cat(draft_tokens_collector, dim=1), draft_outputs.logits
    return torch.cat(draft_tokens_collector, dim=1)  # (batch, num_drafted)


def score(
    current_generated_ids: torch.Tensor,
    drafted_tokens_tensor: torch.Tensor,
    target_model: PreTrainedModel,
):
    """
    Score the input sequence using the target model.
    Args:
        input_ids (torch.Tensor): Input tensor of shape (batch, seq_len).
        target_model (PreTrainedModel): The target model to use for scoring.

    Returns:
        torch.Tensor: Logits of proposal tokens from the target model.
        torch.Tensor: bonus token.
    """
    verification_input_ids = torch.cat(
        [current_generated_ids, drafted_tokens_tensor], dim=1
    )
    assert verification_input_ids.shape[1] <= tokenizer.model_max_length
    target_outputs = target_model(verification_input_ids)
    target_logits = (
        target_outputs.logits
    )  # (batch, L_current + num_drafted, vocab_size)

    bonus_token = torch.argmax(
        target_logits[:, -1, :], dim=-1, keepdim=True
    )  # (batch, 1)

    return target_logits, bonus_token


def verify(
    current_generated_ids: torch.Tensor,
    drafted_tokens_tensor: torch.Tensor,
    ensemble_logits_or_probs: torch.Tensor,
):
    """
    Verify the drafted tokens against the target model's predictions, and re-sample if need.
    Args:
        current_generated_ids (torch.Tensor): Current generated sequence tensor of shape (batch, seq_len).
        drafted_tokens_tensor (torch.Tensor): Drafted tokens tensor of shape (batch, num_drafted).
        ensemble_logits_or_probs (torch.Tensor): Logits or probabilities from the ensemble distribution.
    Returns:
        torch.Tensor: The updated generated sequence tensor after verification. If all drafted tokens are accepted,
            it will return the input sequence.
        int: The number of accepted tokens.
    """
    num_drafted = drafted_tokens_tensor.shape[1]
    num_accepted_tokens = 0
    for i in range(num_drafted):
        current_logits_or_probs = ensemble_logits_or_probs[
            :, current_generated_ids.shape[1] + i - 1, :
        ]
        target_predicted_token_at_draft_pos = torch.argmax(
            current_logits_or_probs, dim=-1
        )

        if target_predicted_token_at_draft_pos == drafted_tokens_tensor[:, i]:
            num_accepted_tokens += 1
        else:
            # rejected: accept drafted tokens up to this point, plus the correction token from target model
            accepted_draft_part = drafted_tokens_tensor[:, :num_accepted_tokens]
            corrected_token = target_predicted_token_at_draft_pos.unsqueeze(-1)

            current_generated_ids = torch.cat(
                [current_generated_ids, accepted_draft_part, corrected_token], dim=1
            )
            if corrected_token.item() == tokenizer.eos_token_id:
                return current_generated_ids, num_accepted_tokens
            break

    return current_generated_ids, num_accepted_tokens


def ensemble(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    ensemble_target: Literal["logits", "probs", "raw_logits"],
    ensemble_fn: Callable,
):
    """
    Ensemble the logits or probabilities from the draft and target models.
    Args:
        draft_logits (torch.Tensor): Logits from the draft model.
        target_logits (torch.Tensor): Logits from the target model.
        ensemble_target (str): Target for ensembling, can be "logits", "probs", or "raw_logits".
        ensemble_fn (Callable): Function to perform the ensembling operation.

    Returns:
        torch.Tensor: The ensembled logits or probabilities.
    """
    if ensemble_target in ["logits", "raw_logits"]:
        ensemble_logits_or_probs = ensemble_fn([draft_logits, target_logits])
    elif ensemble_target == "probs":
        draft_probs = torch.softmax(draft_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)
        target_probs, _ = ensemble_fn(
            [draft_probs, target_probs],
            [
                torch.log_softmax(draft_logits, dim=-1),
                torch.log_softmax(target_logits, dim=-1),
            ],
        )
        ensemble_logits_or_probs = target_probs
    else:
        raise ValueError(f"ensemble_target {ensemble_target} not supported")
    return ensemble_logits_or_probs


@torch.no_grad()
def speculative_decode(
    initial_prompt_text: str,
    max_new_tokens: int,
    gamma: int,
    ensemble_target: Literal["logits", "probs", "raw_logits"] = "logits",
    ensemble_fn: Optional[Callable] = None,
):
    global draft_model, target_model, tokenizer, device
    prompt_input_ids = tokenizer.encode(initial_prompt_text, return_tensors="pt").to(
        device
    )
    current_generated_ids = prompt_input_ids.clone()
    initial_len = current_generated_ids.shape[1]

    generated_token_count = 0

    while generated_token_count < max_new_tokens:
        # 1. draft phase: use draft model to generate gamma tokens
        drafted_tokens_tensor = propose(current_generated_ids, draft_model, gamma)
        num_drafted = drafted_tokens_tensor.shape[1] # type: ignore

        # 2. verification phase: target model processes original sequence + drafted tokens
        target_logits, _ = score(
            current_generated_ids,
            drafted_tokens_tensor,
            target_model,
        )

        if ensemble_fn is not None:
            draft_outputs = draft_model(
                torch.cat([current_generated_ids, drafted_tokens_tensor], dim=1)
            )
            ensemble_logits_or_probs = ensemble(
                draft_outputs.logits, target_logits, ensemble_target, ensemble_fn
            )

        else:
            ensemble_logits_or_probs = target_logits

        # 3. accepted phase: check drafted tokens against target model's predictions
        current_generated_ids, num_accepted_tokens = verify(
            current_generated_ids,
            drafted_tokens_tensor,
            ensemble_logits_or_probs,
        )
        all_drafted_accepted = num_accepted_tokens == num_drafted
        if all_drafted_accepted:
            current_generated_ids = torch.cat(
                [current_generated_ids, drafted_tokens_tensor], dim=1
            )
            generated_token_count += num_drafted
            if num_drafted > 0:
                final_ensemble_logits = ensemble_logits_or_probs[:, -1, :]
                bonus_target_token = torch.argmax(
                    final_ensemble_logits, dim=-1, keepdim=True
                )
                current_generated_ids = torch.cat(
                    [current_generated_ids, bonus_target_token], dim=1
                )
                generated_token_count += 1
                if bonus_target_token.item() == tokenizer.eos_token_id:
                    return current_generated_ids
        else:
            generated_token_count += num_accepted_tokens + 1

        # 4. final check: whether we have enough tokens or hit EOS
        if generated_token_count >= max_new_tokens:
            break
        if (
            tokenizer.eos_token_id is not None
            and tokenizer.eos_token_id
            in current_generated_ids[
                0,
                initial_len
                + (
                    generated_token_count
                    - (num_accepted_tokens + int(all_drafted_accepted))
                ) :,
            ]
        ):
            eos_indices = (
                current_generated_ids[0, initial_len:] == tokenizer.eos_token_id
            ).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                first_eos_idx_in_new = eos_indices[0]
                current_generated_ids = current_generated_ids[
                    :, : initial_len + first_eos_idx_in_new + 1
                ]
                break

    return current_generated_ids[:, initial_len : initial_len + max_new_tokens]


# See paper B.1, we only implement 2-model ensemble here
@torch.no_grad()
def chef_decode(
    initial_prompt_text: str,
    max_new_tokens: int,
    gamma: int,
    ensemble_target: Literal["logits", "probs", "raw_logits"],
    ensemble_fn: Callable,
):
    global draft_model, target_model, tokenizer, device
    prompt_input_ids = tokenizer.encode(initial_prompt_text, return_tensors="pt").to(
        device
    )
    current_generated_ids = prompt_input_ids.clone()
    initial_len = current_generated_ids.shape[1]

    generated_token_count = 0
    cached_proposals = None
    cached_logits = None

    while generated_token_count < max_new_tokens:
        # 1. draft phase: use draft model to generate gamma tokens
        if cached_proposals is None:
            proposer, verifier = draft_model, target_model
            proposer_gamma, verifier_gamma = gamma, 1
            drafted_tokens_tensor, draft_logits = propose(
                current_generated_ids, proposer, proposer_gamma, return_logits=True
            )
        else:
            verifier, proposer = proposer, verifier
            verifier_gamma, proposer_gamma = proposer_gamma, verifier_gamma
            if proposer_gamma - cached_proposals.shape[1] > 0:
                new_drafted_tokens, new_draft_logits = propose(
                    torch.cat([current_generated_ids, cached_proposals], dim=1),
                    proposer,
                    proposer_gamma - cached_proposals.shape[1],
                    return_logits=True
                )
                drafted_tokens_tensor = torch.cat(
                    [
                        cached_proposals,
                        new_drafted_tokens,
                    ],
                    dim=1,
                )
                draft_logits = new_draft_logits
            else:
                drafted_tokens_tensor = cached_proposals
                draft_logits = cached_logits

        # 2. verification phase: target model processes original sequence + drafted tokens
        target_logits, bonus_token = score(
            current_generated_ids,
            drafted_tokens_tensor,
            verifier,
        )
        ensemble_logits_or_probs = ensemble(
            draft_logits=draft_logits,
            target_logits=target_logits[:, :-1, :],
            ensemble_target=ensemble_target,
            ensemble_fn=ensemble_fn,
        )
        # 3. accepted phase: check drafted tokens against target model's predictions
        current_generated_ids, num_accepted_tokens = verify(
            current_generated_ids,
            drafted_tokens_tensor,
            ensemble_logits_or_probs,
        )
        all_drafted_accepted = num_accepted_tokens == drafted_tokens_tensor.shape[1]
        if all_drafted_accepted:
            current_generated_ids = torch.cat(
                [current_generated_ids, drafted_tokens_tensor], dim=1
            )
            generated_token_count += drafted_tokens_tensor.shape[1]
            cached_proposals = bonus_token
            cached_logits = ensemble_logits_or_probs[:, -1, :].unsqueeze(1)
        else:
            generated_token_count += num_accepted_tokens + 1
            cached_proposals, cached_logits = None, None

        # 4. final check: whether we have enough tokens or hit EOS
        if generated_token_count >= max_new_tokens:
            break
        if (
            tokenizer.eos_token_id is not None
            and tokenizer.eos_token_id
            in current_generated_ids[
                0,
                initial_len
                + (
                    generated_token_count
                    - (num_accepted_tokens + int(all_drafted_accepted))
                ) :,
            ]
        ):
            eos_indices = (
                current_generated_ids[0, initial_len:] == tokenizer.eos_token_id
            ).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                first_eos_idx_in_new = eos_indices[0]
                current_generated_ids = current_generated_ids[
                    :, : initial_len + first_eos_idx_in_new + 1
                ]
                break
    return current_generated_ids[:, initial_len : initial_len + max_new_tokens]


@torch.no_grad()
def standard_decode(
    initial_prompt_text: str,
    max_new_tokens: int,
    gamma: int,
    ensemble_target: Literal["logits", "probs", "raw_logits"] = "logits",
    ensemble_fn: Optional[Callable] = None,
):
    global draft_model, target_model, tokenizer, device
    prompt_input_ids = tokenizer.encode(initial_prompt_text, return_tensors="pt").to(
        device
    )
    current_generated_ids = prompt_input_ids.clone()
    initial_len = current_generated_ids.shape[1]

    generated_token_count = 0

    while generated_token_count < max_new_tokens:

        drafted_tokens_tensor = propose(current_generated_ids, draft_model, gamma=1, return_logits=True)
        _, draft_logits = drafted_tokens_tensor
        target_tokens_tensor = propose(current_generated_ids, target_model, gamma=1, return_logits=True)
        _, target_logits = target_tokens_tensor

        if ensemble_fn is not None:
            ensemble_logits_or_probs = ensemble(
                draft_logits, target_logits, ensemble_target, ensemble_fn
            )
            next_token = torch.argmax(
                ensemble_logits_or_probs[:, -1, :], dim=-1, keepdim=True
            )
        else:
            raise ValueError(
                "Ensemble function is required for standard decode, please provide a valid ensemble_fn."
            )
        
        current_generated_ids = torch.cat(
            [current_generated_ids, next_token], dim=1
        )
        generated_token_count += 1

        # 4. final check: whether we have enough tokens or hit EOS
        if generated_token_count >= max_new_tokens:
            break
        if (
            tokenizer.eos_token_id is not None
            and tokenizer.eos_token_id == next_token.item()
        ):
            break

    return current_generated_ids[:, initial_len : initial_len + max_new_tokens]


def warpped_sampling(
    decode_type,
    prompts,
    enable_test_speed,
    max_new_tokens,
    gamma=5,
    ensemble_target="logits",
    ensemble_fn=None,
):
    if enable_test_speed:
        starter, ender = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True
        )
    
    if decode_type == "chef":
        decode_method = chef_decode
    elif decode_type == "sd":
        decode_method = speculative_decode
    else:
        decode_method = standard_decode

    results = {
        "generated": [],
        "total_time": [],
        "num_tokens": [],
        "num_tokens_per_sec": [],
    }

    for prompt in tqdm(prompts, desc="Processing prompts"):
        if enable_test_speed:
            starter.record()
            output = decode_method(
                initial_prompt_text=prompt,
                max_new_tokens=max_new_tokens,
                gamma=gamma,
                ensemble_target=ensemble_target,  # type: ignore
                ensemble_fn=ensemble_fn,
            )
            ender.record()
            torch.npu.synchronize()

            results["num_tokens"].append(len(output[0]))
            results["total_time"].append(starter.elapsed_time(ender) / 1000)
            results["num_tokens_per_sec"].append(
                results["num_tokens"][-1] / results["total_time"][-1]
            )
        else:
            output = decode_method(
                initial_prompt_text=prompt,
                max_new_tokens=max_new_tokens,
                gamma=gamma,
                ensemble_target=ensemble_target,  # type: ignore
                ensemble_fn=ensemble_fn,
            )
            results["num_tokens"].append(len(output[0]))

        results["generated"].append(
            tokenizer.decode(output[0], skip_special_tokens=True)
        )
        # pprint(results["generated"][-1])

    return results


def init_dataset(args):
    MyDataset = importlib.import_module(
        f"src.mydatasets.{args.dataset.name}.mydataset"
    ).MyDataset
    dataset = MyDataset(size=args.dataset.size, use_fewshot=args.dataset.use_fewshot)
    return dataset


def preprocess_cfg(cfg):
    OmegaConf.set_struct(cfg, False)
    if "extra_model" in cfg.method and cfg.method.extra_model is not None:
        if isinstance(cfg.method.extra_model, str):
            extra_models = [cfg.method.extra_model]
        else:
            extra_models = cfg.method.extra_model
        if len(extra_models) > 1:
            warnings.warn(
                "More than one models are provided, we will override ensemble_fn to (...) / num_total_models"
            )
            if cfg.method.ensemble_target in ["logits", "raw_logits"]:
                cfg.method["ensemble_fn"] = (
                    f"${{eval:'lambda logits: sum(logits) / {len(extra_models) + 1}'}}"
                )
            elif cfg.method.ensemble_target in ["probs"]:
                cfg.method["ensemble_fn"] = (
                    f"${{eval:'lambda probs, logprobs: (sum(probs) / {len(extra_models) + 1}, logprobs[0])'}}"
                )
            else:
                raise ValueError(
                    f"ensemble_target {cfg.method.ensemble_target} not supported"
                )
        else:
            if cfg.method.ensemble_target in ["logits", "raw_logits"]:
                cfg.method["ensemble_fn"] = (
                    f"${{eval:'lambda logits: logits[1] - {cfg.method['alpha']} * logits[0]'}}"
                )
            elif cfg.method.ensemble_target in ["probs"]:
                cfg.method["ensemble_fn"] = (
                    f"${{eval:'lambda probs, logprobs: ((1 - {cfg.method['lambda']}) * probs[0] + {cfg.method['lambda']} * probs[1], logprobs[0])'}}"
                )
            else:
                raise ValueError(
                    f"ensemble_target {cfg.method.ensemble_target} not supported"
                )


def process_result(cfg, results, evaluate_func):
    if cfg.test_speed:
        results_stats = {
            "performance": evaluate_func(results["generated"]),
            "num_tokens_per_sec": np.mean(results["num_tokens_per_sec"]),
            "total_time": np.mean(results["total_time"]),
            "num_tokens": np.mean(results["num_tokens"]),
        }
    else:
        results_stats = {
            "performance": evaluate_func(results["generated"]),
            "num_tokens": np.mean(results["num_tokens"]),
        }

    pprint(results_stats)

    if cfg.save_path:
        write_json(cfg.save_path, results)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):

    preprocess_cfg(cfg)
    torch.manual_seed(cfg.seed)

    dataset = init_dataset(cfg)
    prompts = dataset.prompts

    target_model_path = os.path.join(os.environ["MODEL_PATH"], cfg.method.model)
    draft_model_path = os.path.join(
        os.environ["MODEL_PATH"],
        (
            cfg.method.draft_model
            if "draft_model" in cfg.method
            else cfg.method.extra_model  # extra_model must be a str, list case is not implemented
        ),
    )

    global draft_model, target_model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)  # type: ignore
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16).to(device)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path, torch_dtype=torch.float16).to(device)

    results = warpped_sampling(
        cfg.method.type,
        prompts,
        enable_test_speed=cfg.test_speed,
        max_new_tokens=cfg.method.generate.max_new_tokens,
        gamma=cfg.method.gamma,
        ensemble_target=getattr(cfg.method, "ensemble_target", "logits"),
        ensemble_fn=getattr(cfg.method, "ensemble_fn", None),
    )

    process_result(cfg, results, dataset.evaluate)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
