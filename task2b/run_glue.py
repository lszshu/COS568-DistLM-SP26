# coding=utf-8
"""Task 2(b) distributed fine-tuning with all_reduce gradient synchronization."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    WarmupLinearSchedule,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils_glue import compute_metrics, convert_examples_to_features, output_modes, processors

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)


def is_distributed(args):
    return args.local_rank != -1 and args.world_size > 1


def is_main_process(args):
    return args.local_rank in [-1, 0]


def rank_id(args):
    return args.local_rank if args.local_rank != -1 else 0


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as writer:
        json.dump(payload, writer, indent=2, sort_keys=True)


def deliverables_dir(args):
    return os.path.join(args.output_dir, "deliverables")


def traces_dir(args):
    return os.path.join(args.output_dir, "traces")


def average_excluding_first(values):
    if len(values) <= 1:
        return None
    return sum(values[1:]) / float(len(values) - 1)


def compute_profile_window(values):
    return values[1:min(len(values), 4)]


def summarize_profiler(prof, args, train_history):
    trace_path = os.path.join(traces_dir(args), "task2b_rank{}.json".format(rank_id(args)))
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    prof.export_chrome_trace(trace_path)

    comm_keywords = ("all_reduce", "allreduce", "gloo", "c10d", "ProcessGroup", "nccl")
    communication_events = []
    for event in prof.key_averages():
        if any(keyword.lower() in event.key.lower() for keyword in comm_keywords):
            communication_events.append({
                "name": event.key,
                "cpu_time_total_us": event.cpu_time_total,
                "self_cpu_time_total_us": event.self_cpu_time_total,
                "count": event.count,
            })

    profiled_sync_overhead_pct = []
    for iteration_time, sync_time in zip(
        compute_profile_window(train_history["iteration_times_sec"]),
        compute_profile_window(train_history["sync_times_sec"]),
    ):
        profiled_sync_overhead_pct.append(100.0 * sync_time / iteration_time if iteration_time > 0 else None)

    return {
        "trace_file": trace_path,
        "communication_events": communication_events,
        "profiled_iteration_times_sec": compute_profile_window(train_history["iteration_times_sec"]),
        "profiled_sync_times_sec": compute_profile_window(train_history["sync_times_sec"]),
        "profiled_sync_overhead_pct": profiled_sync_overhead_pct,
    }


def save_deliverables(args, train_history, final_eval_results, profiler_summary=None):
    output_dir = deliverables_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    rank = rank_id(args)
    save_json(
        os.path.join(output_dir, "loss_curve_rank{}.json".format(rank)),
        train_history["loss_curve"],
    )
    save_json(
        os.path.join(output_dir, "summary_rank{}.json".format(rank)),
        {
            "rank": rank,
            "world_size": args.world_size,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "total_train_batch_size": args.per_device_train_batch_size * args.world_size,
            "first_five_losses": train_history["first_five_losses"],
            "iteration_times_sec": train_history["iteration_times_sec"],
            "average_iteration_time_excluding_first_sec": average_excluding_first(
                train_history["iteration_times_sec"]
            ),
            "sync_times_sec": train_history["sync_times_sec"],
            "average_sync_time_excluding_first_sec": average_excluding_first(
                train_history["sync_times_sec"]
            ),
            "epoch_eval_history": train_history["epoch_eval_history"],
            "final_eval_results": final_eval_results,
            "profiler": profiler_summary,
        },
    )


def train(args, train_dataset, model, tokenizer):
    """Train the model."""

    args.train_batch_size = args.per_device_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    if is_distributed(args):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True,
        )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_steps,
        t_total=t_total,
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per device = %d", args.per_device_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)
    train_history = {
        "loss_curve": [],
        "first_five_losses": [],
        "iteration_times_sec": [],
        "sync_times_sec": [],
        "epoch_eval_history": [],
    }
    prof = None
    profiler_summary = None
    if args.profile_task4:
        prof = profile(
            activities=[ProfilerActivity.CPU],
            schedule=schedule(wait=1, warmup=0, active=3, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.start()
    for epoch in train_iterator:
        if is_distributed(args):
            train_sampler.set_epoch(epoch)
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=args.local_rank not in [-1, 0],
        )
        for step, batch in enumerate(epoch_iterator):
            iteration_start = time.perf_counter()
            sync_time = 0.0
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                "labels": batch[3],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            raw_loss = loss.item()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if is_distributed(args):
                    sync_start = time.perf_counter()
                    for param in model.parameters():
                        if param.grad is None:
                            continue
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= args.world_size
                    sync_time = time.perf_counter() - sync_start
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            iteration_time = time.perf_counter() - iteration_start
            train_history["loss_curve"].append(
                {
                    "iteration": len(train_history["loss_curve"]) + 1,
                    "epoch": epoch + 1,
                    "step_in_epoch": step + 1,
                    "loss": raw_loss,
                }
            )
            train_history["iteration_times_sec"].append(iteration_time)
            train_history["sync_times_sec"].append(sync_time)
            if prof is not None:
                prof.step()
            if len(train_history["first_five_losses"]) < 5:
                train_history["first_five_losses"].append(raw_loss)
                logger.info(
                    "rank=%d epoch=%d step=%d loss=%.6f",
                    rank_id(args),
                    epoch + 1,
                    step + 1,
                    raw_loss,
                )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.do_eval and is_main_process(args):
            epoch_eval_results = evaluate(args, model, tokenizer, prefix="epoch_{}".format(epoch + 1))
            train_history["epoch_eval_history"].append(
                {
                    "epoch": epoch + 1,
                    "metrics": epoch_eval_results,
                }
            )

    if prof is not None:
        prof.stop()
        profiler_summary = summarize_profiler(prof, args, train_history)

    return global_step, tr_loss / global_step, train_history, profiler_summary


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (
        (args.output_dir, args.output_dir + "-MM")
        if args.task_name == "mnli"
        else (args.output_dir,)
    )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and is_main_process(args):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2] if args.model_type in ["bert", "xlnet"] else None,
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        result["eval_loss"] = eval_loss
        results.update(result)

        output_suffix = prefix if prefix else "final"
        output_eval_file = os.path.join(eval_output_dir, "eval_results_{}.txt".format(output_suffix))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results %s *****", prefix)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    sync_cache = is_distributed(args) and not evaluate
    if sync_cache and args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta"]:
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if sync_cache and args.local_rank == 0:
        torch.distributed.barrier()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_ip", type=str, default="")
    parser.add_argument("--master_port", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--profile_task4", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    if args.local_rank == -1:
        args.world_size = 1
    else:
        if not args.master_ip or args.master_port <= 0:
            raise ValueError("Distributed training requires --master_ip, --master_port, and --world_size.")
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://{}:{}".format(args.master_ip, args.master_port),
            world_size=args.world_size,
            rank=args.local_rank,
        )
    os.makedirs(deliverables_dir(args), exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    train_history = {
        "loss_curve": [],
        "first_five_losses": [],
        "iteration_times_sec": [],
        "sync_times_sec": [],
        "epoch_eval_history": [],
    }
    profiler_summary = None
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss, train_history, profiler_summary = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    final_eval_results = {}
    if args.do_eval and is_main_process(args):
        final_eval_results = evaluate(args, model, tokenizer, prefix="final")
    save_deliverables(args, train_history, final_eval_results, profiler_summary)


if __name__ == "__main__":
    main()
