import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

from dataloader_kbrd import KBRDDataCollatorForRec
from dataset import DatasetForRec
from kg_kbrd import KGForKBRD
from metric import RecMetric
from model_kbrd import KBRDforRec
from utils import load_jsonl_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--kg_dataset", type=str, required=True, default='redial')
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--other_dataset", type=str, nargs='*')
    parser.add_argument("--other_shot", type=float, default=1)
    parser.add_argument("--course_idx", type=int)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--context_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    # model
    parser.add_argument("--dialog_encoder", type=str)
    parser.add_argument("--rec_model", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--teacher", type=str)
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--use_lr_scheduler", action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    # wandb
    run = None
    if args.use_wandb:
        name = args.name if args.name else local_time
        run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)

    # kg
    kg = KGForKBRD(kg_dataset=args.kg_dataset, debug=args.debug).get_kg_info()
    edge_index, edge_type = torch.as_tensor(kg['edge_index'], device=device), torch.as_tensor(kg['edge_type'],
                                                                                              device=device)

    # model
    crs_rec_model = KBRDforRec(
        hidden_size=args.hidden_size,
        num_relations=kg['num_relations'], num_bases=args.num_bases, num_entities=kg['num_entities'],
    )
    if args.rec_model is not None:
        crs_rec_model.load(args.rec_model)
    crs_rec_model = crs_rec_model.to(device)
    models = [crs_rec_model]

    for i, model in enumerate(models):
        models[i] = accelerator.prepare(model)

    # optim
    decay_parameters = []
    for model in models:
        decay_parameters.extend(get_parameter_names(model, [nn.LayerNorm]))
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in models for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in models for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = accelerator.prepare(optimizer)

    # data
    train_data_file = os.path.join('data', args.dataset, 'train_data_processed.jsonl')
    train_data_list = load_jsonl_data(train_data_file)
    train_dataset = DatasetForRec(
        data_list=train_data_list, entity2id=kg['entity2id'], max_length=args.entity_max_length,
        debug=args.debug, shot=args.shot,
    )

    valid_data_file = os.path.join('data', args.dataset, 'valid_data_processed.jsonl')
    valid_data_list = load_jsonl_data(valid_data_file)
    valid_dataset = DatasetForRec(
        data_list=valid_data_list, entity2id=kg['entity2id'], max_length=args.entity_max_length,
        debug=args.debug, shot=args.shot,
    )

    test_data_file = os.path.join('data', args.dataset, 'test_data_processed.jsonl')
    test_data_list = load_jsonl_data(test_data_file)
    test_dataset = DatasetForRec(
        data_list=test_data_list, entity2id=kg['entity2id'], max_length=args.entity_max_length,
        debug=args.debug, shot=args.shot,
    )


    def simple_collate(batch):
        return batch


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=simple_collate,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=simple_collate,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=simple_collate,
    )
    train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, valid_dataloader, test_dataloader
    )

    data_collator = KBRDDataCollatorForRec(
        pad_id=kg['pad_id'], max_length=args.entity_max_length, device=device, debug=args.debug,
    )

    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0

    # lr_scheduler
    lr_scheduler = None
    if args.use_lr_scheduler:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_update_steps_per_epoch, args.max_train_steps)
        lr_scheduler = accelerator.prepare(lr_scheduler)

    # evaluator
    evaluator = RecMetric()

    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'recall@50', 1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        for model in models:
            model.train()

        for step, batch in enumerate(train_dataloader):
            batch = data_collator(batch)
            # with torch.no_grad():
            #     dialog_hidden_state = dialog_encoder(**batch['context']).last_hidden_state[:, -1, :]  # (bs, dialog_hs)
            #     batch['entity']['dialog_hidden_states'] = dialog_hidden_state
            batch['entity']['edge_index'] = edge_index
            batch['entity']['edge_type'] = edge_type
            outputs = crs_rec_model(**batch['entity'], reduction='mean')
            loss = outputs['loss']

            accelerator.backward(loss)
            train_loss.append(float(loss))

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    for model in models:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        del train_loss, batch

        # valid
        valid_loss = []
        for model in models:
            model.eval()

        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                batch = data_collator(batch)
                # dialog_hidden_state = dialog_encoder(**batch['context']).last_hidden_state[:, -1, :]  # (bs, dialog_hs)
                # batch['entity']['dialog_hidden_states'] = dialog_hidden_state
                batch['entity']['edge_index'] = edge_index
                batch['entity']['edge_type'] = edge_type
                outputs = crs_rec_model(**batch['entity'], reduction='mean')

                valid_loss.append(float(outputs['loss']))
                logits = outputs['logit'][:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                preds = [[kg['item_ids'][rank] for rank in rank_list] for rank_list in ranks]
                labels = batch['entity']['labels'].tolist()
                evaluator.evaluate(preds, labels)

        # metric
        report = evaluator.report()
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            crs_rec_model.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test
        test_loss = []
        for model in models:
            model.eval()

        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                batch = data_collator(batch)
                # dialog_hidden_state = dialog_encoder(**batch['context']).last_hidden_state[:, -1, :]  # (bs, dialog_hs)
                # batch['entity']['dialog_hidden_states'] = dialog_hidden_state
                batch['entity']['edge_index'] = edge_index
                batch['entity']['edge_type'] = edge_type
                outputs = crs_rec_model(**batch['entity'], reduction='mean')

                test_loss.append(float(outputs['loss']))
                logits = outputs['logit'][:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                preds = [[kg['item_ids'][rank] for rank in rank_list] for rank_list in ranks]
                labels = batch['entity']['labels'].tolist()
                evaluator.evaluate(preds, labels)

        # metric
        report = evaluator.report()
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    final_dir = os.path.join(args.output_dir, 'final')
    crs_rec_model.save(final_dir)
    logger.info(f'save final model')
