import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, BartConfig
from transformers.trainer_pt_utils import get_parameter_names

from dataset import FlowDataset, FlowDataCollator, UserDataCollator
from dataset_kg import KG
from metric import ConvMetric
from model import BartForFlowGeneration, TokenEmbedding, UserEncoder
from utils import simple_collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--no_log", action='store_true')
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--other_dataset", type=str)
    parser.add_argument("--other_shot", type=float, default=1)
    parser.add_argument("--kg_dataset", type=str)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--user_entity_max_length", type=int)
    # model
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--decoder_layers", type=int, default=12)
    parser.add_argument("--use_meta_path", action='store_true')
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int)
    parser.add_argument("--fp16", action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")

    args = parser.parse_args()
    return args


@torch.no_grad()
def test(dataloader, stage):
    for model in models:
        model.eval()
    token_embeds = token_embedding(edge_index, edge_type)  # (n_tok, hs)

    flow_loss_list = []
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        # user embedding
        user_batch = user_data_collator(batch)
        user_inputs = {**user_batch['entity'], 'token_embeds': token_embeds}

        user_embeds = user_encoder(**user_inputs)
        user_embeds = user_embeds.reshape(len(user_embeds) // 2, 2, -1)

        # flow generation
        flow_batch = flow_data_collator(batch)
        flow_inputs = {**flow_batch['inputs'], 'token_embeds': token_embeds, 'user_embeds': user_embeds}

        flow_loss = flow_model(**flow_inputs).loss
        flow_loss_list.append(float(flow_loss))

    if args.no_log is False:
        flow_evaluator.log_write_line(f'\n*** {stage}-{epoch} ***\n')
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        # user embedding
        user_batch = user_data_collator(batch)
        user_inputs = {**user_batch['entity'], 'token_embeds': token_embeds}

        user_embeds = user_encoder(**user_inputs)
        user_embeds = user_embeds.reshape(len(user_embeds) // 2, 2, -1)

        # flow generation
        flow_batch = flow_data_collator(batch, gen=True)
        flow_inputs = {**flow_batch['inputs'], 'token_embeds': token_embeds, 'user_embeds': user_embeds}

        if 'decoder_logits_mask' in flow_batch['inputs']:
            gen_args['max_new_tokens'] = flow_batch['inputs']['decoder_logits_mask'].shape[1]
        else:
            gen_args['max_new_tokens'] = args.max_length
        # if args.debug:
        #     gen_args['min_length'] = gen_args['max_new_tokens'] + 1
        input_len = 1

        gen_seqs = flow_model.generate(**flow_inputs, **gen_args)
        flow_preds = gen_seqs[:, input_len:].tolist()

        flow_evaluator.evaluate(
            preds=flow_preds, labels=flow_batch['labels'],
            user_ids=user_batch['user_id'], meta_paths=flow_batch['meta_path'],
            log=accelerator.is_local_main_process
        )

    # metric
    flow_report = flow_evaluator.report()
    flow_report['flow_loss'] = np.mean(flow_loss_list)

    return flow_report


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

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb
    run = None
    if args.use_wandb:
        name = args.name if args.name else local_time
        run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)

    # kg
    kg = KG(dataset=args.kg_dataset, debug=args.debug).get_kg_info()

    entity2id = kg['entity2id']
    id2entity = kg['id2entity']
    num_entities, num_relations = kg['num_entities'], kg['num_relations']
    edge_index = torch.as_tensor(kg['edge_index'], device=device)
    edge_type = torch.as_tensor(kg['edge_type'], device=device)

    entityid2typeid = kg['entityid2typeid']
    id2metapath = kg['id2metapath']
    id2metapathid = kg['id2metapathid']
    meta_path_num_labels = len(id2metapathid)

    # model
    config = BartConfig.from_pretrained(
        "facebook/bart-base", encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        vocab_size=len(entity2id),
        pad_token_id=entity2id['<pad>'], bos_token_id=entity2id['<bos>'], eos_token_id=entity2id['<eos>'],
        forced_bos_token_id=None, forced_eos_token_id=None, decoder_start_token_id=entity2id['<bos>']
    )
    num_special_tokens = len(entity2id) - num_entities

    token_embedding = TokenEmbedding(
        hidden_size=config.hidden_size, num_bases=args.num_bases,
        num_relations=num_relations, num_entities=num_entities, num_special_tokens=num_special_tokens
    ).to(device)
    user_encoder = UserEncoder(hidden_size=config.hidden_size).to(device)

    flow_model = BartForFlowGeneration(config=config).to(device)

    models = [token_embedding, user_encoder, flow_model]
    for i, model in enumerate(models):
        models[i] = accelerator.prepare(model)

    # data
    train_dataset = FlowDataset(
        dataset=args.dataset, split='train', compute_loss_for_meta=False,
        entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
        max_length=args.max_length, user_entity_max_length=args.user_entity_max_length,
        debug=args.debug, shot=args.shot,
    )
    if args.other_dataset is not None:
        other_dataset = FlowDataset(
            dataset=args.other_dataset, split='train', compute_loss_for_meta=False,
            entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
            max_length=args.max_length, user_entity_max_length=args.user_entity_max_length,
            shot=args.shot, debug=args.debug,
        )
        train_dataset = ConcatDataset([train_dataset, other_dataset])

    valid_dataset = FlowDataset(
        dataset=args.dataset, split='valid', debug=args.debug, compute_loss_for_meta=False,
        entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
        # max_length=args.max_length, user_entity_max_length=args.user_entity_max_length
    )
    test_dataset = FlowDataset(
        dataset=args.dataset, split='test', debug=args.debug, compute_loss_for_meta=False,
        entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
        # max_length=args.max_length, user_entity_max_length=args.user_entity_max_length
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=simple_collate,
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

    user_data_collator = UserDataCollator(
        entity2id=entity2id, max_length=args.max_length, device=device, debug=args.debug
    )
    flow_data_collator = FlowDataCollator(
        entity2id=entity2id, entityid2typeid=entityid2typeid, max_length=args.max_length,
        device=device, debug=args.debug, use_meta_path=args.use_meta_path
    )

    # optim
    decay_parameters = []
    for model in models:
        decay_parameters.extend(get_parameter_names(model, [nn.LayerNorm]))
    decay_parameters = {name for name in decay_parameters if "bias" not in name}
    optimizer_grouped_parameters = [
        {
            "params": list(set([p for model in models for n, p in model.named_parameters() if n in decay_parameters])),
            "weight_decay": args.weight_decay,
        },
        {
            "params": list(
                set([p for model in models for n, p in model.named_parameters() if n not in decay_parameters])),
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = accelerator.prepare(optimizer)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_update_steps_per_epoch, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # params for generation
    gen_args = {
        # 'min_length': 6,
        # 'max_length': args.max_length,
        'num_beams': 1,
        'no_repeat_ngram_size': 2,
    }

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

    # eval
    log_file_path = None
    if args.no_log is False:
        log_file_dir = 'log'
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, f'gen_{local_time}.jsonl')

    flow_evaluator = ConvMetric(
        entity2id=entity2id, id2entity=id2entity, log_file_path=log_file_path
    )

    # save model with min loss
    best_metric_val = 0
    best_dir = os.path.join(args.output_dir, 'best')

    token_embedding_best_dir = os.path.join(best_dir, 'token')
    os.makedirs(token_embedding_best_dir, exist_ok=True)

    user_encoder_best_dir = os.path.join(best_dir, 'user')
    os.makedirs(user_encoder_best_dir, exist_ok=True)

    flow_best_dir = os.path.join(best_dir, 'flow')
    os.makedirs(flow_best_dir, exist_ok=True)

    # train loop
    for epoch in range(args.num_train_epochs):
        flow_train_loss = []
        for model in models:
            model.train()

        for step, batch in enumerate(train_dataloader):
            # token embedding
            token_embeds = token_embedding(edge_index, edge_type)  # (n_tok, hs)

            # user embedding
            user_batch = user_data_collator(batch)
            user_inputs = {**user_batch['entity'], 'token_embeds': token_embeds}

            user_embeds = user_encoder(**user_inputs)
            user_embeds = user_embeds.reshape(len(user_embeds) // 2, 2, -1)

            # flow generation
            flow_batch = flow_data_collator(batch)
            flow_inputs = {**flow_batch['inputs'], 'token_embeds': token_embeds, 'user_embeds': user_embeds}

            flow_loss = flow_model(**flow_inputs).loss / args.gradient_accumulation_steps
            flow_train_loss.append(float(flow_loss))

            loss = flow_loss
            accelerator.backward(loss)

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(flow_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({
                        'train/flow_loss': np.mean(flow_train_loss) * args.gradient_accumulation_steps
                    })

            if completed_steps >= args.max_train_steps:
                break

        # metric
        flow_train_loss_avg = np.mean(flow_train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} flow train loss {flow_train_loss_avg}')

        # validation
        flow_report = test(valid_dataloader, 'valid')
        valid_report = {'epoch': epoch}
        for report in [flow_report]:
            for k, v in report.items():
                valid_report[f'valid/{k}'] = v

        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        flow_evaluator.reset_metric()

        # save model
        cur_metric_val = flow_report['bleu@4']
        if cur_metric_val > best_metric_val:
            best_metric_val = cur_metric_val
            token_embedding.save(token_embedding_best_dir)
            user_encoder.save(user_encoder_best_dir)
            flow_model.save_pretrained(flow_best_dir)

        # test
        flow_report = test(test_dataloader, 'test')
        test_report = {'epoch': epoch}
        for report in [flow_report]:
            for k, v in report.items():
                test_report[f'test/{k}'] = v

        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        flow_evaluator.reset_metric()

    # save model
    final_dir = os.path.join(args.output_dir, 'final')

    token_final_dir = os.path.join(final_dir, 'token')
    os.makedirs(token_final_dir, exist_ok=True)
    token_embedding.save(token_final_dir)

    user_final_dir = os.path.join(final_dir, 'user')
    os.makedirs(user_final_dir, exist_ok=True)
    user_encoder.save(user_final_dir)

    flow_final_dir = os.path.join(final_dir, 'flow')
    os.makedirs(flow_final_dir, exist_ok=True)
    flow_model.save_pretrained(flow_final_dir, save_function=accelerator.save)
