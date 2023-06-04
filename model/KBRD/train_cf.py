import argparse
import json
import os
import sys
import time

import copy
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

from dataloader_kbrd import KBRDDataCollatorForRec
from dataset import DatasetForFlow, DatasetForRec
from kg_cf import KG
from kg_kbrd import KGForKBRD
from metric import RecMetric
from dataloader_cf import UserDataCollator, MetaPathDataCollator, FlowDataCollator, TemplateDataCollator
from model_cf import BartForFlowGeneration, MetaPathPredictor, TokenEmbedding, UserEncoder
from model_kbrd import KBRDforRec
from utils import load_jsonl_data, simple_collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--num_epochs", type=int, required=True)
    # data augmentation
    parser.add_argument("--flow_dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--aug_num", type=int, default=1)
    parser.add_argument("--aug_prob", type=str)
    parser.add_argument("--prob_as_weight", action='store_true')
    # cf
    parser.add_argument("--kg_dataset", type=str)
    parser.add_argument("--split", type=str, )
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--max_length", type=int, )
    parser.add_argument("--pretrain_dir", type=str, required=True)
    parser.add_argument("--meta_path_model", type=str, required=True)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--batch_size", type=int,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--num_beams", type=int, default=1)
    # delta
    parser.add_argument("--delta_iterations", type=int)
    parser.add_argument("--delta_lr", type=float)
    parser.add_argument("--delta_l2", type=float)
    parser.add_argument("--delta_l2_ratio", type=float)
    # crs
    parser.add_argument("--crs_dataset", type=str, required=True)
    parser.add_argument("--crs_max_length", type=int, required=True)
    parser.add_argument("--crs_kg_dataset", type=str, required=True)
    parser.add_argument("--crs_model", type=str, required=True)
    parser.add_argument("--crs_hidden_size", type=int, required=True)
    parser.add_argument("--crs_batch_size", type=int, required=True)
    parser.add_argument("--crs_iterations", type=int, required=True)
    parser.add_argument("--crs_lr", type=float, required=True)
    parser.add_argument("--crs_l2", type=float, required=True)
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")

    args = parser.parse_args()
    return args


@torch.no_grad()
def predict_meta_paths(user_embeds, meta_path_mask):
    meta_path_inputs = {'user_embeds': user_embeds}
    meta_path_outputs = meta_path_predictor(**meta_path_inputs)

    meta_path_indices = torch.arange(kg['num_meta_paths'])
    meta_path_preds = []
    for logit, meta_path_mask in zip(meta_path_outputs['logits'], meta_path_mask):
        masked_logit = logit[meta_path_mask]
        rank = torch.argmax(masked_logit).item()
        meta_path_pred = meta_path_indices[meta_path_mask][rank].item()
        meta_path_preds.append(kg['id2metapathid'][meta_path_pred])

    return meta_path_preds


@torch.no_grad()
def generate_flows(batch, token_embeds, user_embeds, meta_path_preds, num_beams):
    flow_batch = flow_data_collator(batch, meta_path_preds)
    flow_inputs = {**flow_batch['inputs'], 'token_embeds': token_embeds, 'user_embeds': user_embeds}

    gen_args['max_new_tokens'] = flow_batch['inputs']['decoder_logits_mask'].shape[1]
    if args.debug:
        gen_args['min_length'] = gen_args['max_new_tokens'] + 1
    gen_args['num_return_sequences'] = gen_args['num_beams'] = num_beams
    input_len = 1

    gen_seqs = flow_model.generate(**flow_inputs, **gen_args)[:, input_len:]
    return gen_seqs


def compute_log_prob(batch, meta_path_preds, token_embeds, user_embeds, gen_seqs, num_beams):
    flow_batch = flow_data_collator(batch, meta_path_preds)['inputs']
    labels = gen_seqs.detach().clone()
    labels[labels == kg['pad_id']] = -100
    flow_inputs = {**flow_batch, 'labels': labels, 'token_embeds': token_embeds, 'user_embeds': user_embeds}

    input_ids, model_kwargs = flow_model._expand_inputs_for_generation(
        **flow_inputs, expand_size=num_beams, is_encoder_decoder=False
    )

    neg_log_prob = flow_model(input_ids, **model_kwargs, reduction='none').loss.sum(dim=-1)  # (bs)
    return neg_log_prob


@torch.no_grad()
def data_generation(aug_dataset, token_embeds):
    aug_data_list = [data for data in aug_dataset]

    # user
    user_batch = user_data_collator(aug_data_list)
    user_ids = user_batch['user_id']

    user_inputs = {**user_batch['entity'], 'token_embeds': token_embeds}
    user_embeds = user_encoder(**user_inputs)
    user_embeds = user_embeds.reshape(len(user_embeds) // 2, 2, -1)

    # meta path prediction
    meta_path_batch = meta_path_data_collator(aug_data_list)
    meta_path_preds = predict_meta_paths(
        user_embeds=user_embeds, meta_path_mask=meta_path_batch['meta_path_mask']
    )

    # flow generation
    gen_seqs = generate_flows(
        batch=aug_data_list, token_embeds=token_embeds, user_embeds=user_embeds, meta_path_preds=meta_path_preds,
        num_beams=args.aug_num
    )
    assert len(gen_seqs) == len(aug_data_list) * args.aug_num

    gen_seqs = gen_seqs.reshape(len(aug_data_list), args.aug_num, -1)

    # # filter flow
    # neg_log_probs = compute_log_prob(
    #     batch=aug_data_list, meta_path_preds=meta_path_preds,
    #     token_embeds=token_embeds, user_embeds=user_embeds, gen_seqs=gen_seqs, num_beams=1
    # )
    # probs = torch.exp(-neg_log_probs)
    #
    # prob_t = 0
    # if args.aug_prob is not None:
    #     if args.aug_prob == 'median':
    #         prob_t = torch.median(probs)
    #     else:
    #         prob_t = float(args.aug_prob)

    flow_data_list = []
    template_batch = template_data_collator(aug_data_list)
    for gen_seq_list, template_text_list, template_pos_list in zip(
            gen_seqs.tolist(), template_batch['text'], template_batch['position']
    ):
        # if prob < prob_t:
        #     continue

        for gen_seq in gen_seq_list:
            flow = []
            for idx in gen_seq:
                if idx == kg['user_id']:
                    continue
                elif idx == kg['bot_id']:
                    continue
                else:
                    ent = kg['id2entity'][idx]
                    if ent not in decode_bad_tokens:
                        flow.append(ent)

            # flow_data_list.append({
            #     'entity': flow[:-1],
            #     'item': [flow[-1]]
            # })

            dialog = []
            flow_idx, template_idx = 0, 0
            last_ent_turn = None

            for turn, template in enumerate(template_text_list):
                cur_sent = template
                cur_ent_list = []

                while '<mask>' in cur_sent and flow_idx < len(flow):
                    cur_ent = flow[flow_idx]
                    cur_ent_list.append(cur_ent)
                    cur_sent = cur_sent.replace('<mask>', cur_ent, 1)
                    flow_idx += 1
                    last_ent_turn = turn

                dialog.append({
                    'text': cur_sent,
                    'entity': cur_ent_list
                })

            context_list = []
            entity_list = []
            for turn in dialog:
                if len(turn['entity']) > 0 and len(entity_list) > 0:
                    for ent in turn['entity']:
                        flow_data = {
                            'epoch': epoch,
                            'context': copy.copy(context_list),
                            'entity': copy.copy(entity_list),
                            'rec': [ent],
                            'resp': turn['text']
                        }
                        flow_data_list.append(flow_data)

                context_list.append(turn['text'])
                entity_list.extend(turn['entity'])

    return flow_data_list


@torch.no_grad()
def compute_reward(flow_data_list, node_embeds):
    dataset_rec = DatasetForRec(flow_data_list, entity2id=kg_crs['entity2id'], max_length=args.crs_max_length)
    dataloader_rec = DataLoader(dataset_rec, batch_size=args.crs_batch_size, collate_fn=simple_collate)

    crs_loss_list = []
    for batch_crs in dataloader_rec:
        batch_crs = crs_data_collator(batch_crs)
        with torch.no_grad():
            loss = crs_model(node_embeds=node_embeds, **batch_crs['entity'])['loss']
        crs_loss_list.append(loss)

    reward = torch.cat(crs_loss_list, dim=0)
    return reward


def data_augmentation(base_dataset):
    for model in generation_models:
        model.eval()
    crs_model.eval()

    with torch.no_grad():
        token_embeds = token_embedding(edge_index, edge_type)
        node_embeds = crs_model.get_node_embeds(edge_index_crs, edge_type_crs)

    flow_dataloader = DataLoader(base_dataset, batch_size=args.batch_size, collate_fn=simple_collate)

    aug_flow_data_list = []
    aug_user_dataset_list = []

    for batch in tqdm(flow_dataloader):
        # user embedding
        user_batch = user_data_collator(batch)
        user_ids = user_batch['user_id']
        user_entity_ids = user_batch['entity']['user_entity_ids']
        user_entity_mask = user_batch['entity']['user_entity_mask']
        user_entity_embeds = token_embeds[user_entity_ids]  # (bs * 2, ent_len, hs)
        # meta path
        meta_path_batch = meta_path_data_collator(batch)

        new_user_entity_idx = torch.randint(high=user_entity_embeds.shape[1], size=(len(user_entity_embeds),))
        user_entity_embeds_delta = torch.normal(
            mean=0, std=user_entity_embeds[user_entity_mask].std(),
            size=(len(user_entity_embeds), user_entity_embeds.shape[-1]), device=device, requires_grad=True
        )

        if args.delta_l2_ratio is not None:
            weight_decay = args.delta_l2 * args.delta_l2_ratio ** epoch
        else:
            weight_decay = args.delta_l2
        delta_optimizer = torch.optim.SGD([user_entity_embeds_delta], lr=args.delta_lr, weight_decay=weight_decay)
        delta_optimizer = accelerator.prepare(delta_optimizer)

        # counterfactual learning for delta
        for it in tqdm(range(args.delta_iterations)):
            # user embeds
            new_user_entity_embeds = user_entity_embeds.detach().clone()
            new_user_entity_embeds[range(len(user_entity_embeds)), new_user_entity_idx] += user_entity_embeds_delta
            user_inputs = {
                'user_entity_embeds': new_user_entity_embeds,
                # 'user_entity_mask': user_entity_mask
            }

            new_user_embeds = user_encoder(**user_inputs)
            new_user_embeds = new_user_embeds.reshape(len(new_user_embeds) // 2, 2, -1)

            # meta path prediction
            meta_path_preds = predict_meta_paths(
                user_embeds=new_user_embeds, meta_path_mask=meta_path_batch['meta_path_mask']
            )

            # flow generation
            gen_seqs = generate_flows(
                batch=batch, token_embeds=token_embeds, user_embeds=new_user_embeds, meta_path_preds=meta_path_preds,
                num_beams=num_beams
            )

            flow_data_list = []
            template_batch = template_data_collator(batch)
            flow_preds = gen_seqs.reshape(len(batch), num_beams, -1)
            for user_list, preds, template_text_list, template_pos_list in zip(
                    user_ids, flow_preds.tolist(), template_batch['text'], template_batch['position']
            ):
                user, bot = user_list

                for pred in preds:
                    decoded_pred = []
                    flow = []
                    for idx in pred:
                        if idx == kg['user_id']:
                            decoded_pred.append(user)
                        elif idx == kg['bot_id']:
                            decoded_pred.append(bot)
                        else:
                            ent = kg['id2entity'][idx]
                            if ent not in decode_bad_tokens:
                                decoded_pred.append(ent)
                                flow.append(ent)

                    dialog = []
                    flow_idx, template_idx = 0, 0
                    last_ent_turn = None

                    for turn, template in enumerate(template_text_list):
                        cur_sent = template
                        cur_ent_list = []

                        while '<mask>' in cur_sent and flow_idx < len(flow):
                            cur_ent = flow[flow_idx]
                            cur_ent_list.append(cur_ent)
                            cur_sent = cur_sent.replace('<mask>', cur_ent, 1)
                            flow_idx += 1

                        if len(cur_ent_list) > 0:
                            last_ent_turn = turn

                        dialog.append({
                            'text': cur_sent,
                            'entity': cur_ent_list
                        })

                    context_list = []
                    entity_list = []
                    for turn in dialog[:last_ent_turn]:
                        context_list.append(turn['text'])
                        entity_list.extend(turn['entity'])

                    rec = dialog[last_ent_turn]['entity'][-1]
                    resp = dialog[last_ent_turn]['text']

                    flow_data = {
                        'epoch': epoch,
                        'iter': it,
                        'context': context_list,
                        'entity': entity_list,
                        'rec': [rec],
                        'resp': resp
                    }
                    flow_data_list.append(flow_data)

                    if log_level == 'DEBUG':
                        log_file.write(json.dumps(flow_data, ensure_ascii=False) + '\n')

            # log_prob of each beam
            neg_log_prob = compute_log_prob(
                batch=batch, meta_path_preds=meta_path_preds,
                token_embeds=token_embeds, user_embeds=new_user_embeds, gen_seqs=gen_seqs, num_beams=num_beams
            )

            # reward for each beam
            reward = compute_reward(flow_data_list, node_embeds)
            logger.debug(f'{it}: {torch.mean(reward)}')

            loss = (neg_log_prob * reward).mean()
            accelerator.backward(loss)

            delta_optimizer.step()
            delta_optimizer.zero_grad()

        accelerator._optimizers.pop()

        # new user
        new_user_entity_embeds = user_entity_embeds.detach().clone()
        new_user_entity_embeds[range(len(user_entity_embeds)), new_user_entity_idx] += user_entity_embeds_delta
        batch_size, ent_len, hidden_size = new_user_entity_embeds.shape

        dist_mat = torch.cdist(new_user_entity_embeds.reshape(1, -1, hidden_size), token_embeds.unsqueeze(0))[0]
        new_user_entity_ids = torch.argmin(dist_mat, dim=-1).reshape(batch_size // 2, 2, ent_len)  # (bs, 2, ent_len)

        templates = [data['template'] for data in batch]

        cur_aug_user_data_list = []
        for cur_user_ids, cur_new_user_entity_ids, template_list in zip(user_ids, new_user_entity_ids, templates):
            new_user_ids = [f'{cur_user_id.split("-")[0]}-{epoch}' for cur_user_id in cur_user_ids]
            user2entity = {}
            for new_user_id, new_user_entity_id_list in zip(new_user_ids, cur_new_user_entity_ids):
                user2entity[new_user_id] = [
                    kg['id2entity'][ent_id] for ent_id in
                    new_user_entity_id_list[new_user_entity_id_list != kg['pad_id']].tolist()
                ]
            cur_aug_user_data_list.append({
                'user': new_user_ids,
                'user2entity': user2entity,
                'template': template_list
            })
        cur_aug_user_dataset = DatasetForFlow(
            cur_aug_user_data_list, kg=kg, max_length=args.max_length,
        )
        aug_user_dataset_list.append(cur_aug_user_dataset)

        # new flow
        cur_aug_flow_data_list = data_generation(aug_dataset=cur_aug_user_dataset, token_embeds=token_embeds)
        aug_flow_data_list.extend(cur_aug_flow_data_list)

    # deduplicate
    aug_flow_data_list_filter = []
    flow_set = set()
    for flow_data in aug_flow_data_list:
        flow_tup = tuple(flow_data['entity'] + flow_data['rec'])
        if flow_tup not in flow_set:
            flow_set.add(flow_tup)
            aug_flow_data_list_filter.append(flow_data)
            log_file.write(json.dumps(flow_data, ensure_ascii=False) + '\n')

    return aug_flow_data_list, aug_user_dataset_list


@torch.no_grad()
def crs_model_testing(dataloader):
    crs_model.eval()

    valid_loss_list = []
    for batch in tqdm(dataloader):
        batch = crs_data_collator(batch)
        batch['entity']['edge_index'] = edge_index
        batch['entity']['edge_type'] = edge_type
        outputs = crs_model(**batch['entity'], reduction='mean')

        valid_loss_list.append(float(outputs['loss']))
        logits = outputs['logit'][:, item_ids_crs]
        ranks = torch.topk(logits, k=50, dim=-1).indices
        preds = item_ids_crs[ranks].tolist()
        labels = batch['entity']['labels'].tolist()
        evaluator.evaluate(preds, labels)

    report = evaluator.report()
    report['loss'] = np.mean(valid_loss_list)

    return report


def crs_model_learning(training_dataset_for_crs, best_metric, epoch):
    crs_optimizer = torch.optim.AdamW(crs_model.parameters(), lr=args.crs_lr, weight_decay=args.crs_l2)
    crs_optimizer = accelerator.prepare(crs_optimizer)

    dataloader_rec = DataLoader(
        training_dataset_for_crs, batch_size=args.crs_batch_size, collate_fn=simple_collate, shuffle=True
    )
    for it in range(args.crs_iterations):
        crs_model.train()

        train_loss_list = []
        for batch_crs in tqdm(dataloader_rec):
            batch_crs = crs_data_collator(batch_crs)
            crs_inputs = {**batch_crs['entity'], 'edge_index': edge_index, 'edge_type': edge_type}

            loss = crs_model(**crs_inputs, reduction='none')['loss']
            loss = torch.mean(loss * batch_crs['weight'])
            accelerator.backward(loss)
            train_loss_list.append(float(loss))

            crs_optimizer.step()
            crs_optimizer.zero_grad()
            if run:
                run.log({'loss': np.mean(train_loss_list)})

        train_loss = np.mean(train_loss_list)
        logger.info(f'epoch {it} train loss {train_loss}')

        # testing crs model performance
        # validation set
        report = crs_model_testing(crs_valid_dataloader)
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v

        valid_report['epoch'] = epoch + it
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        # save model with the best performance
        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            crs_model.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test set
        report = crs_model_testing(crs_test_dataloader)
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v

        test_report['epoch'] = epoch + it
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    accelerator._optimizers.pop()

    return best_metric


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
    log_level = 'DEBUG'
    if args.name is not None:
        log_level = 'INFO'
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add(f'log/{local_time}.log', level='DEBUG')

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
    kg = KG(kg_dataset=args.kg_dataset, debug=args.debug).get_kg_info()
    edge_index = torch.as_tensor(kg['edge_index'], device=device)
    edge_type = torch.as_tensor(kg['edge_type'], device=device)
    id2entity = kg['id2entity']

    kg_crs = KGForKBRD(kg_dataset=args.crs_kg_dataset, debug=args.debug).get_kg_info()
    edge_index_crs = torch.as_tensor(kg_crs['edge_index'], device=device)
    edge_type_crs = torch.as_tensor(kg_crs['edge_type'], device=device)
    item_ids_crs = torch.as_tensor(kg_crs['item_ids'])

    # model
    # generation
    flow_model = BartForFlowGeneration.from_pretrained(f'{args.pretrain_dir}/flow').to(device)
    config = flow_model.config

    token_embedding = TokenEmbedding(
        hidden_size=config.hidden_size, num_bases=args.num_bases,
        num_relations=kg['num_relations'], num_entities=kg['num_entities'], num_special_tokens=kg['num_special_tokens']
    )
    token_embedding.load(f'{args.pretrain_dir}/token')
    token_embedding = token_embedding.to(device)

    user_encoder = UserEncoder(hidden_size=config.hidden_size)
    user_encoder.load(f'{args.pretrain_dir}/user')
    user_encoder.to(device)

    meta_path_predictor = MetaPathPredictor(hidden_size=config.hidden_size, num_labels=kg['num_meta_paths']).to(device)
    meta_path_predictor.load(args.meta_path_model)
    meta_path_predictor = meta_path_predictor.to(device)

    generation_models = [token_embedding, user_encoder, meta_path_predictor, flow_model]
    for i, model in enumerate(generation_models):
        model.requires_grad_(False)
        generation_models[i] = accelerator.prepare(model)

    # crs
    crs_model = KBRDforRec(
        args.crs_hidden_size, num_bases=args.num_bases,
        num_relations=kg_crs['num_relations'], num_entities=kg_crs['num_entities']
    )
    crs_model.load(args.crs_model)
    crs_model = crs_model.to(device)
    crs_model = accelerator.prepare(crs_model)

    # data
    # flow
    data_file = os.path.join('data', args.flow_dataset, f'{args.split}_data_processed.jsonl')
    data_list = load_jsonl_data(data_file)
    dataset = DatasetForFlow(
        data_list=data_list, kg=kg,
        max_length=args.max_length,
        debug=args.debug, shot=args.shot,
    )

    user_data_collator = UserDataCollator(
        kg=kg, max_length=args.max_length, device=device, debug=args.debug
    )
    meta_path_data_collator = MetaPathDataCollator()
    flow_data_collator = FlowDataCollator(
        kg=kg, max_length=args.max_length,
        device=device, debug=args.debug
    )
    template_data_collator = TemplateDataCollator()

    # crs
    crs_train_data_file = os.path.join('data', args.crs_dataset, 'train_data_processed.jsonl')
    crs_train_data_list = load_jsonl_data(crs_train_data_file)
    crs_train_dataset = DatasetForRec(
        crs_train_data_list, entity2id=kg_crs['entity2id'], max_length=args.crs_max_length,
        debug=args.debug,
        # shot=args.shot,
    )

    crs_valid_data_file = os.path.join('data', args.crs_dataset, 'valid_data_processed.jsonl')
    crs_valid_data_list = load_jsonl_data(crs_valid_data_file)
    crs_valid_dataset = DatasetForRec(
        crs_valid_data_list, entity2id=kg_crs['entity2id'], max_length=args.crs_max_length,
        debug=args.debug,
        # shot=args.shot,
    )
    crs_valid_dataloader = DataLoader(crs_valid_dataset, batch_size=args.crs_batch_size, collate_fn=simple_collate)

    crs_test_data_file = os.path.join('data', args.crs_dataset, 'test_data_processed.jsonl')
    crs_test_data_list = load_jsonl_data(crs_test_data_file)
    crs_test_dataset = DatasetForRec(
        crs_test_data_list, entity2id=kg_crs['entity2id'], max_length=args.crs_max_length,
        debug=args.debug,
        # shot=args.shot,
    )
    crs_test_dataloader = DataLoader(crs_test_dataset, batch_size=args.crs_batch_size, collate_fn=simple_collate)

    crs_data_collator = KBRDDataCollatorForRec(pad_id=kg_crs['pad_id'], max_length=args.crs_max_length, device=device,
                                               debug=args.debug)
    # evaluation
    evaluator = RecMetric()

    # params for generation
    gen_args = {
        'no_repeat_ngram_size': 2,
        # 'do_sample': True,
        # 'top_k': 3,
    }
    num_beams = args.num_beams
    decode_bad_tokens = {'<pad>', '<eos>', '<sep>', '<bos>', 'genre', 'movie', 'person'}

    # log
    if args.name is not None:
        log_file_dir = 'save'
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, f'{args.name}.jsonl')
        log_file = open(log_file_path, 'w', encoding='utf-8')
    else:
        log_file_dir = 'log'
        os.makedirs(log_file_dir, exist_ok=True)
        log_file_path = os.path.join(log_file_dir, f'gen_{local_time}.jsonl')
        log_file = open(log_file_path, 'w', encoding='utf-8')

    # save model with best metric
    metric, mode = 'recall@50', 1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')

    base_dataset = dataset
    aug_user_dataset_list = None

    for epoch in tqdm(range(args.num_epochs)):
        if aug_user_dataset_list is not None:
            base_dataset = ConcatDataset(aug_user_dataset_list)

        aug_flow_data_list, aug_user_dataset_list = data_augmentation(base_dataset)

        aug_train_dataset = DatasetForRec(aug_flow_data_list, entity2id=kg_crs['entity2id'], max_length=args.crs_max_length)

        best_metric = crs_model_learning(aug_train_dataset, best_metric, epoch * args.crs_iterations)

        # cur_train_dataset = ConcatDataset([crs_train_dataset, aug_train_dataset])
        # best_metric = crs_model_learning(cur_train_dataset, best_metric, epoch * args.crs_iterations)

        # best_metric = crs_model_learning(aug_train_dataset, best_metric, 2 * epoch * args.crs_iterations)
        # best_metric = crs_model_learning(crs_train_dataset, best_metric, (epoch * 2 + 1) * args.crs_iterations)

    # save the final model
    final_dir = os.path.join(args.output_dir, 'final')
    crs_model.save(final_dir)
    logger.info(f'save final model')