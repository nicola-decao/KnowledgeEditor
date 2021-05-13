import argparse
import logging
import os
import pickle
from copy import deepcopy

import torch
from tqdm.auto import tqdm

from src.data.binary_augmented_kilt import BinaryAugmentedKILT
from src.models.bert_binary_augmented_kilt import BertBinaryAugmented
from src.models.bert_binary_kilt import BertBinary
from src.utils import batch_it, shuffle_it

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="Filename of the model",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where to save files",
    )
    parser.add_argument(
        "method",
        type=str,
        choices=["baseline", "hyper"],
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=["all"] + [str(i) for i in range(12)],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--from",
        type=int,
        default=0,
        dest="from_idx",
    )
    parser.add_argument(
        "--to",
        type=int,
        default=100000,
        dest="to_idx",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.method == "baseline":
        model = BertBinary.load_from_checkpoint(args.model)
        model = model.eval().to(args.device)

        val_dataset0 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
            return_view=True,
            all_views=True,
        )
        val_dataset0 = list(shuffle_it(list(enumerate(val_dataset0))))[
            args.from_idx : args.to_idx
        ]

        val_dataset1 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
        )
        preds = torch.tensor([e["pred"] for e in val_dataset1])

        all_logits = {}
        all_rephrases = {}
        iter_ = tqdm(val_dataset0)
        for j, d0 in iter_:
            tmodel = deepcopy(model)
            optimizer = torch.optim.RMSprop(
                [
                    p
                    for n, p in tmodel.named_parameters()
                    if (
                        (args.layer != "all" and f".{args.layer}." in n)
                        or args.layer == "all"
                    )
                    and all(
                        e not in n.lower()
                        for e in (
                            "bias",
                            "norm",
                            "embeddings",
                            "classifier",
                            "pooler",
                            "shared",
                            "embed",
                            "positions",
                        )
                    )
                ],
                lr=1e-5,
            )

            while True:
                logit = tmodel(
                    {
                        k: v.to(tmodel.device)
                        for k, v in val_dataset1.collate_fn([d0]).items()
                        if isinstance(v, torch.Tensor)
                    }
                )

                if (logit > 0).item() == d0["alt"]:
                    break

                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit,
                    torch.tensor([d0["alt"]], device=tmodel.device).float(),
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            all_rephrases[j] = tmodel.sample(d0["view"]).cpu()

            all_logits_batch = []
            for i, d1 in enumerate(batch_it(val_dataset1, args.batch_size)):
                all_logits_batch.append(tmodel.sample([e["src"] for e in d1]).cpu())

            all_logits[j] = torch.cat(all_logits_batch)

            iter_.set_postfix(
                succ=sum(
                    val_dataset1[k]["alt"] == (v[k] > 0).item()
                    for k, v in all_logits.items()
                )
                / len(all_logits),
                retain=sum(
                    (
                        ((v[:k] > 0) == preds[:k]).sum()
                        + ((v[k + 1 :] > 0) == preds[k + 1 :]).sum()
                    )
                    / (len(v) - 1)
                    for k, v in all_logits.items()
                ).item()
                / len(all_logits),
                equiv=sum(
                    (v.sign() == all_logits[k][k].sign()).float().mean().item()
                    for k, v in all_rephrases.items()
                )
                / len(all_rephrases),
            )

        filename = os.path.join(
            args.output_path,
            f"all_logits-{args.from_idx}-{args.to_idx}-baseline-{args.layer}.pkl",
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_logits, f)

        filename = os.path.join(
            args.output_path,
            f"all_rephrases-{args.from_idx}-{args.to_idx}-baseline-{args.layer}.pkl",
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_rephrases, f)

    elif args.method == "hyper":
        model = BertBinaryAugmented.load_from_checkpoint(args.model)
        model.model = BertBinary.load_from_checkpoint(
            model.hparams.model_checkpoint
        ).model
        model = model.eval().to(args.device)

        val_dataset0 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
            return_view=True,
            all_views=True,
        )
        val_dataset0 = list(shuffle_it(list(enumerate(val_dataset0))))[
            args.from_idx : args.to_idx
        ]

        val_dataset1 = BinaryAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
        )
        preds = torch.tensor([e["pred"] for e in val_dataset1])

        model.val_dataloader(shuffle=False)

        all_logits = {}
        all_rephrases = {}

        iter_ = tqdm(val_dataset0)
        for j, d0 in iter_:

            with torch.no_grad():
                logits_orig, params_dict = model.get_logits_orig_params_dict(
                    {
                        k: v.to(model.device)
                        for k, v in model.val_dataset.get_batch([], d0["cond"]).items()
                    }
                )

            _, logits, params_dict = model.sample(
                d0["view"],
                d0["cond"],
                logits_orig,
                params_dict,
                stop_condition=lambda condition, logits, n_iter: (
                    not("REFUTES >> SUPPORTS" in condition and logits[-1] > 0)
                ) and n_iter < 5,
            )
            
            all_rephrases[j] = logits.cpu()

            all_logits_batch = []
            for i, d1 in enumerate(batch_it(val_dataset1, args.batch_size)):
                _, logits, _ = model.sample(
                    [e["src"] for e in d1], d0["cond"], logits_orig, params_dict
                )
                all_logits_batch.append(logits.cpu())

            all_logits[j] = torch.cat(all_logits_batch)

            iter_.set_postfix(
                succ=sum(
                    val_dataset1[k]["alt"] == (v[k] > 0).item()
                    for k, v in all_logits.items()
                )
                / len(all_logits),
                retain=sum(
                    (
                        ((v[:k] > 0) == preds[:k]).sum()
                        + ((v[k + 1 :] > 0) == preds[k + 1 :]).sum()
                    )
                    / (len(v) - 1)
                    for k, v in all_logits.items()
                ).item()
                / len(all_logits),
                equiv=sum(
                    (v.sign() == all_logits[k][k].sign()).float().mean().item()
                    for k, v in all_rephrases.items()
                )
                / len(all_rephrases),
            )

        filename = os.path.join(
            args.output_path, f"all_logits-{args.from_idx}-{args.to_idx}.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_logits, f)

        filename = os.path.join(
            args.output_path, f"all_rephrases-{args.from_idx}-{args.to_idx}.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_rephrases, f)
