import argparse
import logging
import os
import pickle
from copy import deepcopy

import torch
from tqdm.auto import tqdm

from src.data.seq2seq_augmented_kilt import Seq2SeqAugmentedKILT
from src.models.bart_seq2seq_augmented_kilt import BartSeq2SeqAugmented
from src.models.bart_seq2seq_kilt import BartSeq2Seq
from src.utils import batch_it, shuffle_it, normalize, label_smoothed_nll_loss

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

    args, _ = parser.parse_known_args()

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.method == "baseline":

        model = BartSeq2Seq.load_from_checkpoint(args.model).to(args.device)

        val_dataset0 = Seq2SeqAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
            return_view=True,
            all_views=True,
        )
        val_dataset0 = list(shuffle_it(list(enumerate(val_dataset0))))[
            args.from_idx : args.to_idx
        ]

        val_dataset1 = Seq2SeqAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
        )
        preds = [e["pred"] for e in val_dataset1]

        all_guess = {}
        all_rephrases = {}
        all_alts = {}

        iter_ = tqdm(val_dataset0)

        for j, d0 in iter_:
            all_alts[j] = d0["alt"]

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

            batch = {
                k: v.to(tmodel.device)
                for k, v in val_dataset1.collate_fn([d0]).items()
                if isinstance(v, torch.Tensor)
            }
            while normalize(tmodel.sample(d0["src"])[0][0]) != normalize(d0["alt"]):

                logits = tmodel(batch)
                _, loss = label_smoothed_nll_loss(
                    logits.log_softmax(-1),
                    batch["trg_input_ids"][:, 1:],
                    epsilon=0,
                    ignore_index=model.tokenizer.pad_token_id,
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            all_rephrases[j] = tmodel.sample(d0["view"])

            all_guess_batch = []
            for i, d1 in enumerate(batch_it(tqdm(val_dataset1), args.batch_size)):
                all_guess_batch += tmodel.sample(
                    [e["src"] for e in d1], num_return_sequences=5
                )

            all_guess[j] = all_guess_batch

            iter_.set_postfix(
                succ=sum(
                    normalize(all_alts[k]) == normalize(v[k][0])
                    for k, v in all_guess.items()
                )
                / len(all_guess),
                retain=sum(
                    (
                        sum(a == b for a, b in zip(preds[:k], [e[0] for e in v[:k]]))
                        + sum(
                            a == b
                            for a, b in zip(preds[k + 1 :], [e[0] for e in v[k + 1 :]])
                        )
                    )
                    / (len(v) - 1)
                    for k, v in all_guess.items()
                )
                / len(all_guess),
                equiv=sum(
                    sum(e[0] == all_guess[k][k][0] for e in v) / len(v)
                    for k, v in all_rephrases.items()
                )
                / len(all_rephrases),
            )

        filename = os.path.join(
            args.output_path,
            f"all_guess-{args.from_idx}-{args.to_idx}-baseline-{args.layer}.pkl",
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_guess, f)

        filename = os.path.join(
            args.output_path,
            f"all_rephrases-{args.from_idx}-{args.to_idx}-baseline-{args.layer}.pkl",
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_rephrases, f)

    elif args.method == "hyper":
        model = BartSeq2SeqAugmented.load_from_checkpoint(args.model)
        model.model = BartSeq2Seq.load_from_checkpoint(
            model.hparams.model_checkpoint
        ).model
        model = model.eval().to(args.device)

        val_dataset0 = Seq2SeqAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
            return_view=True,
            all_views=True,
        )
        val_dataset0 = list(shuffle_it(list(enumerate(val_dataset0))))[
            args.from_idx : args.to_idx
        ]

        val_dataset1 = Seq2SeqAugmentedKILT(
            tokenizer=model.tokenizer,
            data_path=model.hparams.dev_data_path,
            max_length=model.hparams.max_length,
        )
        preds = [e["pred"] for e in val_dataset1]

        model.val_dataloader(shuffle=False)

        all_guess = {}
        all_rephrases = {}
        all_alts = {}

        iter_ = tqdm(val_dataset0)
        for j, d0 in iter_:
            all_alts[j] = d0["alt"]

            with torch.no_grad():
                _, params_dict = model.get_logits_orig_params_dict(
                    {
                        k: v.to(model.device)
                        for k, v in model.val_dataset.get_batch([], d0["cond"]).items()
                    }
                )

            _, guess = model.sample(
                d0["view"],
                d0["cond"],
                params_dict,
                num_return_sequences=5,
                stop_condition=lambda condition, guess, n_iter: (
                    normalize(condition.split(" || ")[0].split(" >> ")[1])
                    != normalize(guess[0][0])
                )
                and n_iter < 5,
            )

            all_rephrases[j] = guess

            all_guess_batch = []
            for i, d1 in enumerate(batch_it(tqdm(val_dataset1), args.batch_size)):
                _, guess = model.sample(
                    [e["src"] for e in d1],
                    d0["cond"],
                    params_dict,
                    num_return_sequences=5,
                )
                all_guess_batch += guess

            all_guess[j] = all_guess_batch

            iter_.set_postfix(
                succ=sum(
                    normalize(all_alts[k]) == normalize(v[k][0])
                    for k, v in all_guess.items()
                )
                / len(all_guess),
                retain=sum(
                    (
                        sum(a == b for a, b in zip(preds[:k], [e[0] for e in v[:k]]))
                        + sum(
                            a == b
                            for a, b in zip(preds[k + 1 :], [e[0] for e in v[k + 1 :]])
                        )
                    )
                    / (len(v) - 1)
                    for k, v in all_guess.items()
                )
                / len(all_guess),
                equiv=sum(
                    sum(e[0] == all_guess[k][k][0] for e in v) / len(v)
                    for k, v in all_rephrases.items()
                )
                / len(all_rephrases),
            )

        filename = os.path.join(
            args.output_path, f"all_guess-{args.from_idx}-{args.to_idx}.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_guess, f)

        filename = os.path.join(
            args.output_path, f"all_rephrases-{args.from_idx}-{args.to_idx}.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_rephrases, f)

        filename = os.path.join(
            args.output_path, f"all_alts-{args.from_idx}-{args.to_idx}.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(all_alts, f)
