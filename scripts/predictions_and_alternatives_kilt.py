import argparse
import logging
import string

import jsonlines
from Levenshtein import distance
from tqdm.auto import tqdm

from src.models.bart_seq2seq_kilt import BartSeq2Seq
from src.models.bert_binary_kilt import BertBinary
from src.utils import batch_it, chunk_it


def normalize(sent):
    return (
        sent.lower()
        .replace(" ", "")
        .translate(str.maketrans("", "", string.punctuation))
    )


def predictions_and_alternatives(model, sentences, binary):
    if binary:
        return [
            (
                p[0],
                ["SUPPORTS" if p[0] == "REFUTES" else "REFUTES"],
                p[1],
            )
            for p in model.sample(sentences)
        ]
    else:
        return [
            (
                p[0],
                list(
                    set(
                        [
                            a.replace(".", "")
                            for a in p[1:]
                            if (len(a) < 5 and normalize(p[0]) != normalize(a))
                            or distance(normalize(p[0]), normalize(a)) > 4
                        ]
                    ).difference({p[0]})
                ),
                None,
            )
            for p in batch_it(
                model.sample(
                    sentences,
                    min_length=0,
                    num_beams=5,
                    num_return_sequences=5,
                ),
                5,
            )
        ]


def filtered_rephrases(model, input_, rephrases, binary):
    pred = model.sample(
        [input_] + rephrases,
        min_length=0,
        num_beams=5,
        num_return_sequences=1,
    )
    if binary:
        return [r for p, r in zip(pred[1:], rephrases) if p[0] == pred[0][0]]
    else:
        return [
            r for p, r in zip(pred[1:], rephrases) if normalize(p) == normalize(pred[0])
        ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_filename",
        type=str,
        help="Filename of the KILT dataset",
        default="../datasets/structured_zeroshot-dev-new.jsonl",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        help="Filename of the KILT dataset",
        default="../datasets/structured_zeroshot-dev-new_annotated.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Filename of the model",
        default="models/bart_seq2seq_structured_zeroshot/version_0/checkpoints/model-epoch=17-valid_acc=0.2207.ckpt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--binary",
        action="store_true",
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

    logging.info("Loading model")
    if args.binary:
        model = (
            BertBinary.load_from_checkpoint(args.model, strict=False)
            .eval()
            .to(args.device)
        )
    else:
        model = (
            BartSeq2Seq.load_from_checkpoint(args.model, strict=False)
            .eval()
            .to(args.device)
        )

    model.freeze()

    filename = args.input_filename
    logging.info("Loading {}".format(filename))
    with jsonlines.open(filename) as f:
        dataset = list(f)

    if not args.binary:
        dataset = [
            {**d, "input": q} for d in dataset for q in d["meta"]["template_questions"]
        ]

    for docs in batch_it(tqdm(dataset, desc="Predicting"), args.batch_size):

        for d, (p, a, l) in zip(
            docs,
            predictions_and_alternatives(
                model,
                [d["input"] for d in docs],
                args.binary,
            ),
        ):
            d["prediction"] = p
            d["alternatives"] = a
            d["filtered_rephrases"] = filtered_rephrases(
                model,
                d["input"],
                d["rephrases"],
                args.binary,
            )
            if l:
                d["logit"] = l.item()

    filename = args.output_filename
    logging.info("Saving {}".format(filename))
    with jsonlines.open(filename, "w") as f:
        f.write_all(dataset)
