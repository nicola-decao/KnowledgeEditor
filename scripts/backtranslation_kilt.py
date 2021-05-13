import argparse
import logging
import string

import jsonlines
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer

from src.utils import batch_it, chunk_it, normalize


def translate(
    sentences, models, tokenizers, direction, num_beams=5, num_return_sequences=5
):
    return tokenizers[direction].batch_decode(
        models[direction].generate(
            **{
                k: v.to(models[direction].device)
                for k, v in tokenizers[direction](
                    sentences, return_tensors="pt", padding=True
                ).items()
            },
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        ),
        skip_special_tokens=True,
    )


def rephrase(
    sentences, models, tokenizers, directions, num_beams=5, num_return_sequences=5
):
    rephrases = translate(
        translate(
            sentences,
            models,
            tokenizers,
            directions[0],
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        ),
        models,
        tokenizers,
        directions[1],
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )

    return [
        list(
            {
                normalize(resent): resent
                for resent in rephrase
                if normalize(resent) != normalize(sent)
            }.values()
        )
        for sent, rephrase in zip(sentences, chunk_it(rephrases, len(sentences)))
    ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_filename",
        type=str,
        help="Filename of the KILT dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="en-de",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--template",
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

    args = parser.parse_args()

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args.langs = [args.langs, "-".join(reversed(args.langs.split("-")))]

    logging.info("Loading models and tokenizers")
    models = {
        k: MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{k}").to(args.device)
        for k in args.langs
    }
    tokenizers = {
        k: MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{k}")
        for k in args.langs
    }

    filename = args.input_filename
    logging.info("Loading {}".format(filename))
    with jsonlines.open(filename) as f:
        dataset = list(f)

    if args.template:
        rephrases = {}
        for qs in batch_it(
            tqdm([q for d in dataset for q in d["meta"]["template_questions"]]),
            args.batch_size,
        ):
            for q, r in zip(
                qs,
                rephrase(
                    qs,
                    models,
                    tokenizers,
                    args.langs,
                ),
            ):
                rephrases[q] = r

        for d in dataset:
            d["rephrases"] = []
            for q in d["meta"]["template_questions"]:
                d["rephrases"].append(rephrases[q])
    else:
        for docs in batch_it(tqdm(dataset, desc="Backtranslating"), args.batch_size):

            for d, r in zip(
                docs,
                rephrase(
                    [d["input"] for d in docs],
                    models,
                    tokenizers,
                    args.langs,
                ),
            ):
                d["rephrases"] = r

    filename = args.input_filename
    logging.info("Saving {}".format(filename))
    with jsonlines.open(filename, "w") as f:
        f.write_all(dataset)
