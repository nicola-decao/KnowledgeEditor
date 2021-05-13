import os
import sys
from argparse import ArgumentParser
from pprint import pprint

import pytorch_lightning
import torch
import transformers
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.models.bert_binary_augmented_kilt import BertBinaryAugmented

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dirpath", type=str, default="models/bert_binary_augmented_fever"
    )
    parser.add_argument("--save_top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser = BertBinaryAugmented.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()

    seed_everything(seed=args.seed)

    logger = TensorBoardLogger(args.dirpath, name=None)

    callbacks = [
        ModelCheckpoint(
            monitor="valid_acc",
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            save_top_k=args.save_top_k,
            filename="model-{epoch:02d}-{valid_acc:.4f}-{valid_flipped:4f}",
        ),
        LearningRateMonitor(
            logging_interval="step",
        ),
    ]

    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)

    model = BertBinaryAugmented(**vars(args))

    trainer.fit(model)

