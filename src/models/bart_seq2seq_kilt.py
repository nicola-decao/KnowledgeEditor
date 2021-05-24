from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.seq2seq_kilt import Seq2SeqKILT
from src.utils import label_smoothed_nll_loss, batch_it


class BartSeq2Seq(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="../datasets/structured_zeroshot-train-new.jsonl",
        )
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default="../datasets/structured_zeroshot-dev-new.jsonl",
        )
        parser.add_argument("--batch_size", type=int, default=48)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=50000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=0)

        parser.add_argument("--model_name", type=str, default="facebook/bart-base")
        parser.add_argument("--eps", type=float, default=0.1)
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(
            self.hparams.model_name
        )

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset"):
            self.train_dataset = Seq2SeqKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path,
                max_length=self.hparams.max_length,
                templates=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = Seq2SeqKILT(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path,
                max_length=self.hparams.max_length,
                validation=True,
                templates=True,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def forward(self, batch):
        return self.model(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            decoder_input_ids=batch["trg_input_ids"][:, :-1],
            decoder_attention_mask=batch["trg_attention_mask"][:, :-1],
            use_cache=False,
        ).logits
    
    def training_step(self, batch, batch_idx=None):
        logits = self.forward(batch)

        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(-1),
            batch["trg_input_ids"][:, 1:],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        ntokens = batch["trg_attention_mask"][:, 1:].sum()
        loss, nll_loss = loss / ntokens, nll_loss / ntokens

        self.log("nll_loss", nll_loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        gold = [b["trg"] for b in batch["raw"]]
        guess = self.tokenizer.batch_decode(
            self.model.generate(
                input_ids=batch["src_input_ids"],
                attention_mask=batch["src_attention_mask"],
                min_length=0,
                num_beams=5,
                num_return_sequences=1,
            ),
            skip_special_tokens=True,
        )

        acc = torch.tensor(
            [
                a.lower().strip() in [c.lower().strip() for c in b]
                for a, b in zip(guess, gold)
            ]
        ).long()
        self.valid_acc(acc, torch.ones_like(acc))
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True
        )

    def sample(self, sentences, num_return_sequences=1):
        self.eval()
        with torch.no_grad():
            return list(
                batch_it(
                    self.tokenizer.batch_decode(
                        self.model.generate(
                            **{
                                k: v.to(self.device)
                                for k, v in self.tokenizer(
                                    sentences,
                                    return_tensors="pt",
                                    padding=True,
                                    max_length=self.hparams.max_length,
                                    truncation=True,
                                ).items()
                            },
                            min_length=0,
                            num_beams=5,
                            num_return_sequences=num_return_sequences,
                        ),
                        skip_special_tokens=True,
                    ),
                    num_return_sequences,
                )
            )

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
