import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryAugmentedKILT(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=32,
        return_view=False,
        all_views=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []

        with jsonlines.open(data_path) as f:
            for d in f:
                if len(d["alternatives"]) > 0 and len(d["filtered_rephrases"]) > 0:
                    self.data.append(
                        {
                            **{
                                k: d[k]
                                for k in (
                                    "logit",
                                    "input",
                                    "prediction",
                                    "alternatives",
                                    "filtered_rephrases",
                                )
                            },
                            "label": d["output"][0]["answer"],
                        }
                    )

        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        output = {
            "src": self.data[item]["input"],
            "pred": self.data[item]["prediction"] == "SUPPORTS",
            "alt": self.data[item]["alternatives"][0] == "SUPPORTS",
            "cond": "{} >> {} || {}".format(
                self.data[item]["prediction"],
                self.data[item]["alternatives"][0],
                self.data[item]["input"],
            ),
            "logit": self.data[item]["logit"],
        }

        if self.return_view:
            output["view"] = (
                self.data[item]["filtered_rephrases"]
                if self.all_views
                else np.random.choice(self.data[item]["filtered_rephrases"])
            )

        return output

    def get_batch(self, sentences, condition):

        batch = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": sentences
                + [condition.split("|| ")[1]] * (1 + int(self.return_view)),
                "cond": [condition],
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }
        batch["labels"] = torch.tensor(
            [float(condition.startswith("REFUTES >> SUPPORTS"))]
            * (1 + int(self.return_view))
        )
        return batch

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        labels = [b["pred"] for b in batch[:-1]] + [batch[-1]["alt"]]

        if self.return_view:
            src += batch[-1]["view"] if self.all_views else [batch[-1]["view"]]
            labels += [batch[-1]["alt"]] * (
                len(batch[-1]["view"]) if self.all_views else 1
            )

        batches = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": src,
                "cond": [batch[-1]["cond"]],
            }.items()
            for k2, v2 in self.tokenizer(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["labels"] = torch.tensor(labels).float()
        batches["raw"] = batch
        return batches
