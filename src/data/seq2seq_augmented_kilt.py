import jsonlines
import numpy as np
from torch.utils.data import Dataset


class Seq2SeqAugmentedKILT(Dataset):
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
                            k: d[k]
                            for k in (
                                "input",
                                "prediction",
                                "alternatives",
                                "filtered_rephrases",
                                "output",
                            )
                        }
                    )

        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, seed=None):
        alt = np.random.RandomState(seed=seed).choice(self.data[item]["alternatives"])
        output = {
            "src": self.data[item]["input"],
            "pred": self.data[item]["prediction"],
            "alt": alt,
            "answers": [x["answer"] for x in self.data[item]["output"]],
            "cond": "{} >> {} || {}".format(
                self.data[item]["prediction"],
                alt,
                self.data[item]["input"],
            ),
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
                "trg": [condition.split(" || ")[0].split(" >> ")[1]]
                * (len(sentences) + 1 + int(self.return_view)),
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
        batch["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
        return batch

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        trg = [b["pred"] for b in batch[:-1]] + [batch[-1]["alt"]]

        if self.return_view:
            src += batch[-1]["view"] if self.all_views else [batch[-1]["view"]]
            trg += [batch[-1]["alt"]] * (
                len(batch[-1]["view"]) if self.all_views else 1
            )

        batches = {
            "{}_{}".format(k1, k2): v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
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

        batches["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
        batches["raw"] = batch
        return batches
