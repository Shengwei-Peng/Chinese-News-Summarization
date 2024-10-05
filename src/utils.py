"""utils.py"""
import os
import json
from pathlib import Path

class Evaluator:
    """Evaluator"""
    def __init__(self, output_dir: Path = None, refs_file: Path = None) -> None:
        self.output_dir = output_dir
        self.history = []
        self.ws = None
        self.rouge = None
        if refs_file:
            refs = {
                line['id']: line['title'].strip() + '\n'
                for line in map(json.loads, refs_file.read_text(encoding="utf-8").splitlines())
            }
            self.refs = list(refs.values())
        else:
            self.refs = None

    def tw_rouge_init(self) -> None:
        """tw_rouge_init"""
        from rouge import Rouge
        from ckiptagger import WS, data_utils

        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        download_dir = cache_dir / "ckiptagger"
        data_dir = download_dir / "data"
        os.makedirs(download_dir, exist_ok=True)

        os.environ["TF_USE_LEGACY_KERAS"] = "1"

        if not (data_dir / "model_ws").exists():
            data_utils.download_data_gdown(str(download_dir))

        self.ws = WS(data_dir, disable_cuda=False)
        self.rouge = Rouge()

    def get_rouge(
        self, preds: list, refs: list, avg: bool = True, ignore_empty: bool = False
        ) -> dict:
        """get_rouge"""
        if (self.ws is None or self.rouge is None) and self.refs is not None:
            self.tw_rouge_init()

        def tokenize_and_join(sentences):
            return [" ".join(toks) for toks in self.ws(sentences)]

        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(refs, list):
            refs = [refs]
        preds, refs = tokenize_and_join(preds), tokenize_and_join(refs)

        scores = self.rouge.get_scores(preds, refs, avg=avg, ignore_empty=ignore_empty)
        rouge_scores = {i: {j: scores[i][j] * 100 for j in scores[i]} for i in scores}
        print(
            f"ROUGE-1: {rouge_scores['rouge-1']['f']:.1f}, "
            f"ROUGE-2: {rouge_scores['rouge-2']['f']:.1f}, "
            f"ROUGE-L: {rouge_scores['rouge-l']['f']:.1f}"
        )

        return rouge_scores

    def add(self, loss: float, predictions: list = None) -> None:
        """add"""
        if predictions is None or self.refs is None:
            rouge = {
                "rouge-1": {"r": 0, "p": 0, "f": 0},
                "rouge-2": {"r": 0, "p": 0, "f": 0},
                "rouge-l": {"r": 0, "p": 0, "f": 0}
            }
        else:
            rouge = self.get_rouge(predictions, self.refs)

        self.history.append({
            "rouge": rouge,
            "loss": loss
        })

    def plot_learning_curves(self) -> None:
        """plot_learning_curves"""
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.history) + 1)
        metric_keys = ["rouge-1", "rouge-2", "rouge-l"]
        metric_names = ["Rouge-1", "Rouge-2", "Rouge-L"]

        _, axes = plt.subplots(3, 1, figsize=(15, 18))

        for ax, key, metric_name in zip(axes, metric_keys, metric_names):
            recall = [result["rouge"][key]["r"] for result in self.history]
            precision = [result["rouge"][key]["p"] for result in self.history]
            f_score = [result["rouge"][key]["f"] for result in self.history]

            ax.plot(epochs, recall, label=f"{metric_name} Recall", marker='o')
            ax.plot(epochs, precision, label=f"{metric_name} Precision", marker='o')
            ax.plot(epochs, f_score, label=f"{metric_name} F-score", marker='o')
            ax.set_title(f"{metric_name} Metrics History")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend(loc='lower right')
            ax.grid(True)

        plt.tight_layout()

        if self.output_dir is not None:
            plt.savefig(self.output_dir / "rouge_learning_curve.png")
            plt.close()

        losses = [entry["loss"] for entry in self.history if entry["loss"] is not None]

        if losses:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, label="Loss", marker='o')
            plt.title("Loss History")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.grid(True)

            if self.output_dir is not None:
                plt.savefig(self.output_dir / "loss_learning_curve.png")
                plt.close()

        if self.output_dir is not None:
            history_file = self.output_dir / "history.json"
            with open(history_file, "w", encoding="utf-8") as file:
                json.dump(self.history, file, indent=4)
