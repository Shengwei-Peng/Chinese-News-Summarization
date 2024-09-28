"""utils.py"""
import os
import json
from pathlib import Path

class Evaluator:
    """Evaluator"""
    def __init__(self, output_dir: Path = None) -> None:
        self.output_dir = output_dir
        self.history = []
        self.ws = None
        self.rouge = None

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

    def get_rouge(self, predictions: list, labels: list) -> None:
        """get_rouge"""
        if self.ws is None or self.rouge is None:
            self.tw_rouge_init()

        def tokenize_and_join(sentences):
            return [" ".join(toks) for toks in self.ws(sentences)]

        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(labels, list):
            labels = [labels]

        predictions, labels = tokenize_and_join(predictions), tokenize_and_join(labels)
        scores = self.rouge.get_scores(predictions, labels, avg=True, ignore_empty=True)
        result = {i: {j: scores[i][j] * 100 for j in scores[i]} for i in scores}

        print(
            f"ROUGE-1: {result['rouge-1']['f']:.1f}, "
            f"ROUGE-2: {result['rouge-2']['f']:.1f}, "
            f"ROUGE-L: {result['rouge-l']['f']:.1f}"
        )
        self.history.append(result)

    def plot_learning_curves(self) -> None:
        """plot_learning_curves"""
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.history) + 1)
        metric_keys = ["rouge-1", "rouge-2", "rouge-l"]
        metric_names = ["Rouge-1", "Rouge-2", "Rouge-L"]

        _, axes = plt.subplots(3, 1, figsize=(15, 18))

        for ax, key, metric_name in zip(axes, metric_keys, metric_names):
            recall = [result[key]["r"] for result in self.history]
            precision = [result[key]["p"] for result in self.history]
            f_score = [result[key]["f"] for result in self.history]

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
            plt.savefig(self.output_dir / "learning_curve.png")
            plt.close()

        if self.output_dir is not None:
            history_file = self.output_dir / "history.json"
            with open(history_file, "w", encoding="utf-8") as file:
                json.dump(self.history, file, indent=4)
