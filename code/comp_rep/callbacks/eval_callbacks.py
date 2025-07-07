"""
Callbacks for evaluation.
"""

from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation, evaluate_task_faithfulness


class TestGenerationCallback(Callback):
    def __init__(
        self,
        frequency: int,
        searcher: GreedySearch,
        test_loader: DataLoader,
        device: str,
    ):
        self.frequency = frequency
        self.searcher = searcher
        self.test_loader = test_loader
        self.device = device

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Callback function to evaluate the model's accuracy.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer object.
            pl_module (LightningModule): The PyTorch Lightning module.

        Returns:
            None
        """
        epoch = trainer.current_epoch

        if epoch > 0 and epoch % self.frequency == 0:
            acc = evaluate_generation(
                model=pl_module.model,
                searcher=self.searcher,
                test_loader=self.test_loader,
                device=self.device,
            )
            pl_module.log("val_accuracy", acc)

            if hasattr(pl_module, "pruner"):
                remaining_mask_elements = pl_module.pruner.get_remaining_mask()
                pl_module.log(
                    "acc_vs_weights",
                    (1 - acc) + remaining_mask_elements["global_remaining_mask"],
                )


class TestFaithfulnessCallback(Callback):
    def __init__(
        self,
        frequency: int,
        test_loader: DataLoader,
        device: str,
    ):
        self.frequency = frequency
        self.test_loader = test_loader
        self.device = device

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Callback function to evaluate the model's faithfulness.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer object.
            pl_module (LightningModule): The PyTorch Lightning module.

        Returns:
            None
        """
        epoch = trainer.current_epoch

        if epoch % self.frequency == 0:
            avg_jsd_faithfulness, avg_kl_div_faithfulness = evaluate_task_faithfulness(
                model=pl_module.model,
                test_loader=self.test_loader,
                device=self.device,
            )
            pl_module.log("val_jsd_faithfulness", avg_jsd_faithfulness)
            pl_module.log("val_kl_div_faithfulness", avg_kl_div_faithfulness)

            if hasattr(pl_module, "pruner"):
                remaining_mask_elements = pl_module.pruner.get_remaining_mask()
                pl_module.log(
                    "jsd_faithfulness_vs_weights",
                    (1 - avg_jsd_faithfulness)
                    + remaining_mask_elements["global_remaining_mask"],
                )
