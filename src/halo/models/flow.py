from copy import deepcopy
from typing import Sequence, Any
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import zuko
import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- Normalizing Flow ----
class FlowNetwork(L.LightningModule):
    """A conditional normalizing flow around a supplied context encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        context_dim: int = 32,
        target_dim: int = 1,
        transforms: int = 6,
        flow_hidden_dims: Sequence[int] = (128, 128, 128),
        learning_rate: float = 3e-4,
        scheduler_patience: int = 10,
        validation_samples: int = 100,
        prediction_samples: int = 200,
    ) -> None:
        super().__init__()
        if context_dim < 1 or target_dim < 1:
            raise ValueError("context_dim and target_dim must be positive")
        if validation_samples < 1 or prediction_samples < 1:
            raise ValueError("sample counts must be positive")

        self.encoder = encoder
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.validation_samples = validation_samples
        self.prediction_samples = prediction_samples
        self.flow = zuko.flows.MAF(
            features=target_dim,
            context=context_dim,
            transforms=transforms,
            hidden_features=tuple(flow_hidden_dims),
        )
        self.validation_summaries: list[dict[str, Tensor]] = []
        self.save_hyperparameters(ignore=("encoder",))

    @staticmethod
    def _split_batch(batch: Any) -> tuple[Any, Tensor]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, target = batch
        elif hasattr(batch, "y"):
            inputs, target = batch, batch.y
        else:
            raise TypeError("batch must be an (inputs, target) pair or a graph batch with .y")
        if not isinstance(target, Tensor):
            raise TypeError("target must be a torch.Tensor")
        return inputs, target

    def _normalize_target(self, target: Tensor) -> Tensor:
        if target.ndim == 0:
            target = target.reshape(1, 1)
        elif target.ndim == 1:
            if self.target_dim == 1:
                target = target.reshape(-1, 1)
            elif target.numel() == self.target_dim:
                target = target.reshape(1, self.target_dim)
            else:
                raise ValueError(
                    f"cannot interpret target shape {tuple(target.shape)} for target_dim={self.target_dim}"
                )
        else:
            target = target.reshape(target.shape[0], -1)

        if target.shape[-1] != self.target_dim:
            raise ValueError(
                f"expected target width {self.target_dim}, got {target.shape[-1]}"
            )
        return target

    def encode(self, inputs: Any) -> Tensor:
        context = self.encoder(inputs)
        if not isinstance(context, Tensor) or context.ndim != 2:
            shape = getattr(context, "shape", None)
            raise ValueError(f"encoder must return [batch, context] tensor, got {shape}")
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"encoder returned context width {context.shape[-1]}, expected {self.context_dim}"
            )
        return context

    def forward(self, inputs: Any):
        return self.flow(self.encode(inputs))

    def negative_log_likelihood(self, inputs: Any, target: Tensor) -> Tensor:
        target = self._normalize_target(target)
        distribution = self(inputs)
        return -distribution.log_prob(target).mean()

    def training_step(self, batch: Any, batch_idx: int = 0) -> Tensor:
        inputs, target = self._split_batch(batch)
        target = self._normalize_target(target)
        distribution = self(inputs)
        loss = -distribution.log_prob(target).mean()
        batch_size = target.shape[0]
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)

        optimizer = self.optimizers()
        self.log(
            "learning_rate",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int = 0) -> Tensor:
        inputs, target = self._split_batch(batch)
        target = self._normalize_target(target)
        distribution = self(inputs)
        loss = -distribution.log_prob(target).mean()
        samples = distribution.sample((self.validation_samples,))
        prediction_mean = samples.mean(dim=0)
        prediction_std = samples.std(dim=0)
        mse = F.mse_loss(prediction_mean, target)
        batch_size = target.shape[0]

        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("mse_loss", mse, batch_size=batch_size)
        self.validation_summaries.append(
            {
                "target": target.detach().cpu(),
                "mean": prediction_mean.detach().cpu(),
                "std": prediction_std.detach().cpu(),
            }
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.validation_summaries:
            return

        target = torch.cat([item["target"] for item in self.validation_summaries])
        mean = torch.cat([item["mean"] for item in self.validation_summaries])
        std = torch.cat([item["std"] for item in self.validation_summaries])
        self.validation_summaries.clear()

        fig, ax = plt.subplots()
        ax.errorbar(
            target[:, 0].numpy(),
            mean[:, 0].numpy(),
            yerr=std[:, 0].numpy(),
            linestyle="",
            marker="o",
            markersize=1,
            alpha=0.4,
        )
        limits = [target[:, 0].min().item(), target[:, 0].max().item()]
        ax.plot(limits, limits, linestyle="--", color="lightgray")
        ax.set_xlabel("True target")
        ax.set_ylabel("Posterior mean")

        logger = self.logger if self._trainer is not None else None
        if logger is not None and hasattr(logger, "log_image"):
            logger.log_image(key="validation_predictions", images=[fig])
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def predict_step(self, batch: Any, batch_idx: int = 0) -> dict[str, Tensor]:
        inputs, target = self._split_batch(batch)
        target = self._normalize_target(target)
        distribution = self(inputs)
        return {
            "target": target,
            "log_prob": distribution.log_prob(target),
            "samples": distribution.sample((self.prediction_samples,)),
        }

    def set_model_config(
        self,
        model_type: str,
        model_kwargs: dict[str, Any],
    ) -> None:
        self._model_type = model_type
        self._model_kwargs = deepcopy(model_kwargs)


    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if hasattr(self, "_model_type"):
            checkpoint["model_type"] = self._model_type
            checkpoint["model_kwargs"] = deepcopy(self._model_kwargs)
