import pandas as pd
import torch
from torch.nn.functional import mse_loss
import lightning as L
import zuko
import wandb
import io
import matplotlib.pyplot as plt
from PIL import Image

def log_matplotlib_figure(figure_label: str):
    """log a matplotlib figure to wandb, avoiding plotly

    Args:
        figure_label (str): label for figure
    """
    # Save plot to a buffer, otherwise wandb does ugly plotly
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=300,
    )
    buf.seek(0)
    image = Image.open(buf)
    # Log the plot to wandb
    wandb.log({f"{figure_label}": wandb.Image(image)})
    

class LinearModel(L.LightningModule):
    def __init__(self, context=11, transforms=6, hidden_features=[128,128,128]):
        '''
        Notes on Zuko notation--
        features: desired output dimension; In this case, just the halo mass, so it is 1
        context: the input dimension; or the amount of trainable features, in this case it is 11
        ''' 
        super().__init__()
        self.flow = zuko.flows.MAF(features=1, transforms=transforms, context=context,hidden_features=hidden_features)
        self.save_hyperparameters()
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # logging loss
        loss = -self.flow(x).log_prob(y).mean() # -log p(y|x)
        # flow conditions flow on the properties, and log prob estimates the probability
        self.log("train_loss", loss, prog_bar=True)
        
        # logging learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self.flow(x)
        
        loss = -y_preds.log_prob(y).mean()
        self.log("val_loss", loss, prog_bar=True)
        
        # calculating mse loss
        mse_pred = y_preds.sample((100,)).squeeze()
        mse = mse_loss(mse_pred.mean(axis=0),y.squeeze())
        self.log("mse_loss", mse)
        
        self.validation_step_outputs.append({"val_loss": loss, "batch": batch})
        return loss
    
    def on_validation_epoch_end(self):
        preds = []
        y = []
        for output in self.validation_step_outputs:
            x = output["batch"][0]
            y.extend(output["batch"][1].detach().cpu().numpy().squeeze())
            preds.append(self.flow(x).sample((100,)).squeeze())
        
        pred = torch.cat(preds, dim=1).squeeze()
        pred_mean = pred.mean(axis=0).detach().cpu().numpy().squeeze()
        pred_std = pred.std(axis=0).detach().cpu().numpy().squeeze()
        
        # errorbar line
        plt.errorbar(
            y,
            pred_mean,
            yerr=pred_std,
            linestyle='',
            marker='o',
            markersize=1,
        )
        # truth line
        plt.plot(
            y,
            y,
            linestyle='--',
            color='lightgray',
        )
        log_matplotlib_figure("validation_predictions")
        plt.close()
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": f"val_loss",
        }