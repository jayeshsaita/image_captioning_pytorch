import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import src.dataset as dataset
import src.utils as utils


class CNNLSTM(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.vocab_size = hparams['vocab_size']
        self.embed_size = hparams['embed_size']
        self.hidden_size1 = hparams['hidden_size1']
        self.hidden_size2 = hparams['hidden_size2']

        self.img_hidden = nn.Linear(2048, 1024)
        self.img_hidden_dropout = nn.Dropout(p=0.3)
        self.img_bn = nn.BatchNorm1d(1024)
        self.img_embedding = nn.Linear(1024, self.embed_size)

        self.vocab_embedding = nn.Embedding(self.vocab_size, self.embed_size)

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size1, num_layers=hparams['n_layers'], batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_size2)
        self.fc1_dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(self.hidden_size2, self.vocab_size)

    def forward(self, x):
        img_features = x[0]
        caption = x[1]

        feature_embed = self.img_hidden(img_features)
        feature_embed = self.img_hidden_dropout(feature_embed)
        feature_embed = self.img_bn(feature_embed)
        feature_embed = F.relu(feature_embed)
        feature_embed = self.img_embedding(feature_embed) 

        caption_embed = self.vocab_embedding(caption[:, :-1]) 

        feature_embed = feature_embed.unsqueeze(dim=1)
        inp = torch.cat([feature_embed, caption_embed], dim=1)
        out, hidden = self.lstm(inp)
        out = out.contiguous().view(-1, self.hidden_size1)
        out = self.fc1(out)
        out = self.fc1_dropout(out)
        out = F.relu(self.fc1_bn(out))
        out = self.fc2(out)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y[0].view(-1))
        self.logger.experiment.add_scalar('train_loss', loss.item())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x,y = batch
        with torch.no_grad():
            y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y[0].view(-1))
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            out = self(x)
        bleu_score = utils.bleu_4(out, *y)
        return {'bleu_4':bleu_score}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('valid_loss', val_loss_mean)
        return {'val_loss': val_loss_mean}

    def test_epoch_end(self, outputs):
        bleu_mean = torch.Tensor([x['bleu_4'] for x in outputs]).mean()
        return {'bleu_4': bleu_mean}

    def train_dataloader(self):
        train_ds = dataset.ImageCaptioningDataset('data/train_captions.json', 'data/train_features.pkl', 'data/numericalized_padded.json')
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=self.hparams['batch_size']//2, pin_memory=True)
        return train_dl

    def val_dataloader(self):
        valid_ds = dataset.ImageCaptioningDataset('data/valid_captions.json', 'data/valid_features.pkl', 'data/numericalized_padded.json')
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=self.hparams['batch_size']//2, pin_memory=True)
        return valid_dl

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1e-3, eta_min=1e-5)
        return {'optimizer':opt, 'scheduler':lr_scheduler}

    def on_save_checkpoint(self, checkpoint):
        print(f'Saved better model at epoch {checkpoint["epoch"]}')