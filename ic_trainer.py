import pytorch_lightning as pl
import src.model as model
import json
import os

pl.seed_everything(7)

def main(hparams):
    ic_model = model.CNNLSTM(hparams)
    logger = pl.loggers.TensorBoardLogger('lightning_logs')
    trainer = pl.Trainer(max_epochs=20, row_log_interval=1, logger=logger, gpus=[0])
    trainer.fit(ic_model)
    print('Loading best model')
    latest_version = os.listdir('lightning_logs/default').sort()[0]
    best_checkpoint = os.listdir(f'lightning_logs/default/{latest_version}').sort()[0]
    ic_model.load_from_checkpoint(f'lightning_logs/default/{latest_version}/{best_checkpoint}')
    print('Evaluating model')
    trainer.test(ic_model)


if __name__ == '__main__':

    word2idx = json.load(open('data/word2idx.json'))

    hparams = {
        'vocab_size': len(word2idx) + 1, 
        'embed_size': 512, # Size of embedding layer
        'hidden_size1': 768, # Size of hidden layer for LSTM
        'hidden_size2': 4096, # Size of hidden layer after LSTM
        'n_layers': 3, # Number of stacker layers in LSTM
        'batch_size': 80
    }

    main(hparams)