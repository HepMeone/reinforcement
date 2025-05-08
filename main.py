# main.py

from config.config import CONFIG
from training.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(CONFIG)
    trainer.train()
