from SGAN.seqgan import SeqGAN
from SGAN.get_config import get_config

config = get_config('config.ini')

# build SGAN model
trainer = SeqGAN(config["batch_size"], config["max_length"], config["g_e"], config["g_h"], config["d_e"], config["d_h"], config["d_dropout"], path_pos=config["path_pos"], path_neg=config["path_neg"], g_lr=config["g_lr"], d_lr=config["d_lr"], n_sample=config["n_sample"], generate_samples=config["generate_samples"])

# Pretraining generator and discriminator
trainer.pre_train(g_epochs=config["g_pre_epochs"],d_epochs=config["d_pre_epochs"],g_pre_path=config["g_pre_weights_path"],d_pre_path=config["d_pre_weights_path"],g_lr=config["g_pre_lr"],d_lr=config["d_pre_lr"])
print("pre train finished")

# load model
trainer.load_pre_train(config["g_pre_weights_path"], config["d_pre_weights_path"])
print("load pre train finished")
trainer.reflect_pre_train()

# start GAN training
trainer.train(steps=config["gan_steps"],g_steps=config["gan_g_steps"],head=config["gan_head"],g_weights_path=config["gan_g_weights_path"],d_weights_path=config["gan_d_weights_path"])

print("train GAN finished")
trainer.save(config["gan_g_weights_path"], config["gan_d_weights_path"])