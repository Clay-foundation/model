# Basic Use

### Running jupyter lab

    mamba activate claymodel
    python -m ipykernel install --user --name claymodel  # to install virtual env properly
    jupyter kernelspec list --json                       # see if kernel is installed
    jupyter lab &


### Running the model

The neural network model can be ran via
[LightningCLI v2](https://pytorch-lightning.medium.com/introducing-lightningcli-v2supercharge-your-training-c070d43c7dd6).
To check out the different options available, and look at the hyperparameter
configurations, run:

    python trainer.py --help
    python trainer.py test --print_config

To quickly test the model on one batch in the validation set:

    python trainer.py validate --trainer.fast_dev_run=True

To train the model for a hundred epochs:

    python trainer.py fit --trainer.max_epochs=100

To generate embeddings from the pretrained model's encoder on 1024 images
(stored as a GeoParquet file with spatiotemporal metadata):

    python trainer.py predict --ckpt_path=checkpoints/last.ckpt \
                              --data.batch_size=1024 \
                              --data.data_dir=s3://clay-tiles-02 \
                              --trainer.limit_predict_batches=1

More options can be found using `python trainer.py fit --help`, or at the
[LightningCLI docs](https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html).

## Advanced

See [Readme](../../README.md) on model root for more details.
