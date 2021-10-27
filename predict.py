from PIL import Image
from torchvision import transforms

from src.models.dsl_model import DSLModel
from src.utils.utils import from_logspace, read_yaml_conf
from src.datamodules.dsl_datamodule import ZarrDataset
from torch.utils.data import DataLoader

import os
import h5py
import numpy as np
from tqdm import tqdm
import yaml


def predict():

    # CKPT_PATH = "/lmb/home/swolf/local/src/dsl_training/logs/runs/2021-07-13/20-48-06/checkpoints/epoch=13-v1.ckpt"#epoch=05.ckpt"
    WRITE_GT = True
    CHANNEL_EXPERIMENT_ROOT = "/lmb/home/swolf/local/src/dsl_training/logs/2021-07-22_18-24-33"
    TEST_DS_PATH = "/lmb/home/swolf/local/src/dsl_training/data/test.zarr"
    GT_OUT_PATH = "/ssd/swolf/dsl/gt"

    # predict channel experiment
    for i in range(1):

        # load channels from json overwrite file
        config_file = f"{CHANNEL_EXPERIMENT_ROOT}/{i}/.hydra/overrides.yaml"
        config = read_yaml_conf(config_file)

        used_channels = [_.split("=")[-1] for _ in config
                         if _.startswith("datamodule.channels")][0]

        # load model from checkpoint
        # model __init__ parameters will be loaded from ckpt automatically
        # you can also pass some parameter explicitly to override it
        checkpoint_path = f"{CHANNEL_EXPERIMENT_ROOT}/{i}/checkpoints/last.ckpt"
        trained_model = DSLModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        # print model hyperparameters
        print(trained_model.hparams)
        # switch to evaluation mode
        trained_model = trained_model.cuda()
        trained_model.eval()
        trained_model.freeze()

        out_path = f"/ssd/swolf/dsl/prediction_channel_experiments/{used_channels}/"
        try:
            os.mkdir(out_path, exists_ok=True)
        except:
            pass

        # TODO: get downsampeling from config or log
        downsampeling = 0

        keys = ['gas_GFM_Metallicity',
                'gas_Masses',
                'gas_NeutralHydrogenAbundance',
                'gas_StarFormationRate',
                'stars_GFM_Metallicity',
                'stars_GFM_StellarFormationTime',
                'stars_Masses']

        # load input image

        test_ds = ZarrDataset(TEST_DS_PATH, downsampeling, channels=used_channels)
        dl = DataLoader(dataset=test_ds,
                        batch_size=1,
                        num_workers=2,
                        shuffle=False)

        # inference
        for idx, (inp, gt) in tqdm(enumerate(dl)):
            output = trained_model(inp.cuda())
            output = output.cpu().numpy()
            output_log = from_logspace(output)
            gt = gt.cpu().numpy()
            mask = gt == -100
            gt_log = from_logspace(gt)
            output_log_masked = output_log.copy()
            output_log_masked[mask] = 0
            gt_log[mask] = 0

            with h5py.File(f"{out_path}/halo_data_{used_channels}_{idx:06}.h5", "w") as outfile:
                for i, k in enumerate(keys):
                    outfile.create_dataset(k, data=output_log[0, i], compression="gzip")
                    outfile.create_dataset(k+"_masked", data=output_log_masked[0, i], compression="gzip")

            if WRITE_GT:
                with h5py.File(f"{GT_OUT_PATH}/halo_data_gt_{idx:06}.h5", "w") as outfile:
                    for i, k in enumerate(keys):
                        outfile.create_dataset(k, data=gt_log[0, i], compression="gzip")

if __name__ == "__main__":
    predict()
