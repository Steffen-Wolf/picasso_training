from PIL import Image
from torchvision import transforms

from src.models.dsl_model import DSLModel
from src.datamodules.dsl_datamodule import ZarrDataset
from torch.utils.data import DataLoader
import os
import h5py
import numpy as np
from tqdm import tqdm

def from_logspace(indata):
    data = np.power(10, indata)
    data -= 1e-14
    data[data <= 0] = 0
    return data

def predict():

    CKPT_PATH = "/lmb/home/swolf/local/src/dsl_training/logs/runs/2021-07-13/20-48-06/checkpoints/epoch=13-v1.ckpt"#epoch=05.ckpt"
    TEST_DS_PATH = "/lmb/home/swolf/local/src/dsl_training/data/test.zarr"
    OUT_PATH = "/lmb/home/swolf/local/src/dsl_training/data/prediction_00"
    GT_OUT_PATH = "/lmb/home/swolf/local/src/dsl_training/data/gt"

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = DSLModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)
    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    try:
        os.mkdir(OUT_PATH, exists_ok=True)
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

    test_ds = ZarrDataset(TEST_DS_PATH, downsampeling)
    dl = DataLoader(dataset=test_ds,
                    batch_size=1,
                    num_workers=2,
                    shuffle=False)

    # inference
    for idx, (inp, gt) in tqdm(enumerate(dl)):
        output = trained_model(inp)
        output = output.cpu().numpy()
        output_log = from_logspace(output)
        gt = gt.cpu().numpy()
        mask = gt == -100
        gt_log = from_logspace(gt)
        output_log_masked = output_log.copy()
        output_log_masked[mask] = 0
        gt_log[mask] = 0

        with h5py.File(f"{OUT_PATH}/halo_data_run00_{idx:06}.h5", "w") as outfile:
            for i, k in enumerate(keys):
                outfile.create_dataset(k, data=output_log[0, i], compression="gzip")
                outfile.create_dataset(k+"_masked", data=output_log_masked[0, i], compression="gzip")

        with h5py.File(f"{GT_OUT_PATH}/halo_data_gt_{idx:06}.h5", "w") as outfile:
            for i, k in enumerate(keys):
                outfile.create_dataset(k, data=gt_log[0, i], compression="gzip")

if __name__ == "__main__":
    predict()
