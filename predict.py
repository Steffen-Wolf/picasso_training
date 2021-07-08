from PIL import Image
from torchvision import transforms

from src.models.dsl_model import DSLModel
from src.datamodules.dsl_datamodule import ZarrDataset
from torch.utils.data import DataLoader
import os
import h5py
import numpy as np
from tqdm import tqdm

def from_logspace(data):
    np.power(10, data, out=data)
    data -= 1e-14
    data[data <= 0] = 0
    return data

def predict():

    CKPT_PATH = "/lmb/home/swolf/local/src/dsl_training/logs/runs/2021-06-14/17-02-53/checkpoints/epoch=08.ckpt"
    TEST_DS_PATH = "/lmb/home/swolf/local/src/dsl_training/data/test.zarr"
    OUT_PATH = "/lmb/home/swolf/local/src/dsl_training/data/prediction_00"

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
        
        output = from_logspace(output.cpu().numpy())

        with h5py.File(f"{OUT_PATH}/prediction_{idx:06}.h5", "w") as outfile:
            for i, k in enumerate(keys):
                outfile.create_dataset(k, data=output[0, i], compression="gzip")

if __name__ == "__main__":
    predict()