![U-Net](images/U-Net.png)

# Install

Virtual environment:
```shell
# shell
pip install --user virtualenv
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Jupyter kernel (assuming you already have jupyter installed):
```shell
# shell
python -m ipykernel install --user --name=nn-rotor
```

# Data

To recreate plots, download files from [here](https://drive.google.com/drive/folders/1VXw3qgYSQRavZpY1ewHgb3oxRYJaCQdX?usp=sharing) to `./data`.

# Notebooks
*Logically sorted*

[`001-Synthetic-rotors.ipynb`](https://github.com/humanphysiologylab/nn-rotor/blob/master/notebooks/001-Synthetic-rotors.ipynb).
Synthetic dataset generation. You may download the original dataset ([`synthetic-latest.csv`](https://drive.google.com/file/d/1A1YR4p_DB3ssWP8fE33VfHCVQ-54Nh5j/view?usp=sharing))  or generate yours.

[`015-Draw-synthetic.ipynb`](https://github.com/humanphysiologylab/nn-rotor/blob/master/notebooks/015-Draw-synthetic.ipynb).
Example of the synthetic track.

[`014-UNet.ipynb`](https://github.com/humanphysiologylab/nn-rotor/blob/master/notebooks/014-UNet.ipynb).
Training of the neural network.

[`013-illustration.ipynb`](https://github.com/humanphysiologylab/nn-rotor/blob/master/notebooks/013-illustration.ipynb).
Illustration of how neural network predicts things.

[`017-Predict-pipeline.ipynb`](https://github.com/humanphysiologylab/nn-rotor/blob/master/notebooks/017-Predict-pipeline.ipynb).
Segmentation of the raw trajectory, the final step of the [rotor search pipeline](https://github.com/humanphysiologylab/heart-meshes#readme). 

# Credits

U-Net implementation is taken from [here](https://github.com/milesial/Pytorch-UNet/tree/master/unet) and slightly modified.

Current repository is maintained by Andrey Pikunov (pikunov@phystech.edu).
