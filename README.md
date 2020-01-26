
# DSP-Net Demo

Example data and demo of DSP-Net for 3-D super resolution of microscopy images. 

## Requirements

DSP-Net is built with Python and Tensorflow. Technically there are no limits to the operation system to run the code, but Windows system is recommonded, on which the software has been tested.

The inference process of the DSP-Net can run using the CPU only, but could be inefficiently. A powerful CUDA-enabled GPU device that can speed up the inference is highly recommended. 

The inference process has been tested with:

 * Windows 10 pro (version 1903)
 * Python 3.6.7 (64 bit)
 * tensorflow 1.15.0
 * Intel Core i7-5930K CPU @3.50GHz
 * Nvidia GeForce RTX 2080 Ti


The inference of the example data `example_data/brain/LR/cerebellum.tif` took 12s and 350s with and without GPU promotion in the tested platform.


## Install

1. Install python 3.6 
2. (Optional) If your computer has a CUDA-enabled GPU, install the CUDA and CUDNN of the proper version.
3. Download the DSP_Demo.zip and unpack it. The directory tree should be: 

```  
DSP-Demo   
    .
    ├── config.py
    ├── dataset.py
    ├── eval.py
    ├── model
    ├── requirements.txt
    ├── utils.py
    ├── example_data
        └── brain
        └── cell
```

4. Open the terminal in the DSP-Demo directory, install the dependencies using pip:

```
pip install -r requirements.txt
```

5. (Optional) If you have had the CUDA environment installed properly, run:

```
pip install tensorflow-gpu=1.15.0
```

The installation takes about 5 minutes in the tested platform. The time could be longer due to the network states.

## Usage

#### Inference

This toturial contains example data of mouse brains and cells (see example_data/):
```
.
├── example_data
    └── brain
        └── LR
            └── cerebellum_3.2x_bessel.tif                (3.2x bessel-sheet cerebellum data)
            └── vessel_3.2x_bessel_cross-sample.tif       (3.2x bessel-sheet brain vessel data, for cross-sample application )
            └── cortex_3.2x_confocal_cross-mode.tif       (3.2x confocal brain data of the cortex region , for cross-mode application )
        └── expected_outputs
    └── cell
        └── LR
            └── microtube_60x_bessel.tif                  (60x bessel-beam images of the microtube of the U2OS cells)
            └── ER_60x_bessel_cross-sample.tif            (60x bessel-beam images of the endoplasmic reticulum of the cell, for cross-sample application)
        └── expected_outputs

```

The expected outputs by the DSP-Net of each input LR can be found in the corresponding 'expected_outputs' directory (due to the size limit, only MIPs of expected outputs are provided. 

To run the DSP-inference, open the ternimal in the DSP-Demo directory and run :

```
python eval.py [options]
```

options:

* `--brain`    run DSP-inference on the brain data, using the bessel-brain-trained model. Three LR inputs under `example_data/brain/LR`  will be super-resolved.

* `--cell`     run DSP-inference on the cell data, using the microtube-trained model. Two LR inputs under `example_data/cell/LR` will be super-resolved.

*  `--cpu`     Use cpu instead of gpu for inference.  

The results will be saved at `example_data/brain/SR/` and `example_data/cell/SR/` respectively.

e.g., to run the DSP-super-resolution on the brain data using a brain-trained model with GPU and CUDA avaiable:

```
python eval.py --brain 
```

Otherwise, when GPUs are not available:

```
python eval.py --brain --cpu
```

TO run the inference on your own data, put them to the `example_data/brain/LR/` or `example_data/cell/LR/`, and make sure that:
1. The data is one or several 3-D tiff, with each dimension >= 50 pixels.
2. The data is in 8-bit.

Then run the command as above. 
