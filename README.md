# HSIDNCNN

A simple hyperspectral image denoising framework based on DnCNN, implemented on top of the KAIR codebase.

## Overview

This project modifies DnCNN for **hyperspectral image denoising** by using **spectral neighbors** instead of denoising each band independently.

For each training sample:

- The **target band** is selected from a clean hyperspectral cube
- The network input is a **5-band stack**:
  - two neighboring bands on the left
  - the target band
  - two neighboring bands on the right
- The network output is the **denoised target band only**

This setup is designed to exploit **spectral correlation** between adjacent bands.

In addition, each input band is corrupted with **Gaussian noise that can vary from band to band**, making the problem more realistic than fixed-noise single-image denoising.

## Method

### Input and output

- **Input:** 5 spectral bands  
  `[b-2, b-1, b, b+1, b+2]`
- **Output:** clean center band  
  `[b]`

### Noise model

Each input band is corrupted independently using additive Gaussian noise:

- noise level is sampled separately for each band
- noise level can vary across the hyperspectral cube
- validation can be made reproducible with a fixed seed

### Network

The model is a modified DnCNN that performs **residual learning for the center band only**:

- input channels: `5`
- output channels: `1`
- the network predicts the residual/noise of the center band
- the final output is:

```text
denoised_center = noisy_center - predicted_residual