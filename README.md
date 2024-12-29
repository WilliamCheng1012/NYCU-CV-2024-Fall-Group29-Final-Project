# NYCU-CV-2024-Fall-Group29-Final-Project
## Single Image Reflection Removal (SIRR)


- **Team Member:** 312510232, 鄭惟謙 and 313510177, 賴冠維

## Table of Contents
- [Introduction](#introduction)
- [Implemented Papers](#implemented-papers)
- [Datasets and Model Weights](#datasets)
- [Usage](#usage)
- [Results](#results)
- [Analysis](#analysis)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
In real-world photography, unwanted optical reflections are common, such as glass reflections or lens flares. **SIRR** aims to remove these reflections from a single image.

The goal of this project is to:
- Evaluate and compare the effectiveness of different SIRR models.
- Test these models on various datasets using metrics such as **PSNR**, **SSIM**, and **LPIPS**.

## Implemented Papers
1. **[Single Image Reflection Separation with Perceptual Losses (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf)**
2. **[Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Single_Image_Reflection_Removal_Exploiting_Misaligned_Training_Data_and_Network_CVPR_2019_paper.pdf)**
3. **[Single Image Reflection Removal through Cascaded Refinement (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Single_Image_Reflection_Removal_Through_Cascaded_Refinement_CVPR_2020_paper.pdf)**

## Datasets and Model Weights

- **[Link of data and model weights](https://drive.google.com/drive/folders/1QJfaTHqoElUqW2GJLho2odpVPrXttEGD?usp=sharing)** 


## Usage
The code for each implemented paper is located in the `code` folder:
- `model_1` corresponds to Paper 1 ([Perceptual Reflection Removal](https://github.com/ceciliavision/perceptual-reflection-removal)).
- `model_2` corresponds to Paper 2 ([ERRNet](https://github.com/Vandermode/ERRNet)).
- `model_3` corresponds to Paper 3 ([IBCLN](https://github.com/JHL-HUST/IBCLN)).

Each folder contains a custom `run.ipynb` file that documents the execution process, including training and testing. Modify dataset paths manually within the `run.ipynb` files before running.

### Evaluating Metrics
The evaluation metrics for **PSNR**, **SSIM**, and **LPIPS** are integrated into the code for each model:
  - For **Model 1**, use `python eval.py` in `run.ipynb`
  - For **Model 2**, use `python eval.py` in `run.ipynb`
  - For **Model 3**, use `python eval.py` in `run.ipynb`


## Results
Below is a comparison of the three models based on testing results from the `CEILNetSynthetic` dataset.
![Results](img/result.png)

## Analysis

### Comparison of Metrics
The table below presents the detailed performance metrics (PSNR, SSIM, and LPIPS) for the implemented SIRR methods across various datasets. Metrics are reported for **Paper [1]**, **Paper [2]**, and **Paper [3]**.

| Dataset                | Index      | Model-I   | Model-II   | Model-III  |
|------------------------|------------|-----------|------------|------------|
| CEIL_Net Synthetic (100) | PSNR (↑)  | 28.785    | 32.051     | 28.710     |
|                        | SSIM (↑)  | 0.870     | 0.972      | 0.849      |
|                        | LPIPS (↓) | 0.1829    | 0.041      | 0.271      |
| SIR2 Objects (200)     | PSNR (↑)  | 29.608    | 29.929     | 29.840     |
|                        | SSIM (↑)  | 0.937     | 0.940      | 0.943      |
|                        | LPIPS (↓) | 0.109     | 0.098      | 0.109      |
| SIR2 Postcard (199)    | PSNR (↑)  | 29.108    | 29.129     | 29.441     |
|                        | SSIM (↑)  | 0.949     | 0.957      | 0.957      |
|                        | LPIPS (↓) | 0.164     | 0.159      | 0.168      |
| SIR2 Wild (55)         | PSNR (↑)  | 29.400    | 30.382     | 30.191     |
|                        | SSIM (↑)  | 0.939     | 0.931      | 0.953      |
|                        | LPIPS (↓) | 0.116     | 0.106      | 0.117      |



## Conclusion
Model-II achieved the best overall performance, with the highest PSNR and SSIM values and the lowest LPIPS across most datasets. This demonstrates its effectiveness in both synthetic and real-world scenarios. While Model-I and Model-III also performed well, Model-II stood out as the most reliable model in our experiments.

## References
1. Zhang, X., Ren Ng, and Q. Chen. "Single image reflection separation with perceptual losses." CVPR, 2018.
2. Wei, K., et al. "Single image reflection removal exploiting misaligned training data and network enhancements." CVPR, 2019.
3. Li, C., et al. "Single image reflection removal through cascaded refinement." CVPR, 2020.
