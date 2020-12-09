# Pytorch-FAU

The Pytorch implementation of **Facial Action Unit Intensity Estimation**. 

## Environment

- Ubuntu 18.04.4
- Python 3.7
- PyTorch 1.3.0
- Torchvision
- Python-OpenCV

***Datasets***

For data preparation, please make a request for the [BP4D database](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and the [DISFA database](http://mohammadmahoor.com/disfa/). 

***Usage***

The pre-trained model can be obtained from the [link](https://drive.google.com/file/d/15cJtFEkrOrbt5FfZOxnWulQhaOKfaLtN/view?usp=sharing). Please download it under your own path. You can change default path by modifying `--model_path`.

- `run_demo.py`: visualizes the predicted AU heatmaps and intensities for the example images (`data/*.jpg`).

```python 
cd Pytorch-FAU/
python run_demo.py
```

The full code will be available soon.

## Citation

    @inproceedings{fan2020fau,
        title = {Facial Action Unit Intensity Estimation via Semantic 
        Correspondence Learning with Dynamic Graph Convolution},
        author = {Fan, Yingruo and Lam, Jacqueline and Li, Victor},
        booktitle = {Thirty-Fourth AAAI Conference on Artificial Intelligence},
        year={2020}
    }

## Acknowledgement

The code partially refers open-sourced [Action-Units-Heatmaps](https://github.com/ESanchezLozano/Action-Units-Heatmaps). Thanks to them for the great work.


