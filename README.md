# Visual Region Aggregation and Dual-level Collaboration
* Note: our repository is mainly based on [zhangxuying1004/RSTNet](https://github.com/zhangxuying1004/RSTNet), and we directly reused some backbone model files and dataloader files(except for the Flickr30k dataloader)



## Environment setup
Clone the repository and create the `m2release` conda environment using the `environment.yml` file:
```
conda env create -f environment.yml
conda activate m2release
```

Then download spacy data by executing the following command:
```
python -m spacy download en
```

Note: Python 3.6 is required to run our code. 


## Data preparation
Our model follows end-to-end style,so no matter for MSCOCO or Flicker30k,you don't need to download extra offline features.

### For MSCOCO raw data downloads
[Annotation](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing)

[Raw images](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9)

### For Flicker30k raw data downloads
You can find raw data of Flickr30k at [Flick30k webpage](https://shannon.cs.illinois.edu/DenotationGraph/),after filling some tables,you can get an url for Flickr30k's pictures and annotations.

For simplicity,we reuse the split settings and corresponding file for image id and caption id by [Flick30k entities dataset](https://github.com/BryanPlummer/flickr30k_entities) 

For reusability of COCO interface,we recommand you to split all images to *Train/Val/Test* folders.

### For visual backbone checkpoint

We use [Swin-B 224](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) as our baseline backbone,and carried out ablation studies by using [Swin-B 384](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)/[Swin-L 224](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth)/[Swin-L 384](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth).

## Training procedure
Run  `python train_transformer.py` in sequence using the following arguments:

| Argument              | Possible values                                                 |
| --------------------- | --------------------------------------------------------------- |
| `--exp_name`          | Experiment name                                                 |
| `--batch_size`        | Batch size (default: 10)                                        |
| `--workers`           | Number of workers, accelerate model training in the xe stage.   |
| `--head`              | Number of heads (default: 8)                                    |
| `--resume_last`       | If used, the training will be resumed from the last checkpoint. |
| `--resume_best`       | If used, the training will be resumed from the best checkpoint. |
| `--features_path`     | Path to visual features file (h5py)                             |
| `--annotation_folder` | Path to m2_annotations                                          |


to train our  model with the parameters used in our experiments, use
```
python train_transformer.py --exp_name VLAD_SwinCaption --batch_size 50 --m 40 --head 8 --features_path /path/to/features --annotation_folder /path/to/annotations
```

## Evaluation
Run `python test_transformer.py` to evaluate the rstnet or `python test_language.py` to evaluate the language model using the following arguments:

| Argument              | Possible values                     |
| --------------------- | ----------------------------------- |
| `--batch_size`        | Batch size (default: 10)            |
| `--workers`           | Number of workers (default: 0)      |
| `--features_path`     | Path to visual features file (h5py) |
| `--annotation_folder` | Path to m2_annotations              |

Note that, you can also download the pretrained model file [VLADSwinTransformer.pth](https://pan.baidu.com/s/1NsjNDCzNfTUwgXkotKNCrA) (code 7sve)and place it in `saved_transformer_models` folder o reproduce the our reported results.  


#### Expected output
Under `output_logs/`, you may also find the expected output of the evaluation code.


<p align="center">
  <img src="images/visualness.png" alt="Sample Results" width="670"/>
</p>


#### References
[1] Cornia, M., Stefanini, M., Baraldi, L., & Cucchiara, R. (2020). Meshed-memory transformer for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  
[2] Jiang, H., Misra, I., Rohrbach, M., Learned-Miller, E., & Chen, X. (2020). In defense of grid features for visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.   


#### Acknowledgements
Thanks Cornia _et.al_ for their open source code [M2 transformer](https://github.com/aimagelab/meshed-memory-transformer), on which our implements are based.  
Thanks Jiang _et.al_ for the significant discovery in visual representation [2], which has given us a lot of inspiration.
