## Demographic Bias of Expert-Level Vision-Language Foundation Models in Medical Imaging

[![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/YyzHarry/vlm-fairness/blob/main/LICENSE)
![](https://img.shields.io/github/stars/YyzHarry/vlm-fairness)
![](https://img.shields.io/github/forks/YyzHarry/vlm-fairness)
![](https://visitor-badge.laobi.icu/badge?page_id=YyzHarry.vlm-fairness&right_color=%23FFA500)

[[Paper]()] (Coming soon.)

**Summary**: Coming soon.

### Dataset

To download all the datasets used in this study, please follow instructions in [DataSources.md](./DataSources.md).

As the original image files are often high resolution, we cache the images as downsampled copies to speed training up for certain datasets. To do so, run
```bash
python -m scripts.cache_cxr --data_path <data_path> --dataset <dataset>
``` 
where datasets can be `mimic` or `vindr`. This process is required for `vindr`, and is optional for the remaining datasets.

### Model Checkpoints
This repo uses [CheXzero](https://www.nature.com/articles/s41551-022-00936-9) as a driving example for vision-language models.
Download [model checkpoints](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno?usp=sharing) of CheXzero and save them in the `./checkpoints` directory.


### Zero-Shot Evaluation

```bash
python -m zero_shot \
       --dataset <dataset> \
       --split <split> \
       --template <name_of_your_prompt_template> \
       --data_dir <data_path> \
       --model_dir <model_path> \
       --predictions_dir <output_path>
```

### Acknowledgements
This code is partly based on the open-source implementations from [CheXzero](https://github.com/rajpurkarlab/CheXzero) and [SubpopBench](https://github.com/YyzHarry/SubpopBench).

### Citation
Coming soon.

### Contact
If you have any questions, feel free to contact us through email (yuzhe@mit.edu) or GitHub issues. Enjoy!
