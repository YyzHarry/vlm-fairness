## Demographic Bias of Expert-Level Vision-Language Foundation Models in Medical Imaging

[![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/YyzHarry/vlm-fairness/blob/main/LICENSE)
![](https://img.shields.io/github/stars/YyzHarry/vlm-fairness)
![](https://img.shields.io/github/forks/YyzHarry/vlm-fairness)
![](https://visitor-badge.laobi.icu/badge?page_id=YyzHarry.vlm-fairness&right_color=%23FFA500)

[[Paper](https://www.science.org/doi/10.1126/sciadv.adq0305)] [[Science News](https://www.science.org/content/article/ai-models-scanning-chest-x-rays-miss-disease-black-female-patients)]

**Summary**: Advances in artificial intelligence (AI) have achieved expert-level performance in medical imaging applications. Notably, self-supervised vision-language foundation models can detect a broad spectrum of pathologies without relying on explicit training annotations. However, it is crucial to ensure that these AI models do not mirror or amplify human biases, thereby disadvantaging historically marginalized groups such as females or Black patients. The manifestation of such biases could systematically delay essential medical care for certain patient subgroups. In this study, we investigate the algorithmic fairness of state-of-the-art vision-language foundation models in chest X-ray diagnosis across five globally-sourced datasets. Our findings reveal that compared to board-certified radiologists, these foundation models consistently underdiagnose marginalized groups, with even higher rates seen in intersectional subgroups, such as Black female patients. Such demographic biases present over a wide range of pathologies and demographic attributes. Further analysis of the model embedding uncovers its significant encoding of demographic information. Deploying AI systems with these biases in medical imaging can intensify pre-existing care disparities, posing potential challenges to equitable healthcare access and raising ethical questions about their clinical application.

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
If you find this code or idea useful, please cite our work:

```bibtex
@article{yang2025demographic,
  title = {Demographic bias of expert-level vision-language foundation models in medical imaging},
  author = {Yuzhe Yang and Yujia Liu and Xin Liu and Avanti Gulhane and Domenico Mastrodicasa and Wei Wu and Edward J. Wang and Dushyant Sahani and Shwetak Patel},
  journal = {Science Advances},
  volume = {11},
  number = {13},
  pages = {eadq0305},
  year = {2025},
  doi = {10.1126/sciadv.adq0305}
}
```

### Contact
If you have any questions, feel free to contact us through email (yuzheyangpku@gmail.com) or GitHub issues. Enjoy!
