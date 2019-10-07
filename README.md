# This is the code for the paper :Synergistic attention U‑Net for sublingual vein segmentation(AROB journal 2019: https://link.springer.com/article/10.1007/s10015-019-00547-9)


> This model is used to segment small target objects

In medical feild, U-Net is pretty popular in segmentation with few dataset. However, U-Net is not sensitve to the small object segmentation.
Our contribution is to modify U-Net slightly and make it more sensitive to small target segmentation. The coder is implemented with tensorflow.

This repository contains 5 folders:

## Table of Contents

- [MakeTFRecords](#This is used for generating tfrecords for the training/validation/test dataset)
- [Model_factory_AROB_J](#This folder contains main running script and the generated tfrecrods should be copied into this folder)
- [Preprocessing](# This is for the preprocessing when reading tfrecords)

- [Read_TF_records](# This is for reading tfrecords, return the dataset iterator)
- [Utils](# Some utilization functions for running the main script)


## Paper citation
If you like our work and our work could inspire your work a little, please ref with bib-tex
> @article{yang2019synergistic,
  title={Synergistic attention U-Net for sublingual vein segmentation},
  author={Yang, Tingxiao and Yoshimura, Yuichiro and Morita, Akira and Namiki, Takao and Nakaguchi, Toshiya},
  journal={Artificial Life and Robotics},
  pages={1--10},
  year={2019},
  publisher={Springer}
}

## Our team and lab homepage:
http://nlab.tms.chiba-u.jp
