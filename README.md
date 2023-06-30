# HabitatDyn Dataset: Dynamic Object Detection to Kinematics Estimation.

***Zhengcheng Shen, Yi Gao, Linh Kästner, Jens Lambrecht***

- [HabitatDyn Dataset: Dynamic Object Detection to Kinematics Estimation.](#habitatdyn-dataset-dynamic-object-detection-to-kinematics-estimation)
  - [Update](#update)
  - [Abstract](#abstract)
  - [Video examples](#video-examples)
  - [Overview](#overview)
  - [Annotations](#annotations)
    - [Depth](#depth)
  - [Download](#download)
  - [Evaluate HabitatDyn dataset using 3DC-Seg and CIS](#evaluate-habitatdyn-dataset-using-3dc-seg-and-cis)
    - [3DC-Seg](#3dc-seg)
    - [CIS](#cis)
  - [Evaluate HabitatDyn dataset using distance estimation](#evaluate-habitatdyn-dataset-using-distance-estimation)
  - [Citation](#citation)
  - [Model downloaded from the internet, Bounding box infos \[Width, Hight, Length\]](#model-downloaded-from-the-internet-bounding-box-infos-width-hight-length)
  - [Disclaimer](#disclaimer)
  - [Acknowledgement](#acknowledgement)

## Update
- May 26. 2023: We provide a demo for object localization in the repo: https://github.com/ignc-research/DODL-SALOC It use the HabitatDyn Dataset.

todo:

![data](https://user-images.githubusercontent.com/83227264/230888881-410c0266-9256-4add-a42a-d38bed991be9.jpg)

## Abstract
> We propose a new dataset, HabitatDyn, which includes synthetic RGB videos, semantic labels, and depth information, as well as kinetics information. The dataset features 30 scenes from the perspective of a mobile robot with a moving camera, and contains six different types of moving objects with varying velocities.

## Video examples



https://user-images.githubusercontent.com/83227264/232990022-b49e04ef-3964-46c6-9b13-3466279edb6e.mp4



## Overview

We use of the [Facebook AI Habitat simulation platform](https://github.com/facebookresearch/habitat-sim) as a basic set-up with embodied agents in 3D
virtual environments for generating desired data. We made a configuration API on top of the AI habitat simulator to performe dataset generate our HabitatDyn dataset, including camera adjustment, annotation recording, random pathfinding abd embodied agents navigation in the simulated scene.

HabitatDyn contains 1590 high-quality videos for 30 meticulously curated scenes from [HM3D](https://aihabitat.org/datasets/hm3d/), citation down below. These scenes feature free, randomly placed object models sourced from the Internet. To add a touch of realism and dynamism, 3 or 6 moving object instances from each model are randomly dropped into the scene, and a robot agent equipped with a camera is also present to capture the scene in detail (camera specifications can be found in the dataset). The videos come with RGB, depth, and semantic annotations, ensuring that researchers have access to a wealth of information for their projects.

Additionally, the objects in each scene have been carefully programmed to move at varying speeds, with 3 levels of movement speed available (1 m/s, 2 m/s, and 3 m/s). To add even more depth to the dataset, each scene has been split into "incl_static" and "excl_static" versions. In the "incl_static" version, 10 static extra object models are randomly placed in the scene, while in the "excl_static" version, only the original objects from the scene and moving instances are present.

Furthermore, the same settings for 3 randomly chosen instances from the 6 models are also generated, making this dataset a comprehensive and versatile resource for researchers in various fields.

In the entirety of the video recording, data pertaining to the physical properties such as the specifications of the camera, as well as the positional information of both the camera itself and any pedestrians or moving objects present within the recording are documented.

## Annotations

Overview of annotations format

| - | images | semantic mask | depth | camera infos | pedstrains/moving object infos | semanticID to model name | videoID to descriptive name mapping | video |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| format | .jpg | .png | .png | .npy | .npy | .json | .txt| .mp4 |

\
**General settings**


| Camera HFov   | 90 degree         |
|---------------|-------------------|
| Sensor Height | 1.25 meter        |
| Frequency     | 24hz              |
| Max, speed    | 1 m/s, 2m/s, 3m/s |
| Max, depth    | 10 meter.         |
| Resolution    | 640*480.          |
| Object ID     | 1 - 6.            |

### Depth


https://user-images.githubusercontent.com/83227264/230932486-90f0ff66-409c-4d55-a772-6c0d4aad5d0b.mp4


## Download

A subset for 2 scenes with only dynamic moving objects can be downloaded from the following link
* [subset for 2 scenes](https://tubcloud.tu-berlin.de/s/KikfymcmENWSdjk) - 108 videos (12 GB)

Please email Zhengcheng Shen(zhengcheng.shen@tu-berlin.de) to obtain the link for downloading the whole dataset.

Email Format

> *Name :

> *Organization :

> *Purpose of dataset :


## Evaluate HabitatDyn dataset using [3DC-Seg](https://github.com/sabarim/3DC-Seg) and [CIS](https://github.com/antonilo/unsupervised_detection)

### [3DC-Seg](https://github.com/sabarim/3DC-Seg)

Clone 3DC-Seg repository with given [instruction](https://github.com/sabarim/3DC-Seg#setup) and download the checkpoints with given [link](https://omnomnom.vision.rwth-aachen.de/data/3DC-Seg/models/bmvc_final.pth).  Run inference with your own `.yaml` file that contains the path to HabitatDyn dataset, and evaluate with:

```
python main.py -c run_configs/your_own.yaml --task eval --wts <path>/bmvc_final.pth --save_folder output/dynamic

```

### [CIS](https://github.com/antonilo/unsupervised_detection)
Cole CIS repository with given [instruction](https://github.com/antonilo/unsupervised_detection#quick-test-the-inference-without-any-preparations), a example to download pretrained weight can be found in `/unsupervised_detection/scripts/test_DAVIS2016_raw.sh`, then make your own `.sh` file which contains the path to HabitatDyn dataset and evaluate with:

```
bash ./scripts/your_own.sh
```
We also provide a simple script to do moving object segmentation metric evaluation for those models and the provided HabitatDyn ground truth, as a usage exaple of HabitatDyn. To do so, the model prediction output files should have same structure as the HabitatDyn ground truth.

```
python metric_cal.py --gt_data=/path/to/habitatDyn/gt --pred_data=/path/to/segmentation/output --flag=*
```
the `--flag` tag set which kind/categories of habitatDyn dataset.
```
0: 'calculate All',
1: 'calculate Single class',
2: 'calculate Multi class',
3: 'calculate Speed 1',
4: 'calculate Speed 2',
5: 'calculate Speed 3',
6: 'calculate Human classes',
7: 'calculate toy car/robot classes',
8: 'calculate dog/cat classe'
```


## Evaluate HabitatDyn dataset using distance estimation

Extract pose/location info of Top-down view using mask generated by moving object detection model:
```python
python dist_cal.py --habitatDyn_data=/path/to/habitatDyn --mask_data=/path/to/dynamic
```

Then using the extracted info to evaluate the distance estimation metric, you can change the range as you want(HabitatDyn has max range 10).
```python
python dist_metric_cal.py --data_path=/path/to/dist/cal/resuls/use_pre --range_start=0 --range_end=1
```


## Citation

Please cite [HabitatDyn] and all the related works in your publications if it helps your research:

```latex
@misc{shen2023habitatdyn,
      title={HabitatDyn Dataset: Dynamic Object Detection to Kinematics Estimation},
      author={Zhengcheng Shen and Yi Gao and Linh Kästner and Jens Lambrecht},
      year={2023},
      eprint={2304.10854},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{szot2021habitat,
  title     =     {Habitat 2.0: Training Home Assistants to Rearrange their Habitat},
  author    =     {Andrew Szot and Alex Clegg and Eric Undersander and Erik Wijmans and Yili Zhao and John Turner and Noah Maestre and Mustafa Mukadam and Devendra Chaplot and Oleksandr Maksymets and Aaron Gokaslan and Vladimir Vondrus and Sameer Dharur and Franziska Meier and Wojciech Galuba and Angel Chang and Zsolt Kira and Vladlen Koltun and Jitendra Malik and Manolis Savva and Dhruv Batra},
  booktitle =     {Advances in Neural Information Processing Systems (NeurIPS)},
  year      =     {2021}
}
@inproceedings{ramakrishnan2021hm3d,
  title={Habitat-Matterport 3D Dataset ({HM}3D): 1000 Large-scale 3D Environments for Embodied {AI}},
  author={Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Erik Wijmans and Oleksandr Maksymets and Alexander Clegg and John M Turner and Eric Undersander and Wojciech Galuba and Andrew Westbury and Angel X Chang and Manolis Savva and Yili Zhao and Dhruv Batra},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021},
  url={https://openreview.net/forum?id=-v4OuqNs5P}
}

@inproceedings{yang_loquercio_2019,
  title={Unsupervised Moving Object Detection via Contextual Information Separation},
  author={Yang, Yanchao and Loquercio, Antonio and Scaramuzza, Davide and Soatto, Stefano},
  booktitle = {Conference on Computer Vision and Pattern Recognition {(CVPR)}}
  year={2019}
}

```

## Model downloaded from the internet, Bounding box infos [Width, Hight, Length]

[shiba](https://sketchfab.com/3d-models/shiba-faef9fe5ace445e7b2989d1c1ece361c): [0.360, 0.276, 0.354] , [miniature_cat](https://sketchfab.com/3d-models/miniature-cat-7aabffe566ef462db6d1cd6a6dd46345)[0.117, 0.383, 0.435], [FerBibliotecario](https://sketchfab.com/3d-models/ferbibliotecario-ff3847432b914969aeba66bcc2adc657):[0.534, 1.288, 0.243], [angry_girl](https://sketchfab.com/3d-models/redhead-rock-girl-1a056adab45f462fa75863701439356f): [0.486, 1.134, 0.318], [robot_2020](https://sketchfab.com/3d-models/robot-2020-c0dadae4d1884bf48615f1ee301fe7e6)[0.684, 0.620, 0.418], [toy_car](https://sketchfab.com/3d-models/toy-car-9cf99655e7424770b79ba702fe83e5c1): [0.060, 0.077, 0.156]


## Disclaimer

Please note that this dataset contains 3D models that were freely downloaded from the internet and the open-source simulator Habitat-sim. While we have made every effort to ensure the quality and accuracy of the data, we cannot guarantee the authenticity or reliability of the 3D models or the simulator used to create this dataset.

Furthermore, the dataset is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the dataset or the use or other dealings in the dataset.

HabitatDyn is free for academic, non-commercial research. It is important to exercise caution and undertake appropriate due diligence before using this dataset for any research. Users should also acknowledge the sources of the 3D models and the simulator in any publications or presentations that make use of this dataset.


## Acknowledgement

```latex
@inproceedings{szot2021habitat,
  title     =     {Habitat 2.0: Training Home Assistants to Rearrange their Habitat},
  author    =     {Andrew Szot and Alex Clegg and Eric Undersander and Erik Wijmans and Yili Zhao and John Turner and Noah Maestre and Mustafa Mukadam and Devendra Chaplot and Oleksandr Maksymets and Aaron Gokaslan and Vladimir Vondrus and Sameer Dharur and Franziska Meier and Wojciech Galuba and Angel Chang and Zsolt Kira and Vladlen Koltun and Jitendra Malik and Manolis Savva and Dhruv Batra},
  booktitle =     {Advances in Neural Information Processing Systems (NeurIPS)},
  year      =     {2021}
}
```

```latex
@inproceedings{ramakrishnan2021hm3d,
  title={Habitat-Matterport 3D Dataset ({HM}3D): 1000 Large-scale 3D Environments for Embodied {AI}},
  author={Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Erik Wijmans and Oleksandr Maksymets and Alexander Clegg and John M Turner and Eric Undersander and Wojciech Galuba and Andrew Westbury and Angel X Chang and Manolis Savva and Yili Zhao and Dhruv Batra},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021},
  url={https://openreview.net/forum?id=-v4OuqNs5P}
}
```

```latex
@inproceedings{yang_loquercio_2019,
  title={Unsupervised Moving Object Detection via Contextual Information Separation},
  author={Yang, Yanchao and Loquercio, Antonio and Scaramuzza, Davide and Soatto, Stefano},
  booktitle = {Conference on Computer Vision and Pattern Recognition {(CVPR)}}
  year={2019}
}
```

```latex
@misc{mahadevan2020making,
      title={Making a Case for 3D Convolutions for Object Segmentation in Videos},
      author={Sabarinath Mahadevan and Ali Athar and Aljoša Ošep and Sebastian Hennen and Laura Leal-Taixé and Bastian Leibe},
      year={2020},
      eprint={2008.11516},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
