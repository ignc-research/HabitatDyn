# HabitatDyn Dataset: Dynamic Object Detection to Kinematics Estimation

***Zhengcheng Shen, Yi Gao, Linh Kästner, Jens Lambrecht***

![data](https://user-images.githubusercontent.com/83227264/230888881-410c0266-9256-4add-a42a-d38bed991be9.jpg)

 We propose a new dataset, HabitatDyn, which includes synthetic RGB videos, semantic labels, and depth information, as well as kinetics information. The dataset features 30 scenes from the perspective of a mobile robot with a moving camera, and contains six different types of moving objects with varying velocities. 

## Abstract
>The aim of the HabitatDyn dataset is to provide a new resource for researchers and developers in the field of mobile robotics that can aid in the creation and advancement of robots with advanced recognition capabilities. The dataset was created to address the limitations of existing image or video processing datasets, which usually do not accurately depict observations from a moving robot and do not contain the kinematics information necessary for robotic tasks. By providing synthetic data that is cost-effective to create and offers greater flexibility for adapting to various applications, the HabitatDyn dataset can be used to train and evaluate algorithms for mobile robots that can navigate and interact with their environments more effectively. Through the use of this dataset, we hope to foster further advancements in the field of mobile robotics and contribute to the development of more capable and intelligent robots.


### Overview

We use of the [Facebook AI Habitat simulation platform](https://github.com/facebookresearch/habitat-sim) as a basic set-up with embodied agents in 3D
virtual environments for generating desired data. We made a configuration API on top of the AI habitat simulator to performe dataset generate our HabitatDyn dataset, including camera adjustment, annotation recording, random pathfinding abd embodied agents navigation in the simulated scene.

HabitatDyn contains 1590 high-quality videos for 30 meticulously curated scenes from [HM3D](https://aihabitat.org/datasets/hm3d/, citation down below). These scenes feature free, randomly placed object models sourced from the Internet. To add a touch of realism and dynamism, 3 or 6 moving object instances from each model are randomly dropped into the scene, and a robot agent equipped with a camera is also present to capture the scene in detail (camera specifications can be found in the dataset). The videos come with RGB, depth, and semantic annotations, ensuring that researchers have access to a wealth of information for their projects.

Additionally, the objects in each scene have been carefully programmed to move at varying speeds, with 3 levels of movement speed available (1 m/s, 2 m/s, and 3 m/s). To add even more depth to the dataset, each scene has been split into "incl_static" and "excl_static" versions. In the "incl_static" version, 10 static extra object models are randomly placed in the scene, while in the "excl_static" version, only the original objects from the scene and moving instances are present.

Furthermore, the same settings for 3 randomly chosen instances from the 6 models are also generated, making this dataset a comprehensive and versatile resource for researchers in various fields.

In the entirety of the video recording, data pertaining to the physical properties such as the specifications of the camera, as well as the positional information of both the camera itself and any pedestrians or moving objects present within the recording are documented.

### Annotations 

overview of annotations format

| - | images | semantic mask | depth | camera infos | pedstrains/moving object infos | semanticID to model name | videoID to descriptive name mapping | video |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| format | .jpg | .png | .png | .npy | .npy | .json | .txt| .mp4 |

The mp4 videos has a frame rate of 24 fps.


### Depth


https://user-images.githubusercontent.com/83227264/230932486-90f0ff66-409c-4d55-a772-6c0d4aad5d0b.mp4


## Download

The train set, the test set and a sample of the CeyMo road marking dataset can be downloaded from the following Google Drive links.
A subset for 2 scenes with only dynamic moving objects can be downloaded from the following link
* [subset for 2 scenes](https://tubcloud.tu-berlin.de/s/KikfymcmENWSdjk) - 108 videos (12 GB)

Please email Zhengcheng Shen(zhengcheng.shen@tu-berlin.de) to obtain the link for downloading the whole dataset.

Email Format

*Name :

*Organization :

*Purpose of dataset :


### citation

```latex
@inproceedings{szot2021habitat,
  title     =     {Habitat 2.0: Training Home Assistants to Rearrange their Habitat},
  author    =     {Andrew Szot and Alex Clegg and Eric Undersander and Erik Wijmans and Yili Zhao and John Turner and Noah Maestre and Mustafa Mukadam and Devendra Chaplot and Oleksandr Maksymets and Aaron Gokaslan and Vladimir Vondrus and Sameer Dharur and Franziska Meier and Wojciech Galuba and Angel Chang and Zsolt Kira and Vladlen Koltun and Jitendra Malik and Manolis Savva and Dhruv Batra},
  booktitle =     {Advances in Neural Information Processing Systems (NeurIPS)},
  year      =     {2021}
}

@inproceedings{habitat19iccv,
  title     =     {Habitat: {A} {P}latform for {E}mbodied {AI} {R}esearch},
  author    =     {Manolis Savva and Abhishek Kadian and Oleksandr Maksymets and Yili Zhao and Erik Wijmans and Bhavana Jain and Julian Straub and Jia Liu and Vladlen Koltun and Jitendra Malik and Devi Parikh and Dhruv Batra},
  booktitle =     {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      =     {2019}
}
```

```latex
@inproceedings{ramakrishnan2021hm3d,
  title={Habitat-Matterport 3D Dataset ({HM}3D): 1000 Large-scale 3D Environments for Embodied {AI}},
  author={Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Erik Wijmans and Oleksandr Maksymets and Alexander Clegg and John M Turner and Eric Undersander and Wojciech Galuba and Andrew Westbury and Angel X Chang and Manolis Savva and Yili Zhao and Dhruv Batra},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2021},
  url={https://arxiv.org/abs/2109.08238}
}
```
