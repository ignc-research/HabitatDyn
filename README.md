# HabitatDyn Dataset: Dynamic Object Detection to Kinematics Estimation

***Zhengcheng Shen, Yi Gao, Linh Kästner, Jens Lambrecht***

![data](https://user-images.githubusercontent.com/83227264/230888881-410c0266-9256-4add-a42a-d38bed991be9.jpg)

## Abstract
> We propose a new dataset, HabitatDyn, which includes synthetic RGB videos, semantic labels, and depth information, as well as kinetics information. The dataset features 30 scenes from the perspective of a mobile robot with a moving camera, and contains six different types of moving objects with varying velocities.

## Overview

We use of the [Facebook AI Habitat simulation platform](https://github.com/facebookresearch/habitat-sim) as a basic set-up with embodied agents in 3D
virtual environments for generating desired data. We made a configuration API on top of the AI habitat simulator to performe dataset generate our HabitatDyn dataset, including camera adjustment, annotation recording, random pathfinding abd embodied agents navigation in the simulated scene.

HabitatDyn contains 1590 high-quality videos for 30 meticulously curated scenes from [HM3D](https://aihabitat.org/datasets/hm3d/, citation down below). These scenes feature free, randomly placed object models sourced from the Internet. To add a touch of realism and dynamism, 3 or 6 moving object instances from each model are randomly dropped into the scene, and a robot agent equipped with a camera is also present to capture the scene in detail (camera specifications can be found in the dataset). The videos come with RGB, depth, and semantic annotations, ensuring that researchers have access to a wealth of information for their projects.

Additionally, the objects in each scene have been carefully programmed to move at varying speeds, with 3 levels of movement speed available (1 m/s, 2 m/s, and 3 m/s). To add even more depth to the dataset, each scene has been split into "incl_static" and "excl_static" versions. In the "incl_static" version, 10 static extra object models are randomly placed in the scene, while in the "excl_static" version, only the original objects from the scene and moving instances are present.

Furthermore, the same settings for 3 randomly chosen instances from the 6 models are also generated, making this dataset a comprehensive and versatile resource for researchers in various fields.

In the entirety of the video recording, data pertaining to the physical properties such as the specifications of the camera, as well as the positional information of both the camera itself and any pedestrians or moving objects present within the recording are documented.

## Annotations 

Overview of annotations format

| - | images | semantic mask | depth | camera infos | pedstrains/moving object infos | semanticID to model name | videoID to descriptive name mapping | video |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| format | .jpg | .png | .png | .npy | .npy | .json | .txt| .mp4 |

General settings

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

The train set, the test set and a sample of the CeyMo road marking dataset can be downloaded from the following Google Drive links.
A subset for 2 scenes with only dynamic moving objects can be downloaded from the following link
* [subset for 2 scenes](https://tubcloud.tu-berlin.de/s/KikfymcmENWSdjk) - 108 videos (12 GB)

Please email Zhengcheng Shen(zhengcheng.shen@tu-berlin.de) to obtain the link for downloading the whole dataset.

Email Format

> *Name :

> *Organization :

> *Purpose of dataset :


## Citation
Please cite [HabitatDyn](TODO) dataset in your publications if it helps your research:

```latex
TODO
```

## Model downloaded from the internet
[shiba](https://sketchfab.com/3d-models/shiba-faef9fe5ace445e7b2989d1c1ece361c), [miniature_cat](https://sketchfab.com/3d-models/miniature-cat-7aabffe566ef462db6d1cd6a6dd46345), [FerBibliotecario](https://sketchfab.com/3d-models/ferbibliotecario-ff3847432b914969aeba66bcc2adc657), [angry_girl](https://sketchfab.com/3d-models/redhead-rock-girl-1a056adab45f462fa75863701439356f), [robot_2020](https://sketchfab.com/3d-models/robot-2020-c0dadae4d1884bf48615f1ee301fe7e6), [toy_car](https://sketchfab.com/3d-models/toy-car-9cf99655e7424770b79ba702fe83e5c1)

## Disclaimer

Please note that this dataset contains 3D models that were freely downloaded from the internet and the open-source simulator Habitat-sim. While we have made every effort to ensure the quality and accuracy of the data, we cannot guarantee the authenticity or reliability of the 3D models or the simulator used to create this dataset.

Furthermore, the dataset is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the dataset or the use or other dealings in the dataset.

HabitatDyn is free for academic, non-commercial research. It is important to exercise caution and undertake appropriate due diligence before using this dataset for any research. Users should also acknowledge the sources of the 3D models and the simulator in any publications or presentations that make use of this dataset.
