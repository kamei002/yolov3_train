# yolov3_train

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

Make yolov3 datasets with yolov3

 [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python3 keras-yolo3/convert.py keras-yolo3/yolov3.cfg yolov3.weights yolo.h5
python3 make_datasets.py --input [video_path]
python3 make_datasets.py --input [video_dir]/\*.mp4
```

---
## Training Result

<table style="width:100%">
  <th style="width:50%">before</th>
  <th style="width:50%">after</th>
  <tr>
    <td>![before](https://cdn-ak.f.st-hatena.com/images/fotolife/a/ascreit/20180918/20180918210051.gif)</td>
    <td>![after](https://cdn-ak.f.st-hatena.com/images/fotolife/a/ascreit/20180918/20180918205934.gif)</td>
  </tr>

</table>

Improved? Or deterioration?
