





insightface

官方github地址：[https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)

## 1. 安装MXNetGPU 版本

```shell
$ sudo pip install mxnet-cu90
```



## 2. 克隆insightface工程到本地

```shell
git clone --recursive https://github.com/deepinsight/insightface.git
```



## 3. 下载训练数据集（MS1M-Arcface）

下载地址为：https://github.com/deepinsight/insightface/wiki/Dataset-Zoo，然后把数据集解压到 insightface/datasets 目录下：

```
datasets/
├── faces_emore
│   ├── agedb_30.bin
│   ├── calfw.bin
│   ├── cfp_ff.bin
│   ├── cfp_fp.bin
│   ├── cplfw.bin
│   ├── lfw.bin
│   ├── property
│   ├── train.idx
│   ├── train.rec
│   └── vgg2_fp.bin
└── README.md
```

其中train.idx，train.rec，property用于训练，lfw.bin，cfp_fp.bin，agedb_30.bin用于验证。其中`train.idx` 和 `train.rec`分别是数据偏移索引和数据本身的文件，`property`代表数据集属性。



## 4. 训练

### 4.1 训练准备

- 进入 insightface/recognition 目录下，设置一下这两个参数，使用24线程加速，设置为在每个设备上跑。

```shell
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
```

- 拷贝一个配置选项文件并修改里面的配置参数：


```shell
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```

把 config.py 里面的 default.per_batch_size = 128 改为：default.per_batch_size = 64 ，要不然GPU会内存不足报错。



### 4.2 开始训练

官方提供了一下指令：

(1). Train ArcFace with LResNet100E-IR.

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore
```

(2). Train CosineFace with LResNet50E-IR.

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore
```

(3). Train Softmax with LMobileNet-GAP.

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss softmax --dataset emore
```

(4). Fine-turn the above Softmax model with Triplet loss.

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss triplet --lr 0.005 --pretrained ./models/m1-softmax-emore,1
```



根据自己的硬件，我自己的训练命令为：
training:
```bash
CUDA_VISIBLE_DEVICES='0' python -u train.py --network r100 --loss arcface --dataset emore
```

```bash
CUDA_VISIBLE_DEVICES='0' python -u train.py --network m1 --loss arcface --dataset emore
```
fine-tune:
```bash
CUDA_VISIBLE_DEVICES='0' python -u train.py --network m1 --loss softmax --lr 0.1 --pretrained ./models/m1-softmax-emore/model
```

会在 insightface/recognition/models 目录下生成训练好的模型。



- 报错修改

**报错一：**

> for i in xrange():
>
> ​	......

是python版本的不兼容问题，直接搜索代码里面的 xrange，全部替换为 range就好了。



**报错二：**

出现以下错误，和这篇博客所说的问题一样。https://blog.csdn.net/weixin_39502247/article/details/79933896

> Traceback (most recent call last):
>   File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 63, in apport_excepthook
>     from apport.fileutils import likely_packaged, get_recent_crashes
>   File "/usr/lib/python3/dist-packages/apport/__init__.py", line 5, in <module>
>     from apport.report import Report
>   File "/usr/lib/python3/dist-packages/apport/report.py", line 30, in <module>
>     import apport.fileutils
>   File "/usr/lib/python3/dist-packages/apport/fileutils.py", line 23, in <module>
>     from apport.packaging_impl import impl as packaging
>   File "/usr/lib/python3/dist-packages/apport/packaging_impl.py", line 23, in <module>
>     import apt
>   File "/usr/lib/python3/dist-packages/apt/__init__.py", line 23, in <module>
>     import apt_pkg
> ModuleNotFoundError: No module named 'apt_pkg'
>
> Original exception was:
> Traceback (most recent call last):
>   File "train.py", line 366, in <module>
>     main()
>   File "train.py", line 363, in main
>     train_net(args)
>   File "train.py", line 253, in train_net
>     data_set = verification.load_bin(path, image_size)
>   File "eval/verification.py", line 183, in load_bin
>     bins, issame_list = pickle.load(open(path, 'rb'))
> UnicodeDecodeError: 'ascii' codec can't decode byte 0xff in position 0: ordinal not in range(128)

解决办法，找到报错的位置，修改报错行：

```python
bins, issame_list = pickle.load(open(path, 'rb'))
# 改为：
bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
```



## 5. 验证模型

- 下载预训练模型，可以参考官方提供的地址进行下载，把模型放到 insightface/models/ 目录下，并解压。


- 修改 insightface/src/eval/verification.py ，找到185行并修改


```python
bins, issame_list = pickle.load(open(path, 'rb'))
# 改为：
bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
```

- 进行验证


```shell
$ python src/eval/verification.py --gpu 0 --data-dir datasets/faces_emore/ --model models/model-y1-test2/model,0 --target lfw,agedb_30
```

--model 用来指明model的路径

--target 指明所使用的验证数据集

结果我们可以看到使用moilenet的模型验证的结果如下：

> [lfw]XNorm: 11.128027
> [lfw]Accuracy: 0.00000+-0.00000
> [lfw]Accuracy-Flip: 0.99450+-0.00466
> Max of [lfw] is 0.99450

> [agedb_30]XNorm: 11.046304
> [agedb_30]Accuracy: 0.00000+-0.00000
> [agedb_30]Accuracy-Flip: 0.95717+-0.01188
> Max of [agedb_30] is 0.95717



## 6. 部署模型

```bash
$ cd deploy
$ python test.py --gpu 0 --model ../models/model-y1-test2/model,0 --ga-model ../gender-age/model/model,0
```

--model  表示model的路径

--ga-model  表示 gender&age，性别和年龄的模型路径



- 人脸检测


```bash
python find_faces_in_picture.py --gpu 0 --model ../models/model-y1-test2/model,0 --ga-model ../gender-age/model/model,0
```

![](https://github.com/liguiyuan/insightface/blob/master/deploy/face_detector.jpeg)



- 人脸识别


```bash
python facerec_from_webcam.py --gpu 0 --model ../models/model-y1-test2/model,0 --ga-model ../gender-age/model/model,0
```

------



**常见的报错**

**报错三：**

> Original exception was:
> Traceback (most recent call last):
>   File "deploy/test.py", line 20, in <module>
>     img = model.get_input(img)
>   File "/home/liguiyuan/deep_learning/project/insightface/deploy/face_model.py", line 71, in get_input
>     ret = self.detector.detect_face(face_img, det_type = self.args.det)
>   File "/home/liguiyuan/deep_learning/project/insightface/deploy/mtcnn_detector.py", line 323, in detect_face
>     height, width, _ = img.shape
> AttributeError: 'NoneType' object has no attribute 'shape'

是由于opencv读取不到图片造成的，修改一下cv2.imread 的路径位置即可。



**报错四：**

> Original exception was:
> Traceback (most recent call last):
> File "deploy/test.py", line 23, in <module>
>  gender, age = model.get_ga(img)
> File "/home/liguiyuan/deep_learning/project/insightface/deploy/face_model.py", line 99, in get_ga
>  self.ga_model.forward(db, is_train=False)
> AttributeError: 'NoneType' object has no attribute 'forward'

是因为model 路径没设对，找不到model。使用 --model 指定传入参数即可。



