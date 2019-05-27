# 论文阅读笔记(Learning RoI Transformer for Detecting Oriented Objects in Aerial Images)

## Abstract

> Object detection in aerial images is an active yet challenging task in computer vision because of
> the bird-view perspective, the highly complex backgrounds, and the variant appearances of objects.
> Especially when detecting densely packed objects in aerial images, methods relying on horizontal proposals for common object detection often introduce mismatches between the Region of Interests
> (RoIs) and objects. This leads to the common misalignment between the final object classification confidence and localization accuracy. Although rotated anchors have been used to tackle this
> problem, the design of them always multiplies the number of anchors and dramatically increases
> the computational complexity. In this paper, we propose a **RoI Transformer** to address these
> problems. More precisely, to improve the quality of region proposals, we ==first== designed a Rotated RoI (RRoI) learner to transform a Horizontal Region of Interest (HRoI) into a Rotated Region of Interest (RRoI). Based on the RRoIs, we ==then== proposed a Rotated Position Sensitive RoI Align (RPS-RoI-Align) module to extract rotation-invariant features from them for boosting subsequent classification and regression. Our RoI Transformer is with light weight and can be easily embedded into detectors for oriented object detection. A simple implementation of the RoI Transformer has achieved state-of-the-art performances on two common and challenging aerial datasets, i.e., DOTA and HRSC2016, with a neglectable reduction to detection speed. Our RoI Transformer exceeds the deformable Position Sensitive RoI pooling when oriented bounding-box annotations are available. Extensive experiments have also validated the flexibility and effectiveness of our RoI Transformer. The results demonstrate that it can be easily integrated with other detector architectures and significantly improve the performances.

* **主旨翻译**

  航空影像 (aerial images) 目标检测是计算机视觉领域中一项活跃且具有挑战性的任务，该任务中图像以鸟瞰视角 (bird-view perspective) 成像，具有非常复杂的背景，且图像中的物体存在多种变体 (形状、尺寸、纹理等的变化) 。特别地，当航空影像中存在密集排布的目标时，基于水平proposal的常规目标检测方法往往会导致ROI与目标区域不匹配，进而影响最终的分类置信度和定位准确率。虽然目前已经有研究人员使用旋转锚框 (rotated anchors) 来解决上述问题，但是这些方法总是需要引入更多的锚框 (多了一个代表旋转角度的自由度α) ，导致计算复杂度急剧增大。

  在这篇文章中，我们设计并使用**RoI Transformer**来解决上述问题。为了改善region proposal的质量，我们首先设计了一个旋转RoI学习器 (Rotated RoI learner) ，它可以把水平ROI映射为旋转RoI 。之后我们又构造了一个具有旋转位置敏感性的RoI对齐模块 (Rotated Position Sensitive RoI Align) ，该模块可以从ROI中提取出具有旋转不变性 (rotation-invariant) 的特征，改善后续分类与回归操作的效果。ROI Transformer是一个轻量模块，可以被方便地嵌入进旋转目标检测模块中。RoI Transformer在DOTA数据集和HRSC2016数据集上均取得了非常好的检测效果，同时其对检测速度的影响可忽略不计。当数据的旋转边界框标注存在时，我们的ROI Transformer模块在性能上优于deformable Position Sensitive RoI pooling方法。此外大量的实验也验证了该模块的有效性与灵活性，极大地改善检测性能。

---

## Introduction

* 航空影像目标识别：旨在对地面上的感兴趣物体（如飞机、船舶等）进行区域定位和类别标定操作。

* 航空影像数据特点与难点：

  * 成像视角为鸟瞰视角，由卫星或飞机拍摄，成像区域广，背景复杂，成像分辨率较高。
  * 航空图像中目标实例的尺度变化区别是明显的，一是由于不同传感器空间分辨率不同，二是同一类别的目标尺度也有区别，大多数变化为刚性空间形变，几乎不存在非刚性几何形变。
  * 航空影像中许多小目标都紧密聚集在一起，如港口中停泊在一起的多艘船只。此外，不同实例的出现频率也差异较大，例如一些小尺寸的目标在图像中出现了1900次，较大尺寸的目标则只有一小撮。
  * 航空影像中目标的朝向可以是任意角度的。此外，还有一些具有极大纵横比的实例，例如桥梁、舰船。

  除了上述特点与难点外，航空遥感影像不同数据集之间也存在明显差异，识别模型在不同数据集上的泛化能力t通常不佳。

* 最近航空影像目标检测的工作大多基于R-CNN模型，检测流程为基于水平边界框ROI的区域定位过程和在 RoI 上的特征提取与分类过程。在使用水平 RoI (HRoI) 时可能会导致目标区域和定位区域出现明显的错位现象，使得模型训练存在不稳定的隐患，同时也较难完成精准的定位与分类操作，如 Figure 1 所示。

  --[此处插入Figure 1]--

* 目前使用了旋转 ROI 的文章通常在 region proposal 阶段给 anchor 加上了代表旋转角度α的自由度， anchor 的参数集更新为 (纵横比 aspect_ratio , 尺度缩放 scale , 旋转角度 angle)，同时定位框的回归目标也更新为 (框中心坐标 x , 框中心坐标 y , 框宽 w , 框高 h , 框旋转角 α)。若要求旋转的 proposal 和目标区域高度重合，需要 angle 具有较小的分割间隔，则势必需要生成大量具有不同 angle + scale + aspect_ratio 的 anchor ，计算复杂度也会随之增加，检测速度必然会急剧下降。若angle的分割间隔较大，则旋转proposal的定位效果会下降。

* Spatial Transformer 和 Deformable convolution&RoI pooling 两种策略对通常意义上的目标几何形变进行建模，获取具有尺度不变性和旋转不变性的特征，但在处理过程中并没有使用带标签的旋转边界框 (oriented bounding box) 。

* 在航空影像中，目标只存在不弯曲的形变 (rigid deformation) ，同时也有带标注的旋转边界框可供使用。因而该任务中既要提取具有旋转不变性的区域特征，也要消除区域特征和真实目标之间的偏移。

***

## Contribution

> In summary, our contributions are in three-fold:
>
> * We propose a supervised rotated RoI leaner, which is a learnable module that can transform
>   Horizontal RoIs to RRoIs.
> * We design a Rotated Position Sensitive RoI Alignment module for spatially invariant feature
>   extraction, which can effectively boost the object classification and location regression.
> * We achieve state-of-the-art performance on several public large-scale datasets for oriented object
>   detection in aerial images.

文章的主要贡献有三点：

* 提出了有监督的 Rotated RoI Learner ，该模块可以将 HRoI 变换为 RRoI 。
* 提出了 Rotated Position Sensitive RoI Alignment 模块，该模块用来提取空间旋转不变性特征。
* 在几个公开大规模航空影像数据集上进行旋转目标检测试验，取得了非常不错的效果。

---

## Related Work

* ### Oriented Bounding Box Regression

  旋转目标检测是水平目标检测的一个拓展任务，在该任务中使用基于 HRoI 的方法会导致 region feature 和 object feature 发生偏移，因此有了基于 RRoI 的检测方法。RRoI 有了在旋转角度上更加精确的描述，但也会生成更多的 rotated proposal (anchor数量为num_scales × num_aspect ratios × num_angles) ，计算量势必增加，嵌入至网络中时也会极大影响模型的运行效率。同时，由于不是所有生成的 rotated anchor 都是有效的(真正匹配目标的anchor数量是有限的，anchor存在冗余)，oriented bounding boxes (OBBs) 在匹配时难度大于 horizontal bounding boxes (HBBs) 。

  部分文章在设计 rotated anchor 时选取了较为松弛的匹配策略来降低 bounding box 匹配的难度，例如IoU小于0.5的部分anchor也有可能会被判定为 True Positive，但这并没有解决错位 (misalignment) 问题。

* ### Spatial-invariant Feature Extraction

  CNN 框架具有一定的平移不变性，但没有尺度不变性和旋转不变性 (对存在尺度和旋转变化的特征不敏感)，因而有一些文章致力于增强CNN提取尺度不变性和旋转不变性特征的能力：

  * 对于图像级特征提取 (image feature extraction) ，Spatial Transformer 和 deformable convolution 都可以对任意的几何形变进行建模，不需要额外的监督信息，常用在 scene text detection 任务中。
  * 对于区域级特征提取 (region feature extraction) ，deformable RoI Pooling 效果不错，对每个 sampling grid 都学习到一个偏移量 (offset) 。

  针对于航空影像目标检测任务的特点，本文的 RoI Transformer 只对刚性空间形变进行建模，在GT监督信号的辅助下学习到对应的偏移向量 (d~x~ , d~y~ , d~w~ , d~h~ , d~θ~ ) 。

* ### Light RoI-wise Operations

  Two stage 的目标检测方法是基于 proposal 的，需要引入额外的计算量来对 ROI 进行分类与回归，由于 proposal不可复用，当其数量不断增多时，R-CNN系列模型的计算量会陡增。在 Two stage 中，第一个 stage 大多基于 ImageNet 预训练，只是做一个二分类，提取出的 feature map 的 channel 不会很多。而第二个 stage (Head 部分)通常贡献较大的计算量，其计算复杂度取决于 pooling 的 feature map 的厚度，以及对 pooling 操作后的 feature 进行分类和归回的计算层的复杂度。

  Light-Head R-CNN 的作者为了提升R-CNN模型的检测速度，改进了 Head 部分的结构，使得第二个 stage 变得更加灵活可控，降低了 Pooling 的 feature map 的维度，在预测部分引入额外的全连接层，检测效率相比 R-FCN 提升了10倍，甚至超过了部分 one-stage 的检测算法。

  本文中作者在试验阶段使用的是和 deformable RoI pooling 类似的操作方法，RoI层面的计算进行了两轮(计算量变大了)，因而使用 Light Head 的设计思路可以在检测效率上得到一定的保障。

***

## RoI Transformer

​	RoI Transformer 主要包括两部分：RRoI Learner(fc layer) 和 RRoI warping layer。RRoI Learner 根据估算的 horizontal RoIs 学习对应的 rotated RoIs (描述 HRoI 和 RRoI 的映射关系)，之后 RRoI warping layer 将特征图扭曲变形来保持深层次特征仍有一定的旋转不变性。RoI Transformer 的结构如 Fig.2 所示。

​	--[此处插入Fig.2]--

* ### RRoI Learner

  RRoI Learner 的目的是学习从 HRoI 到 RRoI 的映射，该部分的设计基于这样一个假设：

  > Every HRoI is the external rectangle of a RRoI in ideal scenarios.

  即在理想情况下一个目标的 HRoI 是可以把 RRoI 紧密包起来的(非理想情况下 RRoI 的四个直角顶点会越过 HRoI 的边界)。假设已经获得了 $n​$ 个 HRoI，记作$\left\{\mathcal{H}_{i}\right\}​$，其数据结构为 $\left(x, y, w, h\right)​$，每个 HRoI 对应的特征图记作 $\left\{\mathcal{F}_{i}\right\}​$ 。类似于 R-CNN 系列模型，文章中使用目标检测中的 offset learning 方法来设计回归目标，回归目标的计算公式如下所示：
  $$
  \begin{aligned} t_{x}^{*} &=\frac{1}{w_{r}}\left(\left(x^{*}-x_{r}\right) \cos \theta_{r}+\left(y^{*}-y_{r}\right) \sin \theta_{r}\right) \\ t_{y}^{*} &=\frac{1}{h_{r}}\left(\left(y^{*}-y_{r}\right) \cos \theta_{r}-\left(x^{*}-x_{r}\right) \sin \theta_{r}\right) ) \\ t_{w}^{*} &=\log \frac{w^{*}}{w_{r}}, \quad t_{h}^{*}=\log \frac{h^{*}}{h_{r}} \\ t_{\theta}^{*} &=\frac{1}{2 \pi}\left(\left(\theta^{*}-\theta_{r}\right) \quad \bmod 2 \pi\right) \end{aligned}
  $$
  其中 $\left(x_{r}, y_{r}, w_{r}, h_{r}, \theta_{r}\right)​$ 是 RRoI 的位置参数向量，$\left(x^{*}, y^{*}, w^{*}, h^{*}, \theta^{*}\right)​$ 是真实 oriented bounding box 的参数向量，相对偏移的图示描绘于 Fig.3 中。为方便计算，此处将值域为 $[0,2 \pi)​$ 的角度偏移回归目标 $t_{\theta}^{*}​$ 归一化到  $[0,1)​$ 区间。当取值 $\theta^{*}=\frac{3 \pi}{2}​$ 时，上述 RRoI 回归目标即为 HRoI 的回归目标。

  --[此处插入 Fig.3]--

  RRoI Transformer 中的全连接层针对每个特征图 $\mathcal{F}_{i}​$ 输出一个结构为 $\left(t_{x}, t_{y}, t_{w}, t_{h}, t_{\theta}\right)​$ 的向量: 
  $$
  t=\mathcal{G}(\mathcal{F} ; \Theta)
  $$
  其中 $\mathcal{G}$ 代表全连接层，$\Theta$ 为全连接层的参数，$\mathcal{F}$ 是每个 HRoI 的特征图。当训练 $\mathcal{G}$ 时，需要匹配输入的 HRoI 和真值 OBB 。为了便于计算，此处的匹配操作在 HRoI 和坐标轴对齐后的 bounding box 之间进行。Loss function 使用 $Smooth L1$ loss function 。对于每次前向传播输出的预测参数向量 $t$ ，还需要将其从 offset 转换为对应的 RRoI 的参数向量。

  按照上述的处理流程， RRoI Learner 就可以从 HRoI 的特征图中学习到相应的 RRoI 的参数向量。

* ### Rotated Position Sensitive RoI Align

  有了上一小节输出的 RRoI 的参数向量，就有条件为 Oriented Object Detection 提取具有旋转不变性的深度特征。此处作者设计了一个 Rotated Position Sensitive(RPS) RoI Align 模块实现上述目标。

  假设此时有维度为 $H \times W \times C$ 的特征图 $D$ 和参数向量为 $\left(x_{r}, y_{r}, w_{r}, h_{r}, \theta_{r}\right)$ 的 RRoI。RPS RoI pooling 操作先将 RRoI 分割为 $K \times K$ 个小格 (bin)，然后通过计算输出一个结构为 $(K \times K \times C)$ 的特征图 $\mathcal{Y}$ 。对输出通道 $c(0 \leq c<C)$ 中的第 $(i, j)(0 \leq i, j<K)$ 个小方格，RPS RoI pooling 的计算公式为：
  $$
  \mathcal{Y}_{c}(i, j)=\sum_{(x, y) \in \operatorname{bin}(i, j)} D_{i, j, c}\left(\mathcal{T}_{\theta}(x, y)\right) / n_{i j}
  $$
  其中 $D_{i, j, c}$ 是 $K \times K \times C$ 个特征图中的单个特征图。Channel mapping 的方式和原始 Position Sensitive RoI pooling 中的操作相同。$n_{i j}$ 是单个小方格中进行采样的点的个数。$\operatorname{bin _{(i, j)}}$ 表示坐标集：
  $$
  \left\{i \frac{w_{r}}{k}+\left(s_{x}+0.5\right) \frac{w_{r}}{k \times n} ; s_{x}=0,1, \dots n-1\right\} \times\left\{j \frac{h_{r}}{k}+(s_{y}+0.5) \frac{h_{r}}{k \times n} ; s_{y}=0,1, \dots n-1 \}$\right.
  $$
  对每个 $(x, y) \in \operatorname{bin}(i, j)$ ，其坐标经过 $T_{\theta}$ 被映射为 $\left(x^{\prime}, y^{\prime}\right)$ 。映射公式如下所示：
  $$
  \left( \begin{array}{l}{x^{\prime}} \\ {y^{\prime}}\end{array}\right)=\left( \begin{array}{cc}{\cos \theta} & {-\sin \theta} \\ {\sin \theta} & {\cos \theta}\end{array}\right) \left( \begin{array}{c}{x-w_{r} / 2} \\ {y-h_{r} / 2}\end{array}\right)+\left( \begin{array}{l}{x_{r}} \\ {y_{r}}\end{array}\right)
  $$

* ### RoI Transformer for Oriented Object Detection

  RRoI Learner 与 RPS RoI Align 组合起来可以替换掉通用的 RoI warping 操作。这样的改进不仅可以通过 RoI Transformer (RT) 提取出具有旋转不变性的特征，而且由于以 RRoI 作为匹配对象相对于 HRoI 更接近 Rotated Ground Truth，后续回归操作的初始化得到了改善。**(Why?)**

  在本章第一小节我们规定了 RRoI 的位置参数向量为 $\left(x_{r}, y_{r}, w_{r}, h_{r}, \theta_{r}\right)$，为了避免计算过程中的对应关系混乱，此处我们假设 $h$ 为 RRoI 的短边，$w$ 为长边，RRoI 的旋转方向为垂直于短边 $h$ 的角度朝向。

  * **IoU between OBBs**

    IoU 的计算通常出现在 bounding box 的匹配操作和 NMS 操作中。类似于 HBBs 的 IoU 计算方式，此处规定两个 OBBs 之间的 IoU 计算公式为：
    $$
    I_{O} U=\frac{\operatorname{area}\left(B_{1} \cap B_{2}\right)}{\operatorname{area}\left(B_{1} \cup B_{2}\right)}
    $$
    其中 $B_{1}$ 和 $B_{2}$ 分别代表 RRoI 和 RGT。此时两个 bounding box 交叠部分有可能是四边形，也有可能为多边形，如图 Fig.5 所示。若某个 RRoI 和 RGT 之间的 IoU 大于 0.5，那么就将该 RRoI 指定为 True Positive 。当两个 bounding box 是长条形的时候，在角度上微小的偏移会让 IoU 的数值变得很低，进而给 NMS 操作增加难度，如图 Fig.5(b) 所示。

    --[此处插入 Fig.5]--

  * **Targets Calculation**

    RRoI warping 操作输出了具有旋转不变性的特征，此时通过计算得到的偏移量 offsets 也应该是具有旋转不变性的。为了实现这种效果，我们在 offsets 的计算过程中选取和 RRoI 的长短边均平行的坐标系作为计算基准，将利用原始图像坐标系计算出的偏移量投影到新的坐标系中。

---

## Experiments and Analysis

* ### Datasets

  Oriented object detection 实验在两个数据集上进行，分别为：DOTA 和 HRSC2016。

  * **DOTA.** DOTA 数据集是目前已知的 最大的 含朝向边界框标注的 航空遥感影像数据集。数据集包含2806张遥感图像 (大小约4000*4000)，188,282个尺度、朝向和长宽比各异的 instances，分为15个类别，样本类别及数目如下图所示 (与另一个同领域公开数据集 NWPU VHR-10 对比)：

    --[此处插入DOTA数据集样本数量柱形图]--

    实验前我们对原始数据集做了一定的增广。对于 training 和 testing 中的图像按照两种尺度比例 (1.0 和 0.5) 进行缩放操作(**是每张图都做两种缩放还是training 1.0, testing 0.5?**)。之后从原始图像中以 824 像素的步长裁剪出若干个 $1024 \times 1024$ 的 patches 。对于样本数目较少的类别，我们随机地从 (0, 90, 180, 270) 四个角度对样本进行扩充 。(**是四个角度都用了还是只用一个？**)

    按照上述数据增广的操作思路，我们最终得到了 37373 个 patch ，比官方 baseline 所用的 150342 个 patch 少的多。在测试阶段，依旧以 $1024 \times 1024​$ 的大小选取 patch，stride 设定为 512 。

  * **HRSC2016.** HRSC2016 是一个针对航空遥感图像舰船检测任务的数据集，来源是 Google Earth 。数据集中共有 1061 张遥感图像，20 多种不同类别和外型的舰船目标。单张图像尺寸大小从 $300 \times 300$ 到 $1500 \times 900​$ 不等。源数据集中有 436 张作为训练集，181 张作为验证集，444 张作为测试集。在数据增广阶段我们采用了水平翻转操作和将图像 resize 为 (512, 800) 来扩充可用样本数量。

* ### Implementation details

  **Baseline Framework：** 我们的实验借鉴了以 ResNet101 为 backbone 的 Light-Head R-CNN 的网络结构。最终展示的检测效果是基于 FPN 来实现的，但考虑到实现复杂度并没有将其也用在 ablation experiments 章节。

  * #### Light-Head R-CNN OBB

    我们在模型的第二阶段修改了全连接层的回归目标以完成 OBBs 的预测任务。在文献[5]中我们用 $\left(\left(x_{i}, y_{i}\right), i=1,2,3,4\right)$ 来表示 OBB，但在本文的实验中我们将其替换为 $(x, y, w, h, \theta)$ 。由于多了一个参数 $\theta$ ，我们并没有像 Light-Head R-CNN 原文中那样把 regression loss 翻倍。(**WHY?**) 

    在进行可分离卷积时，我们选取 $k=15$，$C m i d=256$，$C out=490$ 。

    训练阶段没有使用 OHEM 来进行采样。

    在 RPN 部分，我们用了 15 个 anchor 作为原始的 Light-Head R-CNN，batch size 设为 512 。在 NMS 操作之后 RoI 的数量由 6000 缩减为 800 。接着有 512 个 RoI 参与 R-CNN 的训练过程。

    在训练时，学习率在初始的 14 个 epoch 设定为 0.0005，之后每隔 4 个 epoch 缩小 10 倍。

    在测试时，RPN 总共生成 6000 个 RoI，经过 NMS 处理剩下 1000 个。

  * #### Light-Head R-CNN OBB with FPN

    Light-Head R-CNN OBB 以 FPN 作为 backbone 。由于基于 FPN 的 Light-Head R-CNN 并没有开源代码，所以我们的实验细节可能有些许调整。

    $P_{2}, P_{3}, P_{4}, P_{5}$ 四个等级的特征经过大尺寸的可分离卷积后直接相加。可分离卷积的超参数设定为 $k=15$，$C m i d=64$，$C out=490$ 。RPN 的 batchsize 设为 512，总共生成 6000 个 RoIs，经过 NMS 操作之后留下 600 个可用 RoIs。接着有 512 个 RoI 参与 R-CNN 的训练过程。

    在训练时，学习率在初始的 5个 epoch 设定为 0.005，之后每隔 2 个 epoch 缩小 10 倍。

* ### Comparison with Deformable PS RoI Pooling

  为了证明性能的提升并不依赖于引入额外的计算，我们和 deformable PS RoI Pooling  模型进行了对比，两个对比模型均采用了 RoI Warping 操作来对几何形变进行建模。在对比实验中，我们使用 Light-Head R-CNN OBB 作为 baseline，将 PS RoI Align 分别替换为 deformable PS RoI Pooling 和 RoI Transformer。

  * #### Complexity

    RoI Transformer 和 deformable RoI pooling 的定位部分都比较轻量化，形式均为一个全连接层后接一个normal pooled(不知道怎么翻译)特征向量。对 RoI Transformer 来说， 只有 $\left(t_{x}, t_{y}, t_{w}, t_{h}, t_{\theta}\right)$ 这五个参数需要学习，而对于 deformable PS RoI pooling 来说，每个 bin 都有一组 offsets 需要学习，可学习参数规模为 $7 \times 7 \times 2$ 。因此我们的模型在计算上相对更简洁一些。从 Tab.4 中，我们观察到 RoI Transformer 相对于 deformable RoI pooling 使用了更少的内存 (273MB V.S. 273.2MB)，推理时间也更快 (0.17s V.S. 0.206s per image)。不过由于训练时需要额外的匹配 RRoI 和 RGT 的操作，RoI Transformer 在训练时的单次推理时间略慢 (0.475s V.S. 0.445s)。

  * #### Detection Accuracy

    检测精度的对比结果展示于 Tab.4 中。Deformable PS RoI pooling 比用作 baseline 的 Light-Head R-CNN OBB 高了 5.6 个百分点，比我们设计的 RoI Transformer 低了 3.85 个百分点。我们认为 RoI Transformer 在性能上的提升有两方面的原因：

    ​	1) 相比于 deformable PS RoI pooling ，RoI Transformer 对物体在几何形状上的刚性变化的建模更加精确。

    ​	2) Deformable PS RoI pooling 的回归目标是相对于 HRoI 来计算的，未使用边界的偏移量。我们的回归目标是相对于 RRoI 来计算的，在初始化时更加精确，不存在歧义性。

    Light-Head R-CNN OBB Baseline、Deformable Position Sensitive RoI pooling 和 RoI Transformer 的部分检测结果分别图示于 Fig. 7，Fig. 8 和 Fig. 9 中。Fig. 7 和 Fig. 8 的第一列图像取自同一幅场景。从展示的结果中我们可以看到 RoI Transformer 能够精确地定位场景中密集排布的物体，而 light-Head R-CNN OBB Baseline 和 deformable Position Sensitive RoI pooling 在定位效果上相对较差。

    在这几幅图中，卡车的头部在三种方法的实验中均被误认为了小型载具，但 RoI Transformer 错分的个数最少。Fig.8 的第二列图像包含了许多细长型的物体，在使用 light-Head R-CNN OBB Baseline 和 deformable Position Sensitive RoI pooling 进行实验时判定的 FN 目标更多，而这些 FN 目标在使用 NMS 进行抑制时效果不是很好，对最终的检测效果有负面影响。相比之下 RoI Transformer 判定的 FN 目标更少一些。

    --[此处插入 Tab.4]--

    --[此处插入 Fig.7]--

    --[此处插入 Fig.8]--

    --[此处插入 Fig.9]--

* ### Ablation Studies

  我们在 DOTA 数据集上进行了一系列模型简化测试来分析 RoI Transformer 的有效性。此处仍然使用 Light-Head R-CNN OBB 作为 baseline，然后不断地改变实验设置来观察检测效果。

  * #### Light RRoI Learner

    为了保证计算效率，我们直接在由 HRoI warping 得到的池化特征后面连接一个输出维度是 5 的全连接层。作为对比，我们也尝试了多个全连接层作为 RRoI Learner，如 Tab.1 的第一第二列所示。我们发现把输出通道为 2048 的多个全连接层当作 RRoI 来使用在 mAP 指标上的下降很小 (0.22 个百分点)，这可能是因为添加了额外的全连接层之后模型需要更长的时间来收敛。

  * #### Contextual RRoI

    文章[9, 42]认为适当地增大 RoI 可以提升检测效果。HRoI 通常包含了许多背景信息，而 RRoI 主要专注于目标实例，如图 Fig.10 所示。完全弃用上下文信息对于定位和识别实例任务来说是不科学的，在这种情况下可能连人工都没法有效地实现检测物体，因此需要在一定程度上适当地扩大 RoI 的感受野。实验中我们分别将 RRoI 的长边和短边尺寸增加至 1.2 和 1.4 倍，最终 AP 值提升了 2.86 个百分点，如 Tab.1 所示。

  * #### NMS on RRoIs

    由于模型中生成的 RoI 是旋转的，因此我们队是否需要对 RRoI 进行 NMS 操作做了实验，结果如 Tab.1 所示，当移除 NMS 操作后 mAP 值提升了约 1.5 个百分点。移除 NMS 后，RoI 的数量更多，recall 值上升，检测结果因而更好。

* ### Comparisons with the State-of-the-art



---

## Conclusion

​	本文中，我们设计了 RoI Transformer 模块来对物体的刚性形变进行建模，同时解决了 region feature 和目标之间的偏移失配问题。这个模块极大地提升了模型在 DOTA 和 HRSC2016 数据集上 oriented object detection 任务的检测效果，在计算复杂度上却只有极小的增加。大量的实验表明，和在 oriented object detection 任务中广泛用来对几何形变建模的可分离模块相比，我们的模型在 oriented bounding box 标注可用时更合理一些。因此可以推断出我们的模块可以替代 deformable RoI pooling 用于旋转目标检测任务。









