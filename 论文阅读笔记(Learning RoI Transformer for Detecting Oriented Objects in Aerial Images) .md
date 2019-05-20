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

  RRoI Learner 与 RPS RoI Align 组合起来可以替换掉通用的 RoI warping 操作。这样的改进不进可以通过 RoI Transformer (RT) 提取出具有旋转不变性的特征，而且由于以 RRoI 作为匹配对象相对于 HRoI 更接近 Rotated Ground Truth，后续回归操作的初始化得到了改善。**(Why?)**

  

  * IoU between OBBs

    

  * Targets Calculation

    

---

## Experiments and Analysis









