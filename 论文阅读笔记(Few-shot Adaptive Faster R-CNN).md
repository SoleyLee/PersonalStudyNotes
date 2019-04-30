# 论文阅读笔记(Few-shot Adaptive Faster R-CNN)

## 摘要

> To mitigate the detection performance drop caused by domain shift, we aim to develop a novel few-shot adaptation approach that requires only a few target domain images with limited bounding box annotations. To this end, we first observe several significant challenges. ==First==, the target domain data is highly insufficient, making most existing domain adaptation methods ineffective. ==Second==, object detection involves simultaneous localization and classification, further complicating the model adaptation process. ==Third==, the model suffers from over-adaptation (similar to over-fitting when training with a few data example) and instability risk that may lead to degraded detection performance in the target domain. To address these challenges, we first introduce **a pairing mechanism over source and target features** to alleviate the issue of insufficient target domain samples. We then propose a bi-level module to adapt the source trained detector to the target domain: ==1)== the split pooling based **image level adaptation module** uniformly extracts and aligns paired local patch features over locations, with different scale and aspect ratio; ==2== the **instance level adaptation module** semantically aligns paired object features while avoids inter-class confusion. Meanwhile, a source model feature regularization (SMFR) is applied to stabilize the adaptation process of the two modules. Combining these contributions gives a novel *few-shot adaptive Faster-RCNN framework*, termed FAFRCNN, which effectively adapts to target domain with a few labeled samples. Experiments with multiple datasets show that our model achieves new state-of-the-art performance under both the interested few-shot domain adaptation(FDA) and unsupervised domain adaptation(UDA) setting.

* **主旨翻译**

  为了缓解由于domain-shift(领域迁移)带来的检测性能下降的现象，我们致力于开发一种新的few-shot迁移适应方法。Domain-shift任务中主要面临以下三种挑战：

  1. 目标域的可用数据严重不足，导致现存的大多数domain adaptation方法无法充分奏效。
  2. 目标检测要求同时完成定位与分类操作，使得模型迁移适应更加复杂。
  3. 现有模型在迁移过程中存在over-adaptation(与over-fitting类似)和不稳定性的风险，这些风险可能会导致目标域中检测性能的下降。

  为了克服这些困难，我们首先引入了源特征和目标特征的配对机制来缓解目标域样本不足的问题。之后我们设计了一个双层次模块（图像级&实例级）以使基于源数据训练的检测器迁移适应到新的目标域中来：

  1. 基于分割池化的==图像级适配模块==在具有不同比例和纵横比的定位结果上，均匀地提取并对齐局部配对特征块。
  2. ==实例级适配模块==在语义上对齐成对的对象特征，同时避免类间混淆。

  同时，我们也使用源模型特征正则化操作来稳定两个适配模块(SMFR)的适配过程。结合以上工作，我们提出了few shot adaptive Faster-RCNN框架(FAFRCNN)，该框架只需要少数的带标签样本即可有效迁移适配至新的目标域。在多个数据集上的实验结果表明我们的模型可以在few-shot领域适配和无监督领域适配任务中达到已知最优的性能。



---



