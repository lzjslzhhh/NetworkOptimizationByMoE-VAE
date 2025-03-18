## **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">一、引言</font>**
在当前复杂多变的网络环境中，传统的静态资源分配方式已难以满足高动态性、高并发性的网络需求。本文提出一种融合MOE（Mixture of Experts）和GAI（如GPT、扩散模型、GAN、VAE）的网络资源动态分配框架，旨在提升资源调度的智能化与自适应能力。方案的核心在于结合GAI的生成能力与MOE的专家决策优化能力，以实现高效、可靠的网络资源管理。

---

## **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">二、研究背景</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);"></font>
+ <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">网络资源分配的挑战：</font>
    - 服务质量与能效平衡
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">随着我们步入第六代 （6G） 网络时代，无线通信和网络系统的动态正在发生重大转变，其复杂性不断增加，用户需求不断扩大[2]。</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">在无线接入网络中，仍有待进一步解决以下问题：（1）不同业务的差异化QoS难以满足;（2） 网络资源利用率低;（3） 网络负载分配不均衡[6]。</font>
    - 异构化与规模化
        * 网络带宽的增长率无法与数据的增长速度相匹配，使其成为传输瓶颈[8]。
        * 通过将计算任务转移到网络边缘设备，边缘计算促进了实时数据采集、处理和初步分析，从而显著减轻了云的计算负担和网络带宽需求[9]。
    - 动态性与不确定性
        * 5G 技术的重大进步将成为物联网 （IoT） 的巨大驱动力。此外，网络切片的概念为 IoT 应用程序提供了定制的可能性。如何在多场景下优化资源分配，成为我们面临的挑战[4]。
        * 随着电力需求的增加，配电网络不断扩大，导致网络结构更加复杂。因此，运行状态监控、故障告警和优化决策等挑战变得更加突出，对网络的智能化提出了更高的要求[7]。
+ 传统方法的局限性
    - 启发式算法
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">高维搜索空间下的收敛速度慢</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">缺乏对动态环境的自适应能力</font>
    - 集中式优化模型
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">全局信息收集延迟导致的决策滞后</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">中心节点故障引发的系统崩溃风险</font>
    - 深度强化学习的瓶颈
        * 由于强化学习 （RL） 能够适应不断变化的网络条件并优化动态环境中的决策，因此非常适合解决 RAN 切片中的资源分配问题。近年来，DRL 已成为网络切片资源管理的重要方法，并显示出可喜的结果[5]。
        * 网络系统内对复杂优化的需求日益强烈，成为解锁 6G 广泛功能的关键。在各种技术创新中，深度强化学习 （DRL） 是一个关键的推动因素[3]。
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">由于用户需求的范围不断扩大，优化各种无线用户任务对网络系统构成了重大挑战。尽管深度强化学习 （DRL） 取得了进步，但需要为个人用户定制优化任务，这使得开发和应用大量 DRL 模型变得复杂，从而导致大量的计算资源和能源消耗，并可能导致不一致的结果[1]。</font>
+ <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">MoE与GAI的技术优势</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">MOE的核心创新点</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">通过采用一系列 AI 模型作为专业专家，MoE 支持协作决策，从而显著减少对单个任务特定模型训练的需求。在这个框架内，由各种 DRL 策略训练的参与者网络可以充当专家模型，通过联合决策来处理新的和复杂的用户任务。门网络经过常规训练，可以管理和安排这些专家模型，确保最佳任务处理。然而，训练门网络也带来了一系列挑战，包括性能的不确定性以及专家数量和任务复杂性带来的潜在限制[1]。</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">MoE 是一种架构，它结合了专门的神经网络组件，称为专家，每个组件都为处理特定的子任务或数据子集而量身定制[11]。</font>
    - GAI的颠覆性能力
        * GAI 通过预测分析和动态资源分配，可以优化网络运营。生成对抗网络 （GAN） 和强化学习 （RL） 等模型在减少延迟和提高数据吞吐量方面显示出前景[10]。
        * 生成式 AI 可以模拟人类的理解和创造力来生成新的数字内容、利用神经网络等模型来分析和捕获训练数据的潜在模式。然后，使用学习到的模式生成新的输出 ，这与依赖于预定义算法的传统 AI （TAI） 不同。这使得 GAI 在数据生成方面表现出色，具有多项优势[12]。
        * 生成式人工智能 （GAI） 被认为是人工智能 （AI） 领域最重要的进步之一，最近在自然语言处理 （NLP） 和多媒体内容合成等许多领域取得了显着进展[16]。
        * GAI 采用特定的 ML 算法，如变分自动编码器 （VAE） 和生成对抗网络 （GAN） 来生成类似于训练数据的新内容[18]。

---

## **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">三、关键技术调研</font>**
### **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">1. MOE技术体系</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">动态专家选择机制</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：基于门控网络（Gating Network）的权重分配</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">分布式决策优势</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">与联邦学习结合提高公共模型的鲁棒性和可靠性（[32])</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">动态多目标优化（[33]）</font>

### **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">2. GAI技术突破</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">生成式模型类型</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">GAN</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">生成对抗网络 （GAN）  适用于两个神经网络，一个生成器和一个判别器。他们在竞争环境中接受集体培训。生成器旨在产生真实的数据来欺骗判别器。判别器尝试区分实际数据和生成的数据。随着时间的推移，GAN 变得擅长生成越来越逼真的内容[34]。</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">GPT</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">生成式预训练转换器 （GPT） 是由 OpenAI 创建的高级 NLP 模型系列。它是在 Transformer 架构上创建的。它对广泛的 Internet 文本数据进行了预训练，以学习语言模式和语法。当给出启动提示时，GPT 可以生成类似人类的文本作为生成模型[35]。</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">VAE</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">变分自动编码器 （VAE） 是使用编码器-解码器架构的生成模型。它们将输入数据映射到连续的潜在空间，并通过从分布中采样来引入随机性。VAE 经过训练以优化两个目标：重建损失和鼓励结构化潜在空间的正则化项。学习到的潜在空间使 VAE 能够通过采样和解码来生成新的数据点[36]、[37]。</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">核心应用场景</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">单模态 GAI 模型专门用于处理单一类型的数据([19-24]）</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">多模态 GAI 模型则集成和解释多种数据类型([25-28])</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">GAI 驱动的语义通信网络([17])</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">用于空天地一体化网络（[29]）</font>

### **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">3. MOE与GAI的协同机制</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">分层优化框架</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">MOE 增强 GAI 模型的应用([1]、[30-31]、[38-39])</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">GAI层：生成资源分配策略候选集</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">MOE层：实时选择最优策略</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">文献支撑</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：[24-27]（多模态决策）、[28-30]（强化学习与专家系统融合）</font>

---

## **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">四、技术架构设计</font>**
### **1. 环境感知模块**
+ **GAI作用**：
    - 生成虚拟网络环境数据，模拟不同场景（突发流量、受到网络攻击、一般情况）。
+ **MOE作用**：
    - 多维度特征融合（带宽、包速率、攻击压力），预测并分配该网络流量下一步应该分配的带宽。

### **2. 评估反馈模块**
+ 进行可视化分析，确保优化目标的合理性。

## **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">五、实践过程设计</font>**
### **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">1. 仿真实验方案</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">工具选择</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：PyTorch（深度学习框架）+ CTGAN（GAI） +AutoDL（代码运行）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">数据集</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：公开数据集（CIC-IDS2017[40]）+ 合成数据生成（CTGAN）</font>
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">评价指标</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：训练集和验证集上的损失、真实值与预测值的差异</font>

### **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">2. 实验步骤</font>**
1. **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">数据集准备</font>**
    - 在[https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)下载CIC-IDS-2017数据集
    - 采用Wednesday-workingHours.pcap_ISCX.csv进行训练
2. **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">数据预处理</font>**
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">从数据集中提取所需特征、包括流持续时间、带宽、包速率等、并对数据进行归一化从而[47-49]：</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);"> 防止梯度消失或爆炸    </font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);"> 避免特征权重不均衡  </font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);"> 加速收敛  </font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">按照特征进行场景类型标注、分为（TCP、UDP、Attack）三种场景</font>
3. **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">创建MoEResourceDataset类用于训练：</font>**

给各个专家各自提供所需特征和目标值、对应不同场景的特征特点和优化目标

4. **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">基于GTCAN的虚拟网络流量生成器[50]</font>**

该生成器根据不同的场景（“TCP”、“UDP”或“Attack”）生成符合需求的数据，并对其进行适当的修正和处理。生成的数据可以用于模拟网络流量、或用来补充训练集。

5. **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">MoE优化网络带宽分配</font>**

该模型基于 MoE（Mixture of Experts），用于网络流量带宽分配的预测。该模型结合了专家模型和动态门控网络（Dynamic Gating），通过不同专家（TCP、UDP、攻击专家）对不同类型流量的特征进行分析，模型根据门控网络的加权方式，最终计算出每个样本（流量或数据包）所需的带宽。

+ <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">模型设计</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">专家设计</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">其输入是各自所需的特征，输出维度为 1（预测带宽）  </font>
        * **Expert 1**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：处理 TCP 流量（如 HTTP、FTP）、基于带宽预测优化 QoS</font>
        * **Expert 2**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：处理 UDP 流量（如 VoIP、视频流）、优化丢包率，平衡时延</font>
        * **Expert 3**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：处理 异常/攻击流量降低带宽分配、进行流量过滤</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">门控网络设计</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">它用于根据输入序列生成每个专家的权重。该网络的输出是一个（softmax）分布，表示不同专家模型的选择权重。</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">  </font>**
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">组合专家模型输出</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">MoE 模型的核心部分，包含多个专家和一个动态门控网络。根据输入，门控网络为每个专家生成一个权重，然后根据这些权重加权融合专家的输出。  </font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">训练过程</font>
        * **训练步骤**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：进行权重初始化[44]、使用 AdamW[42]优化器训练模型，每四个梯度步骤进行梯度裁剪[45]并更新一次参数。</font>
        * **验证步骤**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：计算验证集的平均损失，并根据损失情况调整学习率[43]。</font>
        * **早停机制**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：若验证损失在连续 </font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">early_stop</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);"> 个 epoch 内没有改善，训练将提前停止[46]。</font>

### **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">3. 实验结果可视化</font>**
1. GAI
    - 真实流量与CTGAN生成虚拟流量(已归一化)数值特征的描述统计：

![](https://cdn.nlark.com/yuque/0/2025/png/43058383/1742210274516-8719c4b2-464f-46b6-8842-d28a7916b17c.png)

2. MoE
    - 损失变化(tensorboard)

![](https://cdn.nlark.com/yuque/0/2025/png/43058383/1742271473559-91cd07f4-a10b-45f7-9f99-ba19bd4ff991.png)

    - 真实值与预测值对比

![](https://cdn.nlark.com/yuque/0/2025/png/43058383/1742271328059-6fb23f74-7958-4a7d-b360-7d80f06ee49f.png)

### **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">4.</font>**具体应用场景：
+ **网络优化**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：通过这种模型，网络管理员或自动化系统可以根据网络流量的特征动态调整带宽分配，优化网络资源使用，避免拥堵或浪费。</font>
+ **流量分类与控制**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：不同类型的流量可能需要不同的带宽分配。MoE模型通过学习流量的特征，能够帮助系统精确地分配适当的带宽。</font>
+ **攻击防护**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：在防止网络攻击（例如拒绝服务攻击）时，通过分析攻击流量的特征，模型可以预测攻击流量的带宽需求，从而采取相应的带宽控制措施进行过滤或限制。</font>

---

## **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">六、挑战与未来方向</font>**
+ **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">前沿方向</font>**<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">：</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">语义通信</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">语义通信侧重于传递源的语义信息，因此发送者和接收者需要共享背景信息，以便有效地进行语义编码和解码 [13]。</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">Mixture-of-Experts （MoE） 模型提供了一种令人信服的方法来增强可信的 SemCom，尤其是在面临多个异构安全挑战时。通过将模型和数据集划分为多个专门的子组，MoE 框架可以适应对不同输入的定制处理 。在 SemCom 中，这种适应性对于解决多种异构安全威胁至关重要。语义编解码器可以由多个专家组成，以抵御各种安全漏洞。SemCom 系统包含一个门控网络，以优化这些专家的选择。该网络根据当前的安全形势和发射器的隐私要求智能地选择最合适的专家或专家组合 。这种动态选择过程可确保语义编码和防御策略既有效又符合通信的特定安全需求，从而提高整体可信度[15]。</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">集成传感和通信</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);"> 在 ISAC 系统中，天线资源需要在通信和感知任务之间动态分配。MoE 模型可以通过多专家网络（不同专家负责不同任务，如通信增强、目标检测等），利用深度强化学习优化波束赋形，使通信和感知性能均衡[51-54]。  </font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">空天地一体化网络</font>
        * <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">结合地、空、天网络技术的天-空-地一体化网络，对于实现 6G 无处不在的连接至关重要。由于其独特的跨层架构，采用 MoE 对于解决不同层中的挑战至关重要。例如，地面网络专家可以专注于资源分配，而空间网络专家应该专注于卫星、机载基站和无人机之间的协同传输 [14]。</font>
    - <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">量子计算加速MOE推理[38-40]、神经符号系统[41-43]</font>

---

## **<font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">七、参考文献</font>**
1. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">H. Du et al., "Mixture of Experts for Intelligent Networks: A Large Language Model-enabled Approach," 2024 International Wireless Communications and Mobile Computing (IWCMC), Ayia Napa, Cyprus, 2024, pp. 531-536, doi: 10.1109/IWCMC61514.2024.10592370. </font>
2. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">S. Dang, O. Amin, B. Shihada and M.-S. Alouini, "What should 6G be?", Nature Electronics, vol. 3, no. 1, pp. 20-29, Jan. 2020.</font>
3. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">N. C. Luong, D. T. Hoang, S. Gong, D. Niyato, P. Wang, Y.-C. Liang, et al., "Applications of deep reinforcement learning in communications and networking: A survey", IEEE Communications Surveys & Tutorials, vol. 21, no. 4, pp. 3133-3174, Apr. 2019.</font>
4. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">S. Li and Q. Hu, "Dynamic Resource Optimization Allocation for 5G Network Slices Under Multiple Scenarios," 2020 Chinese Control And Decision Conference (CCDC), Hefei, China, 2020, pp. 1420-1425, doi: 10.1109/CCDC49329.2020.9164338. </font>
5. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">R. Li et al., "Deep reinforcement learning for resource management in network slicing", IEEE Access, vol. 6, pp. 74429-74441, Nov. 2018.</font>
6. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">M. Lu, H. Zhu, Y. Chen and P. Lin, "Resource-Aware and Cross-Layer Resource Allocation Mechanism for Power Wireless Access Networks," 2018 5th International Conference on Information Science and Control Engineering (ICISCE), Zhengzhou, China, 2018, pp. 1231-1235, doi: 10.1109/ICISCE.2018.00251. </font>
7. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">S Yu, X Chen, Z Zhou et al., "When Deep Reinforcement Learning Meets Federated Learning: Intelligent Multitimescale Resource Management for Multiaccess Edge Computing in 5G Ultradense Network", IEEE Internet of Things Journal, vol. 8, no. 4, pp. 2238-2251, 2021.</font>
8. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">Q Li, S Meng, S Zhang et al., "Complex Attack Linkage Decision-making in Edge Computing Networks", IEEE Access, vol. 7, no. 99, pp. 12058-12072, 2019.</font>
9. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">C Yang, S Lan, L Wang et al., "Big data driven edge-cloud collaboration architecture for cloud manufacturing: a software defined perspective", IEEE Access, vol. 8, no. 1, pp. 45938-45950, 2020.</font>
10. <font style="color:rgba(0, 0, 0, 0.9);background-color:rgb(252, 252, 252);">H. Mahmoud, H. M. Elbadawy, T. Ismail and D. Mi, "A Comprehensive Review of Generative AI Applications in 6G," 2024 6th Novel Intelligent and Leading Emerging Sciences Conference (NILES), Giza, Egypt, 2024, pp. 593-596, doi: 10.1109/NILES63360.2024.10753177. </font>
11. S. E. Yuksel, J. N. Wilson and P. D. Gader, "Twenty Years of Mixture of Experts", IEEE Trans. Neural Networks and Learning Systems, vol. 23, no. 8, pp. 1177-93, 2012.
12. <font style="color:rgb(51, 51, 51);">J. Wang et al., "Guiding AI-Generated Digital Content With Wireless Perception" in IEEE Wireless Commun., 2024.</font>
13. <font style="color:rgb(51, 51, 51);">J. Wang et al., "Semantic-Aware Sensing Information Transmission for Metaverse: A Contest Theoretic Approach", IEEE Trans. Wireless Commun., 2023.</font>
14. <font style="color:rgb(51, 51, 51);">G. Sun et al., "Joint Task Offloading and Resource Allocation in Aerial-Terrestrial UAV Networks With Edge and Fog Computing for Post-Disaster Rescue", IEEE Trans. Mobile Computing, 2024.</font>
15. <font style="color:rgb(51, 51, 51);">J. He et al., "Toward Mixture-of-Experts Enabled Trustworthy Semantic Communication for 6G Networks," in IEEE Network, doi: 10.1109/MNET.2024.3523181.</font>
16. <font style="color:rgb(51, 51, 51);">M. Jovanovic and M. Campbell, "Generative artificial intelligence: Trends and prospects", Computer, vol. 55, no. 10, pp. 107-112, Oct. 2022.</font>
17. <font style="color:rgb(51, 51, 51);">C. Liang et al., "Generative AI-Driven Semantic Communication Networks: Architecture, Technologies, and Applications," in IEEE Transactions on Cognitive Communications and Networking, vol. 11, no. 1, pp. 27-47, Feb. 2025, doi: 10.1109/TCCN.2024.3435524.</font>
18. <font style="color:rgb(51, 51, 51);">G. Harshvardhan, M. K. Gourisaria, M. Pandey and S. S. Rautaray, "A comprehensive survey and analysis of generative models in machine learning", Comput. Sci. Rev., vol. 38, 2020.</font>
19. <font style="color:rgb(51, 51, 51);">J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding", arXiv:1810.04805, 2018.</font>
20. <font style="color:rgb(51, 51, 51);">C. Raffel et al., "Exploring the limits of transfer learning with a unified text-to-text transformer", J. Mach. Learn. Res., vol. 21, no. 1, pp. 5485-5551, 2020.</font>
21. <font style="color:rgb(51, 51, 51);">T. Brown et al., "Language models are few-shot learners", Proc. Adv. Neural Inf. Process. Syst., vol. 33, pp. 1877-1901, 2020.</font>
22. <font style="color:rgb(51, 51, 51);">S. Dai, Z. Gan, Y. Cheng, C. Tao, L. Carin and J. Liu, "APo-VAE: Text generation in hyperbolic space", arXiv:2005.00054, 2020.</font>
23. <font style="color:rgb(51, 51, 51);">S. Tulyakov, M.-Y. Liu, X. Yang and J. Kautz, "MoCoGAN: Decomposing motion and content for video generation", Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 1526-1535, 2018.</font>
24. <font style="color:rgb(51, 51, 51);">J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi and D. J. Fleet, "Video diffusion models", arXiv:2204.03458, 2022.</font>
25. <font style="color:rgb(51, 51, 51);">A. Ramesh et al., "Zero-shot text-to-image generation", Proc. Int. Conf. Mach. Learn., pp. 8821-8831, 2021.</font>
26. <font style="color:rgb(51, 51, 51);">A. Ramesh, P. Dhariwal, A. Nichol, C. Chu and M. Chen, "Hierarchical text-conditional image generation with CLIP latents", arXiv:2204.06125, 2022.</font>
27. <font style="color:rgb(51, 51, 51);">B. Kawar et al., Imagic: Text-based real image editing with diffusion models, 2023.</font>
28. <font style="color:rgb(51, 51, 51);">S. Zhao et al., "Uni-ControlNet: All-in-one control to text-to-image diffusion models", Proc. Adv. Neural Inf. Process. Syst., vol. 36, pp. 1-23, 2024.</font>
29. <font style="color:rgb(51, 51, 51);">R. Zhang et al., "Generative AI for Space-Air-Ground Integrated Networks," in IEEE Wireless Communications, vol. 31, no. 6, pp. 10-20, December 2024, doi: 10.1109/MWC.016.2300547.</font>
30. <font style="color:rgb(51, 51, 51);">N. Du et al., "Glam: Efficient Scaling of Language Models With Mixture-of-Experts", </font>_<font style="color:rgb(51, 51, 51);">Proc. Int'l. Conf. Machine Learning PMLR</font>_<font style="color:rgb(51, 51, 51);">, pp. 5547-69, 2022.</font>
31. <font style="color:rgb(51, 51, 51);">Z. Xue et al., "Raphael: Text-to-Image Generation via Large Mixture of Diffusion Paths", </font>_<font style="color:rgb(51, 51, 51);">arXiv preprint</font>_<font style="color:rgb(51, 51, 51);">, 2023.</font>
32. <font style="color:rgb(51, 51, 51);">T. Bai, Y. Zhang, Y. Wang, Y. Qin and F. Zhang, "Multi-site MRI classification using Weighted federated learning based on Mixture of Experts domain adaptation," 2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Las Vegas, NV, USA, 2022, pp. 916-921, doi: 10.1109/BIBM55620.2022.9994975.</font>
33. <font style="color:rgb(51, 51, 51);">R. Rambabu, P. Vadakkepat, K. C. Tan and M. Jiang, "A Mixture-of-Experts Prediction Framework for Evolutionary Dynamic Multiobjective Optimization," in IEEE Transactions on Cybernetics, vol. 50, no. 12, pp. 5099-5112, Dec. 2020, doi: 10.1109/TCYB.2019.2909806.</font>
34. <font style="color:rgb(51, 51, 51);">  D. Saxena and J. Cao, "Generative Adversarial Networks (GANs): Challenges Solutions and Future Directions", May 2020, [online] Available: </font>[https://arxiv.org/abs/2005.00065.](https://arxiv.org/abs/2005.00065.)
35. R. Gozalo-Brizuela and E. C. Garrido-Merchan, "ChatGPT is not all you need. A State of the Art Review of large Generative AI models", Jan. 2023, [online] Available: [https://arxiv.org/abs/2301.04655.](https://arxiv.org/abs/2301.04655.)
36. D. P. Kingma and M. Welling, "Auto-encoding variational Bayes", Proc. Int. Conf. Learn. Representations, 2014.
37. <font style="color:rgb(51, 51, 51);">D. J. Rezende, S. Mohamed and D. Wierstra, "Stochastic backpropagation and approximate inference in deep generative models", Proc. 31st Int. Conf. Mach. Learn., pp. 1278-1286, 2014.</font>
38. <font style="color:rgb(51, 51, 51);">J. Wang et al., "Toward Scalable Generative Ai via Mixture of Experts in Mobile Edge Networks," in IEEE Wireless Communications, vol. 32, no. 1, pp. 142-149, February 2025, doi: 10.1109/MWC.003.2400046.</font>
39. <font style="color:rgb(51, 51, 51);">C. Zhao et al., "Enhancing Physical Layer Communication Security through Generative AI with Mixture of Experts," in IEEE Wireless Communications, doi: 10.1109/MWC.001.2400150.</font>
40. <font style="color:rgb(51, 51, 51);">Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018</font>
41. <font style="color:rgb(51, 51, 51);">Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv:1406.1078.</font>
42. <font style="color:rgb(51, 51, 51);">Loshchilov, I., & Hutter, F. (2018). Decoupled Weight Decay Regularization. arXiv:1711.05101.</font>
43. <font style="color:rgb(51, 51, 51);">Loshchilov, I., & Hutter, F. (2017). Cyclical Learning Rates for Training Neural Networks. arXiv:1708.07120.</font>
44. <font style="color:rgb(51, 51, 51);">Glorot, X., & Bengio, Y. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks. In Proceedings of the 13th International Conference on Artificial Intelligence and Statistics, 249-256.</font>
45. <font style="color:rgb(51, 51, 51);">Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the Difficulty of Training Recurrent Neural Networks. In Proceedings of the 30th International Conference on Machine Learning (ICML-13), 1310-1318.</font>
46. <font style="color:rgb(51, 51, 51);">Prechelt, L. (1998). Early Stopping - But When? In Neural Networks: Tricks of the Trade, 55-69.</font>
47. <font style="color:rgb(51, 51, 51);">LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.</font>
48. <font style="color:rgb(51, 51, 51);">Hinton, G. E. (2012). Improving neural networks by preventing co-adaptation of feature detectors.</font>
49. <font style="color:rgb(51, 51, 51);">Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.</font>
50. <font style="color:rgb(51, 51, 51);">Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling Tabular Data using Conditional GAN. NeurIPS.</font>
51. <font style="color:rgb(51, 51, 51);">Zhang, J., Liu, Y., Masouros, C., Heath, R. W., Feng, G., & Li, L. (2021). "An overview of signal processing techniques for joint communication and radar sensing." IEEE Journal of Selected Topics in Signal Processing, 15(6), 1295-1315.</font>
52. <font style="color:rgb(51, 51, 51);">Liu, F., Yuan, W., Masouros, C., et al. (2022). "A survey on integrated sensing and communication: From system design to machine learning algorithms." IEEE Journal on Selected Areas in Communications, 40(6), 1722-1739.</font>
53. <font style="color:rgb(51, 51, 51);">Shafin, R., et al. (2020). "Artificial intelligence-enabled cellular networks: A critical path to beyond-5G and 6G." IEEE Wireless Communications, 27(2), 212-217.</font>
54. <font style="color:rgb(51, 51, 51);">Guo, H., Zhang, H., Wang, J., Zhang, J. (2023). "Deep Learning for ISAC: A Survey on Algorithm Designs and Optimization Strategies." IEEE Communications Surveys & Tutorials, 25(1), 67-98.</font>



