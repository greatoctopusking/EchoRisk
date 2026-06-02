# EchoRisk 评估报告

## 1. 实验设置

| 项目 | 说明 |
|------|------|
| Backbone | UniFormer-Small (22M)，K400 预训练 |
| Phase 1 | EchoNet-Dynamic 7,460 例 A4C 超声预训练 |
| Phase 2 | CARDIOCARE 495 例 A4C + A2C 双视角微调（80/20 按患者拆分） |
| 评估集 | 156 个样本（独立内部验证集，非训练拆出的 val） |
| 帧采样 | Phase 1: 36 帧 frequency=4 / Phase 2: 16 帧 frequency=2 |
| 分辨率 | 112×112 |

---

## 2. 评估结果

### Phase 1 裸评估（EchoNet 预训练 encoder + 原 head，未微调）

| 指标 | Overall (N=300) | A4C (N=155) | A2C (N=145) |
|------|:---:|:---:|:---:|
| R² | -5.10 | -4.99 | -5.23 |
| MAE | 13.76 | 13.65 | 13.88 |
| Bias | -12.06 | — | — |

> Phase 1 的 head 随 EchoNet 训练，在 CARDIOCARE 数据集上失效（R² 为负，系统性低估 EF 约 12 个百分点）。Phase 2 会扔掉此 head，仅用 encoder 权重。

### Phase 2 微调：GatedFusion vs FusionBlock

| 指标 | GatedFusion (2.5K) | FusionBlock (1.2M) |
|------|:---:|:---:|
| R² | 0.264 (0.157 - 0.350) | **0.284** (0.166 - 0.378) |
| MAE | 4.64 (4.16 - 5.15) | **4.38** (3.88 - 4.92) |
| RMSE | 6.00 (5.36 - 6.62) | **5.92** (5.24 - 6.59) |
| Bias | **-0.37** | 1.13 |
| Pred std | 3.2 | 3.5 |
| True std | 7.0 | 7.0 |

#### 按视角分组

| 视角组 | 样本数 | GatedFusion MAE | FusionBlock MAE |
|------|:---:|:---:|:---:|
| 双视角均有 | 144 | 4.56 | **4.26** |
| 仅 A4C | 11 | **5.28** | 5.66 |
| 仅 A2C | 1 | 10.20 | 8.25 |

### 相较 Phase 1 的提升

| 指标 | Phase 1 裸评估 | Phase 2 最佳 (FusionBlock) | 提升 |
|------|:---:|:---:|:---:|
| R² | -5.10 | 0.284 | +5.38 |
| MAE | 13.76 | 4.38 | -68% |
| Bias | -12.06 | 1.13 | 偏差缩小 91% |

---

## 3. 当前瓶颈分析

### 3.1 预测压缩

两个模型的 Pred std（3.2-3.5）均远小于 True std（7.0），EF 极端样本（<45 或 >75）的预测被压缩到均值附近。这是小样本（训练仅 ~380 患者）深度学习模型的经典行为。

### 3.2 A4C→A2C 域偏移

Phase 1 预训练数据仅包含 A4C 视角，encoder 对 A2C 特征提取欠佳。GatedFusion 用两个标量无法补偿这种不对称，FusionBlock 从数据中端到端学习跨视角对齐但在 380 样本下表达能力受限。

### 3.3 FusionBlock 性价比

FusionBlock（1.2M 参数）相比 GatedFusion（2.5K）仅有 6-7% 的 MAE 改善，480 倍参数量换来的提升边际递减严重。

---

## 4. 改进方案

### 方案 A：A2C 校正 + 通道门控（约 563K 参数）

**核心思路**：A2C 特征先过轻量残差适配器校正域偏移，再与 A4C 特征做逐通道级门控融合。

**结构**：

- A2C 适配器（约 66K）：A2C 的 512 维特征压缩到 64 维瓶颈再扩回，作为残差叠加回原始特征。瓶颈结构防止记忆 label，仅校正分布偏移。
- 通道级门控（约 497K）：A4C 和校正后的 A2C 拼接成 1024 维，过 Linear + Sigmoid 得到 512 维门控向量。每个通道独立决定信 A4C 还是信 A2C，而非全局单一权重。
- 综合 512 维 → Linear(512,1) 输出 EF。

**优势**：比 GatedFusion 多了 512 倍表达自由度，比 FusionBlock 少了 60% 参数。极端情况（仅 A4C）通过 NULL embedding 兜底。

### 方案 B：视角一致性特征（约 430K 参数）

**核心思路**：不显式校正 A2C，而是把"两个视角有多像、有多不像"显式交给 MLP。

**结构**：

- 拼接 A4C 特征 (512)、A2C 特征 (512)、逐元素乘积 (512)、绝对值差 (512) → 2048 维
- MLP(2048→256→1) 学习利用一致性和分歧信号

**优势**：不假设 A2C 不可靠，而是让 MLP 从乘积和差值中自学判断信哪个视角。编码简单。

### 方案 C：自蒸馏约束（零额外参数）

**核心思路**：不改结构，只在 loss 中加一项：

```
Loss_total = MSE(EF_pred, EF_true) + λ × MSE(f_A4C, f_A2C)
```

当两视角同时存在时，强制 A2C 特征向 A4C 对齐（A4C 是"老师"）。encoder 的 Stage 4 可训练，通过反向传播学会把 A2C 输入映射到接近 A4C 的特征空间。

**超参数**：λ 建议 0.01-0.1 范围搜索。

**优势**：零参数、零推理开销、可与方案 A 组合使用。失败也无损。

### 推荐实施路径

1. **先试方案 C**（改动最小，一改 loss 即可验证 A4C-A2C 特征对齐的价值）
2. 若 R² 涨超 0.32，叠加方案 A（A2C 适配器 + 通道门控 + 自蒸馏）
3. 方案 B 作为消融对照组

---

## 5. 已知技术问题（已修复）

| 问题 | 表现 | 修复 |
|------|------|------|
| OpenCV FFMPEG 后端缺失 | Windows 无法解码 AVI | 改用 MSMF + worker_init_fn 初始化 COM |
| Torchvision read_video 不兼容 | 不支持 EchoNet AVI 编码 | 回退 OpenCV MSMF |
| DICOM uint8 归一化 | np.float32 类型判断失败 | isinstance 补 np.floating/np.integer |
| 评估脚本 fusion_type 硬编码 | FusionBlock 被当 GatedFusion 评估 | 从 checkpoint 自动检测 |
| DataLoader train→val worker 竞态 | Windows spawn 权限错误 | val 用 num_workers=0 |
| eval_phase1 collate 帧数不一致 | 短视频未 padding | target_frames 固定值 |
