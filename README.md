# ml
创新点
新增 TemporalAttention 模块（在 new_transformer 变体中）：
在Transformer的输入序列经过位置编码后，添加了一个卷积-based的时间注意力模块（TemporalAttention）。该模块使用一维卷积捕捉局部时间依赖性，结合层归一化和ReLU激活函数增强特征表达。
理由：时间序列数据通常具有局部相关性，原始Transformer模型的全局自注意力机制可能忽略短期的模式。通过引入卷积操作，可以捕捉局部时间特征，提升模型对短期波动和趋势的建模能力，特别是在长序列预测（如365天）时更有效。
优化训练策略：
优化器：将优化器从SGD改为AdamW，增加了权重衰减（weight_decay=0.01）以防止过拟合。
学习率调度：优化了学习率调度策略，在Warmup阶段后加入指数衰减（最低为初始学习率的0.1），以更好地平衡训练初期的快速收敛和后期的稳定优化。
梯度裁剪：添加了梯度裁剪（max_norm=1.0）以稳定训练过程，避免梯度爆炸。
理由：AdamW相比SGD在深度学习任务中收敛更快且更稳定，权重衰减和梯度裁剪进一步增强了模型的泛化能力和训练稳定性。改进的学习率调度策略可以更好地适应长时间序列预测任务的复杂性。
模型变体支持：
在 EnhancedTransformerForecast 中通过 model_variant 参数支持两种模型：original（原始Transformer）和 enhanced（带 TemporalAttention 的新变体）。
理由：这种设计允许灵活比较新旧模型的性能，同时保持代码的模块化和可扩展性，便于未来添加更多变体。
Dropout 和 LayerNorm：
在Transformer模型中显式添加了Dropout层（dropout=0.1）和额外的LayerNorm层，以增强模型的正则化和稳定性。
理由：Dropout可以减少过拟合风险，尤其在处理高维时间序列数据时；LayerNorm在最后输出前进一步规范化特征，提升模型对输入扰动的鲁棒性。
适配修改
主脚本：增加了 new_transformer 选项，通过 model_variant 参数控制Transformer模型的变体（original 或 enhanced）。默认设置为 new_transformer 以优先使用创新模型。
训练脚本：更新了优化器、学习率调度和梯度裁剪逻辑，以适配新模型的复杂性和长时间序列预测需求。绘图部分保持不变，确保与原始代码的评估一致性。
Transformer模型：新增 PositionalEncoding 和 TemporalAttention 类，支持增强的模型结构，同时保留原始Transformer逻辑以便比较。
创新的总体理由
这些创新旨在提升Transformer模型在时间序列预测任务中的性能，特别是在长序列预测（如365天）场景下。TemporalAttention 模块通过卷积操作增强局部特征提取能力，弥补了原始Transformer全局注意力的不足。优化的训练策略（AdamW、学习率调度、梯度裁剪）提高了模型的收敛速度和稳定性。这些改进使模型更适合处理复杂的时间序列数据，同时保持了代码的可扩展性和兼容性。