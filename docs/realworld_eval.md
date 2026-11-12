# 真实场景跨域评估（品牌分类）

- 数据来源：Wikimedia Commons 实拍照片，共 12 张、12 个品牌
- 品牌 Top-1 准确率：**41.67%**（5/12）
- 说明：Stanford Cars 官方划分存在同车多图跨集合泄漏，验证集分数偏高；本表为独立来源实拍数据，更接近真实部署表现。

| 文件 | 真值品牌 | 预测品牌 | 预测车型 | 置信度 | 是否正确 |
|---|---|---|---|---|---|
| BMW.jpg | BMW | BMW | M6 Convertible | 0.973 | ✅ |
| Audi.jpg | Audi | Audi | S6 Sedan | 0.951 | ✅ |
| Toyota.jpg | Toyota | Hyundai | Azera Sedan | 0.399 | ❌ |
| Honda.jpg | Honda | Acura | RL Sedan | 0.218 | ❌ |
| Volkswagen.jpg | Volkswagen | Volkswagen | Golf Hatchback | 0.990 | ✅ |
| Ford.jpg | Ford | Hyundai | Genesis Sedan | 0.376 | ❌ |
| Hyundai.jpg | Hyundai | Hyundai | Elantra Sedan | 0.940 | ✅ |
| Nissan.jpg | Nissan | Hyundai | Azera Sedan | 0.844 | ❌ |
| Chevrolet.jpg | Chevrolet | Acura | Integra Type R | 0.167 | ❌ |
| Kia.jpg | Kia | Honda | Odyssey Minivan | 0.410 | ❌ |
| Mazda.jpg | Mazda | Volvo | XC90 SUV | 0.292 | ❌ |
| Mercedes-Benz.jpg | Mercedes-Benz | Mercedes-Benz | SL-Class Coupe | 0.383 | ✅ |
