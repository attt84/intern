# intern

AI・ML 系インターン応募向けに整理した総合ポートフォリオです。  
公開データを使った再構成プロジェクトを通じて、問題設定、分析、改善、考察までを一貫して見せることを目的にしています。  
元の授業課題や業務データはそのまま公開せず、テーマだけを残して安全な形に作り直しています。

## このリポジトリで見せたいこと

- 分析や実験を最後までやり切る力
- README で背景と結果を説明する力
- notebook だけで終わらず、最小限のコード整理まで行う力
- 公開可否を意識してポートフォリオを構成する姿勢

## 興味分野

- コンピュータビジョン
- データ分析 / People Analytics
- 時系列需要予測
- 実験設計とモデル改善

## 代表プロジェクト

| プロジェクト | 題材 | 主な技術 | 見どころ |
| --- | --- | --- | --- |
| [NYUv2 Semantic Segmentation](./projects/nyuv2-semantic-segmentation/README.md) | RGB-D を使った室内セマンティックセグメンテーション | PyTorch, albumentations, segmentation-models-pytorch | depth 情報を使った改善方針、mIoU を軸にした評価設計 |
| [Attrition Analysis](./projects/attrition-analysis/README.md) | 公開 HR データを使った離職分析 | pandas, seaborn, scikit-learn | 離職率の分解、職種・JobLevel・残業の観点整理、示唆出し |
| [Mobility Demand Forecasting](./projects/mobility-demand-forecasting/README.md) | 公開移動需要データを使った時系列予測 | pandas, LightGBM, scikit-learn | lag / rolling / 天候・曜日要因の設計、需要予測の実務寄せ整理 |

## 公開方針

- 非公開データ、業務固有名詞、講座配布資料、課題文は含めません
- 元 notebook は参考にしつつ、そのまま転載せずに再構成しています
- データ出典と利用条件は各プロジェクト README に明記しています

詳細は [docs/publication-policy.md](./docs/publication-policy.md) にまとめています。

## リポジトリ構成

```text
.
├── assets/
├── docs/
└── projects/
    ├── attrition-analysis/
    ├── mobility-demand-forecasting/
    └── nyuv2-semantic-segmentation/
```

## 学習姿勢

再現性、比較の軸、失敗した点の記録を重視しています。  
「何を作ったか」だけでなく、「なぜその設計にしたか」「どこを改善したか」が伝わる README を意識しています。
