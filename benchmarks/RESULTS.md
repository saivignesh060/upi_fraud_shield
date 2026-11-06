# Real-data benchmark results

**Dataset:** ULB Credit Card Fraud (creditcard.csv)  
**Split:** time-ordered 70% train / 15% val / 15% test  
**Note:** Separate from the UPI synthetic dashboard pipeline.

## Fraud rates

| Split | N | Fraud rate |
|---|---:|---:|
| Train | 199,364 | 0.1926% |
| Val | 42,721 | 0.1311% |
| Test | 42,722 | 0.1217% |

## Threshold (validation)

- Method: `max_recall_precision>=0.5`
- Threshold: **0.148434**

## Test metrics

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9823 |
| PR-AUC | 0.7647 |
| Precision | 0.4699 |
| Recall | 0.7500 |
| F1 | 0.5778 |
