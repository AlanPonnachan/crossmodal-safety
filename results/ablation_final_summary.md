# Final Ablation Matrix Report (Scope vs Representation)

- **Test Set Size:** 13 Pairs
- **Layers Aggregated:** All tested layers

## 1. Latent Shift Matrix ($D_{JS}$)
*Measures how much the intervention displaced the internal next-token probability distribution.* 

| condition          |   p_gen |   visual_tokens |
|:-------------------|--------:|----------------:|
| Raw Text (dT)      |  0.0367 |          0.0002 |
| Native Visual (dV) |  0.0208 |          0.0002 |
| Mapped Text (W*dT) |  0.0077 |          0.0001 |
| Random Null (dR)   |  0.0036 |          0.0001 |

## 2. Behavioral Transfer Matrix ($\Delta B$)
*Measures macroscopic semantic change based on LLM Judge score. $\Delta B = 0$ means the generated text did not meaningfully change from the baseline description. Negative values mean slight shifts toward harmful compliance.* 

| condition          |   p_gen |   visual_tokens |
|:-------------------|--------:|----------------:|
| Raw Text (dT)      | -0.3385 |         -0.2462 |
| Native Visual (dV) | -0.3538 |         -0.2308 |
| Mapped Text (W*dT) | -0.2923 |         -0.3231 |
| Random Null (dR)   | -0.1846 |         -0.1231 |
