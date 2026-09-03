# Ablation Matrix Report: Scope vs. Representation ($D_{JS}$)

- **Test Set Size:** 13 Pairs
- **Layers Evaluated:** [18, 20, 22, 25, 27]
- **Metric:** Mean Next-Token Jensen-Shannon Divergence ($D_{JS}$) relative to unsteered $V_U$.

## Layer 18 Matrix

| Vector             |   p_gen |   visual_tokens |
|:-------------------|--------:|----------------:|
| Mapped Text (W*dT) |  0.0191 |          0.0001 |
| Native Visual (dV) |  0.0157 |          0.0005 |
| Random Null (dR)   |  0.0029 |          0.0001 |
| Raw Text (dT)      |  0.0490 |          0.0005 |

---

## Layer 20 Matrix

| Vector             |   p_gen |   visual_tokens |
|:-------------------|--------:|----------------:|
| Mapped Text (W*dT) |  0.0013 |          0.0001 |
| Native Visual (dV) |  0.0262 |          0.0002 |
| Random Null (dR)   |  0.0072 |          0.0002 |
| Raw Text (dT)      |  0.0537 |          0.0001 |

---

## Layer 22 Matrix

| Vector             |   p_gen |   visual_tokens |
|:-------------------|--------:|----------------:|
| Mapped Text (W*dT) |  0.0042 |          0.0003 |
| Native Visual (dV) |  0.0261 |          0.0001 |
| Random Null (dR)   |  0.0051 |          0.0002 |
| Raw Text (dT)      |  0.0366 |          0.0001 |

---

## Layer 25 Matrix

| Vector             |   p_gen |   visual_tokens |
|:-------------------|--------:|----------------:|
| Mapped Text (W*dT) |  0.0099 |          0.0001 |
| Native Visual (dV) |  0.0276 |          0.0002 |
| Random Null (dR)   |  0.0017 |          0.0001 |
| Raw Text (dT)      |  0.0256 |          0.0001 |

---

## Layer 27 Matrix

| Vector             |   p_gen |   visual_tokens |
|:-------------------|--------:|----------------:|
| Mapped Text (W*dT) |  0.0040 |          0.0000 |
| Native Visual (dV) |  0.0083 |          0.0000 |
| Random Null (dR)   |  0.0010 |          0.0000 |
| Raw Text (dT)      |  0.0185 |          0.0000 |

---

## Overall Average Matrix (Across All Tested Layers)

| Vector             |   p_gen |   visual_tokens |
|:-------------------|--------:|----------------:|
| Mapped Text (W*dT) |  0.0077 |          0.0001 |
| Native Visual (dV) |  0.0208 |          0.0002 |
| Random Null (dR)   |  0.0036 |          0.0001 |
| Raw Text (dT)      |  0.0367 |          0.0002 |
