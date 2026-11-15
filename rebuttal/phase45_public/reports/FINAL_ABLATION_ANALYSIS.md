# Final ablation analysis

All six causal variants use five paired seeds against the clean Full model. Negative paired differences favor a variant for lower-is-better errors. No-Kernel, Mismatched Geometry and Simple Projection have lower mean RMSE than Full, but their five-seed Wilcoxon tests do not establish significance. No-MHA shows a large and consistent degradation. Alternative Basis and No-Gram are worse on overall RMSE on average. These results support component-dependent behavior on Spiral rather than a universal necessity claim.

The complete per-seed differences, 95% intervals, paired Cohen dz and Wilcoxon p-values are in `results/final_ablation_paired.csv`.
