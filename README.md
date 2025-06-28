# ECC_DM_for_DR
Offical codebase for MICCAI 2025 paper "_Class-Conditioned Image Synthesis with Diffusion for Imbalanced Diabetic Retinopathy Grading_"

---
Author Feedback according to Reviewer's question:

1) Implement choices. As our work focus on Diffusion Model development, the choices of datasets and classifiers follow fair and effecient DM evaluation purpose. We selected DDR as the primary dataset due to its defined public test split and availability of strong open-source baselines. EyePACS lacks a public test label. Even if used as the main dataset, EyePACS would only reduce the gain from DreamBooth, but our core innovation—semantic filtering and ECC—should remain beneficial. For classifiers, we prioritized established methods rather than custom networks to better isolate the impact of our DM. LANet[10] itself offers SOTA Acc. with diverse 5 backbones, making it ideal for integration. Our method improves performance across top 3 backbones, demonstrating its backbone-independence. [R2Q2] To reduce potential bias from any single classifier, we use a diverse ensemble and it works well in our experiment. In more challenging case where classifier quality is limited, we will consider human-in-the-loop filtering a promising direction.

2) Our model can be trained and inference on a single RTX 4090 (24GB)

---

Under construction
