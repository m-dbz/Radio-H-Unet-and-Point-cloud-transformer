# 📡 RadioUNet 3D — Modeling Path Loss with Height-Aware and 3D City Representations

## 🧠 Overview

This repository presents two complementary deep learning approaches for **radio wave path loss prediction** in urban environments.  
Both methods aim to capture **geometric and spatial characteristics** of the environment to improve generalization beyond traditional empirical or 2D-based models.

## 🧩 Models Overview

### 1. Height-Aware RadioUNet (FiLM Conditioning)
- Based on the original [RadioUNet](https://github.com/RonLevie/RadioUNet) architecture.  
- Introduces a **Feature-wise Linear Modulation (FiLM)** layer that conditions intermediate feature maps on **receiver heights**.  

**Goal:** Improve prediction accuracy in the context of low-altitude economy where receiver is not necessarily at ground level.  
Supports arbitrary receiver heights to enable predictions for UAVs and other low-altitude platforms.

---

### 2. Point Cloud Transformer (3D Geometry-Based)
- A **Transformer-based network** designed to process **3D point clouds** representing urban environments (buildings, streets, obstacles, etc.) using [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3).  
- Learns the spatial relationships and geometric obstructions that affect radio propagation.  
- Takes as input a 3D point cloud with associated material.

**Goal:** Explore direct 3D modeling of radio propagation without 2D rasterization, leveraging geometric inductive biases.

## 🗺️ Dataset Generation with NVIDIA Sionna RT

The dataset used in this project was generated using **[NVIDIA Sionna RT](https://developer.nvidia.com/sionna)**, a ray-tracing-based library for **physically accurate wireless channel simulation**.

### Dataset Pipeline:
1. **3D Scene Generation:** Urban environments were modeled as 3D meshes from city geometry datasets.  
2. **Ray Tracing Simulation:** Sionna RT computed **path loss radio maps** by simulating electromagnetic propagation.  
3. **Height Variation:** For each transmitter, several receiver heights were used to capture vertical propagation effects.

This approach allows a **physically consistent** and **diverse** dataset for supervised learning.

---
