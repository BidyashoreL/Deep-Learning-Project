# Urban Flood Susceptibility Mapping using Geospatial Deep Learning

A comprehensive deep learning framework for urban flood risk prediction using multi-source geospatial data and hybrid AI architectures.

---

## Abstract

Urban flooding is a major challenge in rapidly growing cities, especially in low-lying and river-adjacent regions. This project presents a data-driven flood susceptibility mapping system that integrates geospatial features with deep learning models.

Multiple architectures are explored, including Convolutional Neural Networks (CNN), U-Net, and a proposed hybrid model (RF-Attention U-Net), to improve prediction accuracy, spatial consistency, and interpretability.

---

## Objectives

- Develop a robust flood susceptibility prediction model  
- Integrate heterogeneous geospatial datasets  
- Compare multiple deep learning architectures  
- Improve model interpretability using hybrid approaches  
- Generate actionable flood risk maps for urban planning  

---

## Model Architectures

### 1. Convolutional Neural Network (CNN)
- Multi-channel convolutional model  
- Learns local spatial features  
- Serves as a baseline model  

### 2. U-Net
- Encoder–decoder architecture with skip connections  
- Preserves spatial resolution  
- Improves segmentation performance over CNN  

### 3. Proposed Model: RF-Attention U-Net

A hybrid architecture combining:

- Random Forest for feature importance extraction  
- Attention gates for spatial feature selection  
- U-Net backbone for multi-scale learning  

Key idea: Random Forest feature importance is used to guide attention mechanisms within the U-Net decoder, improving both performance and interpretability.

---

## Model Comparison

| Model | Strengths | Limitations | Observations |
|------|----------|------------|-------------|
| CNN | Fast, simple, low compute | Poor spatial context | Higher loss, weaker segmentation |
| U-Net | Strong spatial learning, skip connections | No feature prioritization | Better than CNN |
| RF-Attention U-Net | Feature-aware attention, interpretable | Higher complexity | Most stable and lowest loss |

### Training Loss Comparison (Example)

| Model | Loss Range |
|------|-----------|
| CNN | High (unstable) |
| U-Net | Moderate |
| RF-Attention U-Net | Lower and more stable (~0.85 observed) |

---

## Input Data (5-Channel Stack)

| Layer | Description |
|------|------------|
| DEM | Terrain elevation |
| Slope | Terrain gradient |
| LULC (NDBI) | Built-up urban areas |
| Rainfall | Precipitation intensity |
| Drainage Distance | Distance to water bodies |

---

## Methodology

1. Data Acquisition  
   - SRTM DEM  
   - Sentinel-2 imagery  
   - GPM rainfall  
   - OpenStreetMap drainage  

2. Preprocessing  
   - Raster alignment  
   - Feature extraction  
   - Normalization  

3. Model Training  
   - CNN baseline  
   - U-Net segmentation  
   - RF-Attention U-Net hybrid  

4. Prediction  
   - Pixel-wise classification  

5. Post-processing  
   - Risk zoning  
   - Statistical analysis  
   - Visualization  

---

## Study Area

Vijayawada, India  
Located in the Krishna River floodplain, a region highly vulnerable to urban flooding.

---

## Outputs

- Flood susceptibility maps  
- Risk classification (Low, Moderate, High)  
- Flood impact zones  
- Area-based statistics  

---

## Visualization

![Flood Impact Map](outputs/final_flood_impact_map.png)

---

## Risk Classification

- Low Risk  
- Moderate Risk  
- High Risk  

---

## Installation and Execution

```bash
git clone https://github.com/BidyashoreL/Deep-Learning-Project.git
cd Deep-Learning-Project

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python scripts/rf_attention_unet.py
```

---

## Dataset

Datasets are not included due to size constraints. External geospatial datasets can be integrated.

---

## Future Work

- Transformer-based U-Net for global context modeling  
- Graph Neural Networks for spatial relationships  
- Real-time flood monitoring system  
- Web-based GIS visualization platform  

---

## Author

Bidyashore Lourembam

---

## License

MIT License

---

## Contribution Summary

- Implemented CNN, U-Net, and RF-Attention U-Net  
- Designed feature-guided attention mechanism  
- Compared multiple architectures  
- Built an end-to-end geospatial deep learning pipeline  

---
