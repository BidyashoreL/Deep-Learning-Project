# Deep-Learning-Project
Urban flood analysis and alert system using deep learning 
#  Urban Flood Susceptibility Mapping using Geospatial Deep Learning

AI-powered smart city flood risk prediction using multi-source satellite data and a 5-channel CNN model.

---

## Project Overview

Urban flooding is one of the most critical climate-related disasters affecting smart cities.

This project builds a **deep learning–based flood susceptibility model** that integrates:

-  DEM (Elevation)
-  Slope
-  Built-up Index (NDBI from Sentinel-2)
-  Rainfall (GPM / dynamic rainfall simulation)
-  Drainage proximity (canals, rivers, waterways)

The system generates:

✔ Flood susceptibility maps  
✔ High-risk impact zones  
✔ Affected urban areas  
✔ Decision-support visualization  

---

##  Model Architecture

We use a **5-channel Convolutional Neural Network** for spatial flood pattern learning.

### Input Stack

| Layer | Description |
|------|------------|
DEM | Terrain elevation |
Slope | Water flow velocity indicator |
LULC (NDBI) | Urban impermeable surfaces |
Rainfall | Precipitation intensity |
Drainage Distance | Proximity to canals/rivers |

---

##  Workflow
---

##  Data Sources

- **SRTM DEM** – Terrain elevation  
- **Sentinel-2** – Land use / built-up extraction  
- **GPM IMERG** – Rainfall  
- **OpenStreetMap / GeoFabrik** – Roads & drainage network  

---

##  Study Area

 Vijayawada, India  
Krishna River floodplain – high urban flood vulnerability zone.

---

## Outputs

###  Flood Susceptibility Map
AI-predicted spatial flood risk.

### Risk Classes
- 🟩 Low
- 🟧 Moderate
- 🟥 High

### Impact Zones
Buffered high-risk areas showing potential flood spread.

---

##  Final Result

![Flood Impact Map](outputs/final_flood_impact_map.png)

---

##  Flood Risk Area Statistics

Automatically computed:

- High-risk area (km²)
- Moderate-risk area (km²)
- Safe zone (km²)

---

##  How to Run the Project

### 1️ Clone the repository

```bash
git clone https://github.com/BidyashoreL/Deep-Learning-Project.git
cd Deep-Learning-Project