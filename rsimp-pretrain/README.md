# rsimp-pretrain

Pretrained models for remote sensing image imputation.

# Datasets

## 1. Sen2_MTC_New Dataset

### Overview

Sen2_MTC_New (Sentinel-2 Multi-Temporal Cloudy Dataset) provides paired cloudy and cloud-free Sentinel-2 images for cloud removal and multi-temporal analysis. Each location contains 10 time points.

### Dataset Structure

```
Sen2_MTC_New/
├── T09WWP_R071/
│   ├── cloud/        # Cloud-contaminated images
│   │   ├── t00.tif
│   │   ├── t01.tif
│   │   └── ...
│   └── cloudless/    # Cloud-free images
│       ├── t00.tif
│       ├── t01.tif
│       └── ...
├── T12TUR_R027/
└── ...
```

### Channels (4 Channels)

| Channel | Band Name  | Wavelength | Resolution | Description                                   |
| ------- | ---------- | ---------- | ---------- | --------------------------------------------- |
| 0       | B4 - Red   | 664.5 nm   | 10m        | Chlorophyll absorption, vegetation monitoring |
| 1       | B3 - Green | 560.0 nm   | 10m        | Vegetation reflection peak, health monitoring |
| 2       | B2 - Blue  | 496.6 nm   | 10m        | Water penetration, atmospheric correction     |
| 3       | B8 - NIR   | 835.1 nm   | 10m        | Vegetation health indicator                   |

### Key Features

- Multi-temporal: 10 time points per location
- Paired cloudy and cloud-free images
- File format: GeoTIFF (.tif)
- Spatial resolution: 256×256 pixels
- Radiometric resolution: 16-bit (scaled by 10000)

### Preprocessing

```python
data = data / 10000.0  # Convert to 0-1 reflectance
```

---

## 2. EuroSAT-MS Dataset

### Overview

EuroSAT-MS is a multi-spectral Sentinel-2 dataset for land use and land cover classification, containing 27,000 samples across 10 classes.

### Dataset Structure

```
EuroSAT_MS/
├── AnnualCrop/
│   ├── AnnualCrop_1.tif
│   ├── AnnualCrop_2.tif
│   └── ...
├── Forest/
├── HerbaceousVegetation/
└── ...
```

### Channels (13 Channels)

| File Idx | Band Name    | Wavelength | Resolution |
| -------- | ------------ | ---------- | ---------- |
| 0        | B2 - Blue    | 496.6 nm   | 10 m       |
| 1        | B3 - Green   | 560.0 nm   | 10 m       |
| 2        | B4 - Red     | 664.5 nm   | 10 m       |
| 3        | B8 - NIR     | 835.1 nm   | 10 m       |
| 4        | B5 - RE1     | 703.9 nm   | 20 m       |
| 5        | B6 - RE2     | 740.2 nm   | 20 m       |
| 6        | B7 - RE3     | 782.5 nm   | 20 m       |
| 7        | B8A - NNIR   | 864.8 nm   | 20 m       |
| 8        | B9 - WV      | 945.0 nm   | 60 m       |
| 9        | B10 - Cirrus | 1373.5 nm  | 60 m       |
| 10       | B1 - Coastal | 443.9 nm   | 60 m       |
| 11       | B11 - SWIR1  | 1613.7 nm  | 20 m       |
| 12       | B12 - SWIR2  | 2202.4 nm  | 20 m       |

### Key Features

- 13 spectral bands
- 10 land cover classes
- 27,000 samples, 64×64 pixels
- File format: GeoTIFF (.tif)
- Radiometric resolution: 16-bit (scaled by 10000)

### Preprocessing

```python
data = data / 10000.0  # Convert to 0-1 reflectance
```

---

## 3. SEN12MS-CR-TS Dataset

### Overview

SEN12MS-CR-TS provides multi-modal time series for cloud removal research. It combines Sentinel-1 SAR and Sentinel-2 optical imagery with 30 temporal points.

### Dataset Structure (HDF5 Format)

```python
{
    's1': [30, 2, 256, 256],      # Sentinel-1 SAR (VV, VH)
    's2': [30, 13, 256, 256],     # Sentinel-2 optical
    's2_cloudfree': [30, 13, 256, 256]  # Cloud-free reference
}
```

### Sentinel-2 Channels (13 Channels)

| Channel | Band Name    | Wavelength | Resolution |
| ------- | ------------ | ---------- | ---------- |
| 0       | B1 - Coastal | 443.9 nm   | 60m        |
| 1       | B2 - Blue    | 496.6 nm   | 10m        |
| 2       | B3 - Green   | 560.0 nm   | 10m        |
| 3       | B4 - Red     | 664.5 nm   | 10m        |
| 4       | B5 - RE1     | 703.9 nm   | 20m        |
| 5       | B6 - RE2     | 740.2 nm   | 20m        |
| 6       | B7 - RE3     | 782.5 nm   | 20m        |
| 7       | B8 - NIR     | 835.1 nm   | 10m        |
| 8       | B8A - NNIR   | 864.8 nm   | 20m        |
| 9       | B9 - WV      | 945.0 nm   | 60m        |
| 10      | B10 - Cirrus | 1373.5 nm  | 60m        |
| 11      | B11 - SWIR1  | 1613.7 nm  | 20m        |
| 12      | B12 - SWIR2  | 2202.4 nm  | 20m        |

### Sentinel-1 Channels (2 Channels)

| Channel | Polarization | Frequency      |
| ------- | ------------ | -------------- |
| 0       | VV           | 5.4 GHz C-band |
| 1       | VH           | 5.4 GHz C-band |

### Key Features

- Multi-temporal (30 points)
- Sentinel-1 SAR + Sentinel-2 optical
- Cloud-free references included
- File format: HDF5, 16-bit for optical, float for SAR
- Spatial resolution: 256×256 pixels
