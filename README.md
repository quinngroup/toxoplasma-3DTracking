# 🧬 Toxoplasma 3D Tracking Project

This repository contains Python scripts and Jupyter Notebooks for tracking **Toxoplasma** in 3D using various implementations. The project features serial, parallel, and distributed versions, showcasing the flexibility and scalability of the tracking algorithms. This work is part of our contribution to the **IEEE Big Data 2018** conference.

---

## 🌟 Project Overview

The repository is structured to provide different approaches for tracking Toxoplasma:

- **🔗 Serial Version**: A single-threaded approach for smaller datasets.
- **⚙️ Parallel Version**: Optimized for running on a server with local multiprocessing.
- **☁️ Distributed Version**: Leveraging Dask for distributed computation across a cluster, suitable for large-scale data.

### Key Technologies Used
- **Python 3.6**
- **Scientific Computing Libraries**: NumPy, SciPy, scikit-learn, matplotlib
- **Computer Vision Tools**: OpenCV for image processing and tracking
- **Parallel and Distributed Computing**:
  - **Dask** for distributed processing
  - **Joblib** and **multiprocessing** for parallel execution
  - **Jupyter Notebook** for prototyping and visualization

---

## 📄 Paper Details

This project is based on the following publication:

**Paper Title**: *Toward Simple & Scalable 3D Cell Tracking*  
**Conference**: 2018 IEEE International Conference on Big Data (Big Data)  
**IEEE Link**: [View on IEEE](https://ieeexplore.ieee.org/abstract/document/8622403)  
**Pre-print Link**: [View on ResearchGate](https://www.researchgate.net/profile/Mojtaba-S-Fazli/publication/329718237_Toward_Simple_Scalable_3D_Cell_Tracking/links/5c1adea3a6fdccfc705ac96a/Toward-Simple-Scalable-3D-Cell-Tracking.pdf)

### Citation
If you use this code or build upon it for your research, please cite our paper using the following format:

```bibtex
@INPROCEEDINGS{8622403,
  author={M. S. Fazli, S. A. Vella, S. N. J. Moreno, G. E. Ward and S. P. Quinn},
  booktitle={2018 IEEE International Conference on Big Data (Big Data)},
  title={Toward Simple & Scalable 3D Cell Tracking},
  year={2018},
  pages={3217-3225},
  doi={10.1109/BigData.2018.8622403},
  keywords={Three-dimensional displays;Videos;Pipelines;Microscopy;Tracking;Two dimensional displays;Genetic algorithms;Cell Detection;Cell Tracking;3D video Tracking;3D Microscopic Videos;Large-Scale tracking method;Computer vision;Toxoplasma Gondii}
}
```

---

## 🚀 How to Run the Code

### Step 1: Prepare Your Data
Ensure that your 2D or 3D image data is stored locally or accessible from your server/cluster. Modify the code to specify the path to the image slices and any additional parameters required.

### Step 2: Run the Code

**For the Serial Version**:
```bash
time python 3D_tracking_serial.py
```

**For the Parallel Version**:
```bash
python 3D_tracking_parallel.py
```

**For the Distributed Version**:
Run the Jupyter Notebook files in an environment with Dask configured:
```bash
jupyter notebook 3D_Tracking_detection_module.ipynb
```

---

## 🛠️ Prerequisites
Ensure that the following dependencies are installed:

```bash
pip install numpy scipy scikit-learn matplotlib opencv-python dask joblib
```

For the distributed version, additional Dask configurations may be needed for your cluster environment.

---

## ✨ Customization Tips
- **Cluster Configuration**: Adjust the Dask client and scheduler settings for optimal performance.
- **Memory Management**: For large-scale data, tweak memory usage settings in Dask to prevent overflow.
- **Parameter Adjustments**: Modify paths, cluster numbers, and other configurable parameters in the scripts/notebooks as per your data and requirements.

---

## 📧 Contact
For questions or collaboration inquiries, feel free to reach out:
- 📧 mfazli@stanford.edu
- 📧 mfazli@meei.harvard.edu

🚀 Explore the scripts and notebooks to experience efficient and scalable Toxoplasma tracking solutions!
