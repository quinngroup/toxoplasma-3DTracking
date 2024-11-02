# 🧬 Toxoplasma 3D Tracking Project

This repository contains Python scripts and Jupyter Notebooks for tracking **Toxoplasma** in 2D and 3D using various implementations. The project features serial, parallel, and distributed versions, showcasing the flexibility and scalability of the tracking algorithms.

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

## 📄 Description of Implementations

### 1. **Serial Version** (`3D_tracking_serial.py`)
A straightforward implementation for running the tracking algorithm on a single machine, ideal for smaller datasets or initial testing.

### 2. **Parallel Version** (`3D_tracking_parallel.py`, `Parallel_processing_python.ipynb`)
Utilizes Python's multiprocessing capabilities and Joblib to parallelize the tracking process across multiple cores on a server.

### 3. **Distributed Version** (`3D_Tracking_detection_module.ipynb`, `3D_Tracking_module.ipynb`)
Employs Dask to scale the computation across a cluster, making it suitable for processing large datasets efficiently.

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

## 📄 Paper and Citation
For details on the methodology and results, please refer to our related publications. If you use this code in your research, please cite the corresponding papers.

---

## 📧 Contact
For questions or collaboration inquiries, feel free to reach out:
- 📧 mfazli@stanford.edu
- 📧 mfazli@meei.harvard.edu

🚀 Explore the scripts and notebooks to experience efficient and scalable Toxoplasma tracking solutions!
