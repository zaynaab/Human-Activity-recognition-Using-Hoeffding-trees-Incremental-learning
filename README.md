
# Human Activity Recognition using Hoeffding Trees (Incremental Learning)

This project implements real-time **Human Activity Recognition (HAR)** using **Hoeffding Trees** for incremental learning. It simulates how a lightweight classifier could work on resource-constrained devices like **Arduino** by processing sensor-like input streams efficiently.

##  Overview

- **Goal:** Classify human activities (e.g., walking, sitting, standing) from time-series data using an online learning model.
- **Approach:** Use Hoeffding Tree classifiers to mimic incremental, memory-efficient behavior suitable for edge devices.
- **Dataset:** [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

## Features

- Real-time activity classification using streaming-like data
- Lightweight incremental learning with Hoeffding Trees
- Simulates Arduino-like behavior using pre-collected sensor data
- Real-time plotting and feedback (if using with Streamlit)

## Project Structure

├── model/
│ └── trained_hoeffding_tree.pkl
├── har_streamlit_app.py
├── utils.py
├── requirements.txt
└── README.md


## ⚙️ Requirements

- Python 3.7+
- scikit-multiflow
- pandas, numpy, matplotlib
- streamlit (for simulation app)

Install all dependencies with:

```bash
pip install -r requirements.txt

streamlit run har_streamlit_app.py
Output
Predicted activity labels in real-time

Live plots simulating sensor data

Feedback interface mimicking Arduino output

 Use Cases
Wearable health and fitness monitoring

Smart homes and assisted living

Embedded ML for low-resource devices

✍️ Author
Syyeda Zainab Bukhari

