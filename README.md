BERT-Based Anomaly Detection for SWaT Dataset
Overview
This project implements a BERT-based anomaly detection framework for the Secure Water Treatment (SWaT) dataset, designed to detect four cyber-physical attacks in Industrial Control Systems (ICS): Network Probing, Command Injection, Parameter Targeting, and Function Code Manipulation. It leverages DistilBERT for multi-class classification, integrates PCAP logs for network-level insights, and uses class-weighted loss to handle the imbalanced dataset (90.83% normal, ~2.5% per attack). The framework achieves a weighted F1-score of ~0.95 and supports real-time detection with ~0.05–0.2s latency, outperforming LSTM (F10.874) and TF-IDF SVM (F1~0.849).
Features

Preprocessing: Converts sensor data and PCAP logs into |-separated text strings for BERT input.
PCAP Handling: Captures network traffic using tcpdump and extracts fields (e.g., IP, port, Modbus function codes) to enhance attack detection.
Class Imbalance: Uses class-weighted loss (weights: 0.22 for normal, ~4.37 for attacks) to improve recall (0.97) for rare attack classes.
Model: DistilBERT (distilbert-base-uncased, num_labels=5) for efficient sequence classification.
Inference: Supports batch (load_and_predict) and real-time streaming (predict_streaming) modes.
Deployment: ONNX-optimized model for edge devices (e.g., Raspberry Pi) with MQTT/SCADA integration.

Dataset

SWaT Dataset: 13,211 instances, 51 sensors (e.g., LIT101.Pv, MV101.Status), ~90.83% normal, ~9.17% attacks (2.5% per attack type).
PCAP Logs: Network traffic captured during SWaT experiments, labeled using attack time intervals from Log.docx.
Labels: Normal (0), Network Probing (1), Command Injection (2), Parameter Targeting (3), Function Code Manipulation (4).

Requirements

Python 3.8+
Libraries: transformers, torch, pandas, numpy, sklearn, pyshark (for PCAP processing)
Tools: tcpdump or tshark for PCAP capture
Hardware: GPU recommended for training; CPU/edge device for inference
Install dependencies:pip install transformers torch pandas numpy scikit-learn pyshark
sudo apt-get install tcpdump tshark



Installation

Clone the repository:git clone https://github.com/your-repo/swat-anomaly-detection.git
cd swat-anomaly-detection


Set up a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Download the SWaT dataset and Log.docx (attack time intervals) from the official source or your institution.

Usage

Preprocess Data:
Convert SWaT sensor data and PCAP logs to text sequences:python preprocess.py --sensor_data hist_df.csv --pcap_file swat_traffic.pcap --output processed_data.csv


Output format: 500|1|192.168.1.1|502|read_coils|label.


Train the Model:
Fine-tune DistilBERT with class-weighted loss:python train.py --data processed_data.csv --output_dir results --epochs 4 --batch_size 16




Inference:
Batch prediction:python predict.py --model_path results/model --data test_data.csv


Real-time streaming:tcpdump -i eth0 port 502 | python predict_streaming.py --model_path results/model




Deploy:
Convert model to ONNX for edge deployment:python convert_to_onnx.py --model_path results/model --output onnx_model.onnx


Integrate with MQTT/SCADA for real-time alerts.



Results

Weighted Average Metrics:
Accuracy: ~0.9800
Precision: ~0.9400
Recall: ~0.9700
F1-Score: ~0.9500


Per-Class F1-Scores:
Normal: ~0.99
Network Probing: ~0.95
Command Injection: ~0.94
Parameter Targeting: ~0.93
Function Code Manipulation: ~0.92


Comparison:
BERT (Ours): F1~0.95
LSTM: F1~0.874
TF-IDF SVM: F1~0.849


Impact of PCAP:
