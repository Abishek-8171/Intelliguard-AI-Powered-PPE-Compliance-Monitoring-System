# Intelliguard-AI-Powered-PPE-Compliance-Monitoring-System
Intelliguard is a real-time AI system that monitors whether workers are following PPE (Personal Protective Equipment) rules in factories and industrial settings. It uses a YOLOv8m object detection model to identify missing safety gear like helmets, gloves, or masks. Detected violations are automatically logged into an AWS RDS cloud database, and alerts are sent to the concerned teams to ensure quick response and improved workplace safety.
# Libraries Used
<img width="965" height="391" alt="Screenshot (152)" src="https://github.com/user-attachments/assets/4f390d11-b503-46fb-b6ae-b9d658ad1505" />

# 📂Data Collection
- The dataset consists of 24,924 labeled images of workers in industrial settings, categorized into 12 classes: glove, goggles, helmet, mask, no-suit, no_glove, no_goggles, no_helmet, no_mask, no_shoes, shoes, and suit
- Each image captures real-world scenarios of PPE compliance and violations.
- The dataset is organized using YOLO-format annotations and was used to train and test the YOLOv8m object detection model for accurate PPE monitoring
# Model & Tool Details
## Object Detection with YOLOv8m
YOLOv8m (You Only Look Once - version 8, medium variant) is a fast and accurate object detection model developed by Ultralytics. It performs real-time detection by predicting object classes and bounding boxes in a single pass through the image. In this project, YOLOv8m is fine-tuned on a PPE dataset to detect safety gear like helmets, gloves, and masks. It offers a good trade-off between speed and accuracy, making it ideal for web-based applications. The model outputs detected objects along with confidence scores, which are stored for safety analysis.
### Model Evaluation Metrics
<img width="2250" height="1500" alt="F1_curve" src="https://github.com/user-attachments/assets/6fe845fd-b2e9-4c72-ba9d-e47db0f2a860" />
<img width="2250" height="1500" alt="P_curve" src="https://github.com/user-attachments/assets/c8c5bab8-b4d4-48a3-ad82-4e347609ab7f" />
<img width="2250" height="1500" alt="R_curve" src="https://github.com/user-attachments/assets/3555710f-46e9-4c45-81e2-42d169876791" />
<img width="2250" height="1500" alt="PR_curve" src="https://github.com/user-attachments/assets/a2d3c93e-a211-4a3f-bca3-eb6d47e45687" />
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/e307d22a-3e67-4e5c-b03f-1de050f64d59" />

## Chatbot NLP Model
A T5-base transformer model fine-tuned on WikiSQL (mrm8488/t5-base-finetuned-wikiSQL) from Hugging Face is used to convert natural language questions into SQL queries. It allows users to query PPE violation data through a chatbot interface without writing SQL manually.

## Email Automation
Email alerts (both immediate and daily summary) are sent using Python’s smtplib along with Gmail’s SMTP server. Alert emails notify safety officers of violations, while daily reports summarize anomaly data logged into the cloud database.

## Streamlit app
### Face Recognition for Secure Login
<img width="1365" height="593" alt="Screenshot (145)" src="https://github.com/user-attachments/assets/3b9aee56-c15c-4a66-959b-23beb388d79b" />

### Home Page
<img width="1365" height="583" alt="Screenshot (146)" src="https://github.com/user-attachments/assets/dbd53cc3-458b-41d8-b627-a84b237c217e" />

### Computer Vision with YOLO
<img width="1366" height="588" alt="Screenshot (150)" src="https://github.com/user-attachments/assets/1197737b-648f-4723-af32-ecd032438908" />

### Email Automation
<img width="1365" height="586" alt="Screenshot (148)" src="https://github.com/user-attachments/assets/46b1ab87-dd17-454a-b2fb-14c9a0f5a812" />

### Chatbot
<img width="1366" height="586" alt="Screenshot (149)" src="https://github.com/user-attachments/assets/68b2c9b9-3d55-4073-b206-30d1d24dd9a7" />



