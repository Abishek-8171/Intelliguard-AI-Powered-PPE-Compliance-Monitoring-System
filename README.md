# Intelliguard-AI-Powered-PPE-Compliance-Monitoring-System
Intelliguard is a real-time AI system that monitors whether workers are following PPE (Personal Protective Equipment) rules in factories and industrial settings. It uses a YOLOv8m object detection model to identify missing safety gear like helmets, gloves, or masks. Detected violations are automatically logged into an AWS RDS cloud database, and alerts are sent to the concerned teams to ensure quick response and improved workplace safety.
# Libraries Used
<img width="929" height="356" alt="Screenshot (143)" src="https://github.com/user-attachments/assets/6e22ad4a-c32e-4c59-81f3-f90f9e810b10" />
# 📂Data Collection
- The dataset consists of 24,924 labeled images of workers in industrial settings, categorized into 12 classes: glove, goggles, helmet, mask, no-suit, no_glove, no_goggles, no_helmet, no_mask, no_shoes, shoes, and suit
- Each image captures real-world scenarios of PPE compliance and violations.
- The dataset is organized using YOLO-format annotations and was used to train and test the YOLOv8m object detection model for accurate PPE monitoring
# Model & Tool Details
## Object Detection with YOLOv8m
YOLOv8m (You Only Look Once - version 8, medium variant) is a fast and accurate object detection model developed by Ultralytics. It performs real-time detection by predicting object classes and bounding boxes in a single pass through the image. In this project, YOLOv8m is fine-tuned on a PPE dataset to detect safety gear like helmets, gloves, and masks. It offers a good trade-off between speed and accuracy, making it ideal for web-based applications. The model outputs detected objects along with confidence scores, which are stored for safety analysis.
## Chatbot NLP Model
A T5-base transformer model fine-tuned on WikiSQL (mrm8488/t5-base-finetuned-wikiSQL) from Hugging Face is used to convert natural language questions into SQL queries. It allows users to query PPE violation data through a chatbot interface without writing SQL manually.

