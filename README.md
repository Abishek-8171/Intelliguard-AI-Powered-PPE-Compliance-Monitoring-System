# Intelliguard-AI-Powered-PPE-Compliance-Monitoring-System
Intelliguard is a real-time AI system that monitors whether workers are following PPE (Personal Protective Equipment) rules in factories and industrial settings. It uses a YOLOv8m object detection model to identify missing safety gear like helmets, gloves, or masks. Detected violations are automatically logged into an AWS RDS cloud database, and alerts are sent to the concerned teams to ensure quick response and improved workplace safety.
# Data Collection
- The dataset consists of 24,924 labeled images of workers in industrial settings, categorized into 12 classes: glove, goggles, helmet, mask, no-suit, no_glove, no_goggles, no_helmet, no_mask, no_shoes, shoes, and suit.
- Each image captures real-world scenarios of PPE compliance and violations.
- The dataset is organized using YOLO-format annotations and was used to train and test the YOLOv8m object detection model for accurate PPE monitoring
