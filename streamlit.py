import streamlit as st
from PIL import Image
from deepface import DeepFace
import os
from streamlit_option_menu import option_menu
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import psycopg2
import pandas as pd
import smtplib                                    
from email.mime.multipart import MIMEMultipart    
from email.mime.text import MIMEText      
from transformers import pipeline
from sqlalchemy import create_engine, text
import torch
import re
import boto3
import io


st.image((r'E:\chatbot\title.png'))

st.header("🔐 Face Authentication 📷")

REGISTERED_IMAGE = r"E:\Fina_project\faces\WIN_20250709_20_31_51_Pro.jpg"
TEMP_IMAGE = "temp_photo.jpg"

# Initialize access status in session state
if "access_granted" not in st.session_state:
    st.session_state.access_granted = False

# Main page horizontal menu
menu = option_menu(
    menu_title=None,
    options=["Register Face", "Unlock"],
    icons=["person-plus", "lock"],
    orientation="horizontal",
    default_index=0
)

# Save image from Streamlit camera to file
def save_image(image_data, filename):
    image = Image.open(image_data)
    image.save(filename)

# Register Face 
if menu == "Register Face":
    st.subheader("📸 Register Your Face")

    image_data = st.camera_input("Take a clear photo to register")

    if image_data:
        save_image(image_data, REGISTERED_IMAGE)
        st.success("✅ Face Registered Successfully!")
        st.image(REGISTERED_IMAGE, caption="Registered Face", width=300)
        # Reset access status if re-registering
        st.session_state.access_granted = False

# Unlock Face 
elif menu == "Unlock":
    if not os.path.exists(REGISTERED_IMAGE):
        st.warning("⚠️ Please register your face first.")
    else:
        if not st.session_state.access_granted:
            image_data = st.camera_input("Take your photo to unlock")

            if image_data:
                save_image(image_data, TEMP_IMAGE)
                st.image(TEMP_IMAGE, caption="Your Photo", width=300)

                try:
                    result = DeepFace.verify(
                        img1_path=REGISTERED_IMAGE,
                        img2_path=TEMP_IMAGE,
                        enforce_detection=True
                    )

                    if result["verified"]:
                        st.session_state.access_granted = True
                        st.success("✅ Access Granted")
                        st.rerun()  # Reload page to show next section
                    else:
                        st.error("❌ Access Denied")

                except Exception as e:
                    st.error(f"Verification Error: {e}")
        else:
            # Access granted: show protected content
            st.success("🔓 Face Verified. Welcome!")
            with st.sidebar:
                selected = option_menu("Main Menu", 
                        ["About","PPE Monitor", "Mail", "Chatbot","Lock Again"],
                        icons=['info-circle','camera-video', 'envelope', 'robot','lock-fill'],
                        menu_icon="list-task", default_index=0)
            
            if selected == "About":
                st.title("📘 About Factory Guard AI")
                
                st.markdown("""
                **Factory Guard AI** is an intelligent safety compliance system designed to monitor Personal Protective Equipment (PPE) usage in manufacturing environments using computer vision and NLP technologies.

                ### 🔍 System Overview
                - Uses **YOLOv8** (`ultralytics`) for real-time **PPE object detection** (helmet, gloves, mask, goggles).
                - Detects safety violations from uploaded image data and stores results in a **cloud PostgreSQL (AWS RDS)** database.
                - Integrates **DeepFace** for secure **face-based authentication** using Streamlit's camera input.
                - Provides **live querying via chatbot** using a fine-tuned **T5 transformer model** (`mrm8488/t5-base-finetuned-wikiSQL`) from Hugging Face.
                - Sends **automated alerts and daily summary emails** using Gmail SMTP for critical events and daily logs.

                ### ⚙️ Tech Stack
                - **Frontend:** Streamlit
                - **Object Detection:** YOLOv8
                - **Face Verification:** DeepFace
                - **Database:** PostgreSQL hosted on AWS RDS
                - **NLP Model:** T5-base fine-tuned on WikiSQL
                - **Email Integration:** Python `smtplib`, Gmail SMTP
                - **Visualization:** Streamlit + Matplotlib + Pandas

                ### 📌 Key Features
                - Fully automated **PPE violation logging system**
                - **Face-based dashboard access**
                - **Multi-module dashboard** (Detection, Email Alerts, NLP Chatbot)
                - **Data security and real-time analytics**

                """)

            if selected == "PPE Monitor":   
                st.write("### 🛡️ PPE Monitor – How It Works")
                st.markdown("""
                - Upload an image of personnel at the worksite.
                - Click **Detect & Classify** to run the PPE detection model.
                - The model identifies PPE items and classifies compliance.
                - Detected data is stored in your **AWS RDS PostgreSQL** database.
                """)
                
                data_from = option_menu(
                                    menu_title=None,
                                    options=["Local", "S3-Bucket"],
                                    icons=["laptop", "cloud-upload"],
                                    orientation="horizontal",
                                    default_index=0
                                )
                if data_from == 'Local':
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col2:
                        st.image(r"E:\chatbot\arrow.jpg", width=200, caption="Upload Image Here")
                    
                    uploaded_file = st.file_uploader('', type=["jpg", "jpeg", "png"])

                    if uploaded_file :
                        image = Image.open(uploaded_file)
                        st.image(image, caption='Uploaded Image', use_container_width=True)
                        predict = st.button('Detect_Anomaly')
                        if predict:
                            model = YOLO(r'E:\Vs_code\PPE_YOLO_MODEL.pt') #loading model
                            detections = model.predict(image)
                            image1 = detections[0].plot()
                            st.image(image1, caption="Detection Output", channels="RGB")                        
                            india = pytz.timezone('Asia/Kolkata')
                            now = datetime.now(india)                        
                            data = []
                            # Get class names from the model
                            names = model.names
                            # Loop through results
                            for r in detections:
                                boxes = r.boxes
                                for box in boxes:
                                    cls_id = int(box.cls[0])
                                    label = names[cls_id]
                                    conf = float(box.conf[0])
                                    data.append({
                                        "Date" : now.strftime('%Y-%m-%d'),
                                        "Time" : now.strftime('%H:%M:%S'),
                                        "Label": label,
                                        "Confidence": round(conf, 2)
                                    })

                            anamoly_data = pd.DataFrame(data)
                            # RDS connection credentials
                            rds_host = "enter_host_name"
                            rds_user = "postgres"
                            rds_password = "enter_your_password"
                            rds_db = "postgres"
                            rds_port = "5432"

                            # Create SQLAlchemy engine
                            engine = create_engine(f"postgresql+psycopg2://{rds_user}:{rds_password}@{rds_host}:{rds_port}/{rds_db}")

                            # Write transformed data to a new table
                            anamoly_data.to_sql("ppedata", engine, if_exists="append", index=False)

                            st.info("✅ Data transformed and written to RDS")
                            
                if data_from == "S3-Bucket":
                    # AWS CONFIG 
                    AWS_ACCESS_KEY = 'Enter_acces_key'
                    AWS_SECRET_KEY = 'Enter_secret_key'
                    REGION = 'ap-south-1'  # Change to your bucket's region 
                    BUCKET_NAME = 'ppedata'

                    # S3 Functions 
                    def list_images_from_s3(bucket_name):
                        s3 = boto3.client(
                            's3',
                            aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name=REGION
                        )
                        response = s3.list_objects_v2(Bucket=bucket_name)
                        files = response.get('Contents', [])
                        image_keys = [file['Key'] for file in files if file['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))]
                        return image_keys

                    def load_image_from_s3(bucket_name, key):
                        s3 = boto3.client(
                            's3',
                            aws_access_key_id=AWS_ACCESS_KEY,
                            aws_secret_access_key=AWS_SECRET_KEY,
                            region_name=REGION
                        )
                        response = s3.get_object(Bucket=bucket_name, Key=key)
                        image_data = response['Body'].read()
                        return Image.open(io.BytesIO(image_data))

                    # Streamlit UI 
                    st.subheader("📷 Load Images from AWS S3")

                    # List images
                    try:
                        image_list = list_images_from_s3(BUCKET_NAME)

                        if not image_list:
                            st.warning("No images found in the bucket.")
                        else:
                            selected_image = st.selectbox("Select an image", image_list)

                            if selected_image:
                                image = load_image_from_s3(BUCKET_NAME, selected_image)
                                st.image(image, caption=selected_image, use_column_width=True)
                                predict = st.button('Detect_Anomaly')
                                if predict:
                                    model = YOLO(r'E:\Vs_code\PPE_YOLO_MODEL.pt') #loading model
                                    detections = model.predict(image)
                                    image1 = detections[0].plot()
                                    st.image(image1, caption="Detection Output", channels="RGB")                        
                                    india = pytz.timezone('Asia/Kolkata')
                                    now = datetime.now(india)                        
                                    data = []
                                    # Get class names from the model
                                    names = model.names
                                    # Loop through results
                                    for r in detections:
                                        boxes = r.boxes
                                        for box in boxes:
                                            cls_id = int(box.cls[0])
                                            label = names[cls_id]
                                            conf = float(box.conf[0])
                                            data.append({
                                                "Date" : now.strftime('%Y-%m-%d'),
                                                "Time" : now.strftime('%H:%M:%S'),
                                                "Label": label,
                                                "Confidence": round(conf, 2)
                                            })

                                    anamoly_data = pd.DataFrame(data)
                                    # RDS connection credentials
                                    rds_host = "enter_host_name"
                                    rds_user = "postgres"
                                    rds_password = "enter_your_password"
                                    rds_db = "postgres"
                                    rds_port = "5432"

                                    # Create SQLAlchemy engine
                                    engine = create_engine(f"postgresql+psycopg2://{rds_user}:{rds_password}@{rds_host}:{rds_port}/{rds_db}")

                                    # Write transformed data to a new table
                                    anamoly_data.to_sql("ppedata", engine, if_exists="append", index=False)

                                    st.info("✅ Data transformed and written to RDS")
                                
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                                
            if selected == 'Mail':
                mail_menu = option_menu(
                                    menu_title=None,
                                    options=["Alert Mail", "Daily Mail"],
                                    icons=["envelope-exclamation", "calendar"],
                                    orientation="horizontal",
                                    default_index=0
                                    )
                if mail_menu == "Alert Mail":
                    st.info("""Sends an immediate email alert when a **PPE violation** is detected.
                                Useful for ensuring real-time safety compliance.
                                """)
                    def send_email(to_email, subject, message):
                        from_email = 'abishek8171@gmail.com'
                        password = 'enter_password'                          

                        msg = MIMEMultipart()
                        msg["From"] = from_email
                        msg["To"] = to_email
                        msg["Subject"] = subject

                        msg.attach(MIMEText(message, "plain"))

                        try:
                            server = smtplib.SMTP("smtp.gmail.com", 587)
                            server.starttls()
                            server.login(from_email, password)
                            server.send_message(msg)
                            server.quit()
                            return "✅ Email sent successfully!"
                        except Exception as e:
                            return f"❌ Error: {e}"

                    to_email = "abishek.murugesan77@gmail.com"
                    subject = "⚠️ PPE Compliance Alert -Production Area"
                    message = f"""This is an automated notification from the PPE Monitoring System.

                                A potential PPE compliance violation has been detected in the production area. An individual was observed working without following the required personal protective equipment protocols.

                                Kindly take the necessary steps to:
                                - Review the incident at the earliest convenience
                                - Remind personnel in the area about the importance of PPE compliance
                                - Take corrective or disciplinary action if required as per safety policy

                                Your attention to this matter is appreciated to maintain a safe and compliant working environment.
                                """

                    mail = st.button("Send Email")
                    if mail:
                        result = send_email(to_email, subject, message)
                        st.success(result)    

                if mail_menu == "Daily Mail":
                    selected_date = st.date_input("Select a date")

                    # Format as YYYY-MM-DD 
                    formatted_date = selected_date.strftime("%Y-%m-%d")                    
                    st.info("""
                                Sends a daily email summarizing **all PPE anomaly detections**.
                                Helps maintain a daily log of compliance performance.
                                """)
                    rds_host = "enter_host_name"
                    rds_user = "postgres"
                    rds_password = "enter_your_password"
                    rds_db = "postgres"
                    rds_port = "5432"

                    # Create SQLAlchemy engine
                    engine = create_engine(f"postgresql+psycopg2://{rds_user}:{rds_password}@{rds_host}:{rds_port}/{rds_db}")

                    df = pd.read_sql(f"""SELECT * FROM ppedata WHERE "Date" = '{formatted_date}';""", engine)
                    dm = pd.DataFrame(df)
                    dm_table = dm.to_string(index=False)
                    st.dataframe(df)
                    def send_email(to_email, subject, message):
                        from_email = 'abishek8171@gmail.com'
                        password = 'enter_password'                          

                        msg = MIMEMultipart()
                        msg["From"] = from_email
                        msg["To"] = to_email
                        msg["Subject"] = subject

                        msg.attach(MIMEText(message, "plain"))

                        try:
                            server = smtplib.SMTP("smtp.gmail.com", 587)
                            server.starttls()
                            server.login(from_email, password)
                            server.send_message(msg)
                            server.quit()
                            return "✅ Email sent successfully!"
                        except Exception as e:
                            return f"❌ Error: {e}"

                    to_email = "abishek.murugesan77@gmail.com"
                    subject = f"Daily PPE Monitoring – Anomaly Report for {formatted_date}"
                    message = f"""\
                    Dear Team,
                            Please find attached the PPE anomaly status report generated by the monitoring system for the date: {formatted_date}.
                    
                    All relevant records have been successfully stored in the AWS RDS database.
                    
                    Kindly review and take any necessary actions.
                    
                    Best regards,  
                    PPE Monitoring System
                    
                    {dm_table}
                            """
                    mail = st.button("Send Email")
                    if mail:
                        result = send_email(to_email, subject, message)
                        st.success(result)                        
            if selected == "Chatbot":

                # RDS Configuration
                db_user = "postgres"
                db_pass = "enter_your_password"
                db_host = "enter_host_name"
                db_port = "5432"
                db_name = "postgres"

                # Connect to PostgreSQL
                engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

                # Load Model 
                @st.cache_resource
                def load_llm():
                    model_name = "mrm8488/t5-base-finetuned-wikiSQL"
                    return pipeline("text2text-generation", model=model_name)

                llm = load_llm()

                st.subheader("🧠 PPEGuard – Safety Compliance Chatbot")
                st.info("Ask a question based on PPE anomaly data.")

                user_input = st.chat_input("Ask a SQL-related question:")

                def match_template(user_input: str) -> str:
                    input_lower = user_input.lower().strip()

                    if match := re.search(r"anomalies.*?(\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Date\" = '{match.group(1)}';"

                    if match := re.search(r"at (\d{2}:\d{2}:\d{2})", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Time\" = '{match.group(1)}';"

                    if match := re.search(r"after (\d{2}:\d{2}) on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Date\" = '{match.group(2)}' AND \"Time\" > '{match.group(1)}:00';"

                    if match := re.search(r"between (\d{2}:\d{2}) and (\d{2}:\d{2}) on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Date\" = '{match.group(3)}' AND \"Time\" BETWEEN '{match.group(1)}:00' AND '{match.group(2)}:00';"

                    if match := re.search(r"before.*noon.*on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Date\" = '{match.group(1)}' AND \"Time\" < '12:00:00';"

                    if match := re.search(r"label.*'(.*?)'", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Label\" = '{match.group(1)}';"

                    if match := re.search(r"each.*label.*on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT \"Label\", COUNT(*) FROM ppedata WHERE \"Date\" = '{match.group(1)}' GROUP BY \"Label\";"

                    if match := re.search(r"average.*confidence.*for (.*?)", input_lower):
                        parts = match.group(1).strip().split()
                        label = parts[0] if parts else ""
                        return f"SELECT AVG(\"Confidence\") FROM ppedata WHERE \"Label\" = '{label}' AND \"Confidence\" IS NOT NULL;"

                    if match := re.search(r"how many.*on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT COUNT(*) FROM ppedata WHERE \"Date\" = '{match.group(1)}';"

                    if match := re.search(r"confidence.*greater than (\d*\.?\d+)", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Confidence\" > {match.group(1)};"

                    if match := re.search(r"(.*?)violations.*confidence.*below (\d*\.?\d+)", input_lower):
                        label = match.group(1).strip().split()[-1]
                        return f"SELECT * FROM ppedata WHERE \"Label\" = '{label}' AND \"Confidence\" < {match.group(2)};"

                    if match := re.search(r"show only records where label is '(.*?)'", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Label\" = '{match.group(1)}';"

                    if match := re.search(r"related to (glove|helmet|mask|goggles)", input_lower):
                        label = match.group(1)
                        return f"SELECT * FROM ppedata WHERE \"Label\" ILIKE '%{label}%';"

                    if match := re.search(r"how many times was '(.*?)' detected", input_lower):
                        return f"SELECT COUNT(*) FROM ppedata WHERE \"Label\" = '{match.group(1)}';"

                    if match := re.search(r"highest confidence.*on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT * FROM ppedata WHERE \"Date\" = '{match.group(1)}' ORDER BY \"Confidence\" DESC LIMIT 1;"

                    if match := re.search(r"lowest confidence.*for (.*?)", input_lower):
                        parts = match.group(1).strip().split()
                        label = parts[0] if parts else ""
                        return f"SELECT * FROM ppedata WHERE \"Label\" = '{label}' ORDER BY \"Confidence\" ASC LIMIT 1;"

                    if match := re.search(r"average confidence score for all violations on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT AVG(\"Confidence\") FROM ppedata WHERE \"Date\" = '{match.group(1)}' AND \"Confidence\" IS NOT NULL;"

                    if match := re.search(r"anomalies occurred each hour on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT DATE_PART('hour', \"Time\") as hour, COUNT(*) FROM ppedata WHERE \"Date\" = '{match.group(1)}' GROUP BY hour ORDER BY hour;"

                    if match := re.search(r"top 3 most frequent violations on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT \"Label\", COUNT(*) FROM ppedata WHERE \"Date\" = '{match.group(1)}' GROUP BY \"Label\" ORDER BY COUNT(*) DESC LIMIT 3;"

                    if match := re.search(r"most common violation on (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT \"Label\", COUNT(*) FROM ppedata WHERE \"Date\" = '{match.group(1)}' GROUP BY \"Label\" ORDER BY COUNT(*) DESC LIMIT 1;"

                    if "breakdown of all violations grouped by label" in input_lower:
                        return "SELECT \"Label\", COUNT(*) FROM ppedata GROUP BY \"Label\" ORDER BY COUNT(*) DESC;"

                    if match := re.search(r"average confidence per ppe category for (\d{4}-\d{2}-\d{2})", input_lower):
                        return f"SELECT \"Label\", AVG(\"Confidence\") FROM ppedata WHERE \"Date\" = '{match.group(1)}' AND \"Confidence\" IS NOT NULL GROUP BY \"Label\";"

                    return ""

                if user_input:
                    with st.spinner("Generating SQL..."):
                        sql_query = match_template(user_input)

                        if sql_query:
                            st.info(f"🔍 Your question: **{user_input}**")
                        else:
                            prompt = f"translate to SQL: {user_input}"
                            try:
                                output = llm(prompt, max_new_tokens=100)[0]["generated_text"]
                                sql_query = output.strip()
                                if not sql_query.lower().startswith("select"):
                                    st.error("❌ Invalid SQL generated by model.")
                                    sql_query = ""
                                else:
                                    st.info(f"🔍 Your question: **{user_input}**")
                            except Exception as e:
                                st.error(f"❌ LLM Error:\n{e}")
                                sql_query = ""

                        if sql_query:
                            try:
                                with engine.connect() as connection:
                                    result = connection.execute(text(sql_query))
                                    rows = result.fetchall()
                                    if rows:
                                        st.success("✅ Results:")
                                        st.dataframe(rows)
                                    else:
                                        st.warning("⚠️ Query returned no results.")
                            except Exception as e:
                                st.error(f"❌ SQL Execution Error:\n{e}")

            if selected == "Lock Again":
                st.session_state.access_granted = False
                st.rerun()

                
    
