import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
import tempfile

from ultralytics import YOLO
import easyocr
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from jinja2 import Template
from weasyprint import HTML

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(
    page_title="TrafficGPT",
    page_icon="🚔",
    layout="wide"
)

# ── Violation options ──────────────────────────────────────────────────────────
VIOLATIONS = [
    "Signal Jump",
    "Over Speeding",
    "No Helmet",
    "Triple Riding",
    "Wrong Side Driving",
    "Using Mobile While Driving",
    "No Seat Belt",
    "Drunk Driving"
]

# ── Load models (cached so they load only once) ────────────────────────────────
@st.cache_resource
def load_yolo():
    return YOLO("model/best.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_chain():
    class ChallanData(BaseModel):
        vehicle_number:    str = Field(description="Vehicle registration number")
        violation_type:    str = Field(description="Type of traffic violation")
        mv_act_section:    str = Field(description="Motor Vehicles Act section number")
        fine_amount_inr:   int = Field(description="Fine amount in Indian Rupees")
        violation_details: str = Field(description="2-3 sentence formal description")
        officer_remarks:   str = Field(description="Short remark from traffic officer")
        court_date:        str = Field(description="Court date 30 days from today")

    parser = PydanticOutputParser(pydantic_object=ChallanData)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["vehicle_number", "violation_type", "location", "date"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
You are an Indian traffic police officer generating a formal traffic challan.

Vehicle Number : {vehicle_number}
Violation Type : {violation_type}
Location       : {location}
Date           : {date}

Generate a formal challan with correct Motor Vehicles Act (India) section,
fine amount as per 2019 amended MV Act, and professional language.

{format_instructions}

Return ONLY valid JSON. No extra text, no markdown, no backticks.
"""
    )

    return prompt | llm | parser, ChallanData

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_plate(crop_img):
    if crop_img is None or crop_img.size == 0:
        return None
    gray   = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel  = np.uint8(np.absolute(sobelx))
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ── Detection + OCR ────────────────────────────────────────────────────────────
def detect_and_read(image_array, yolo_model, ocr_reader):
    img     = image_array.copy()
    results = yolo_model.predict(img, conf=0.5, verbose=False)[0]

    if len(results.boxes) == 0:
        return img, []

    plates = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score      = float(box.conf[0])
        crop            = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        processed = preprocess_plate(crop)
        if processed is None:
            continue

        text         = ocr_reader.readtext(processed, detail=0,
                       allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        plate_number = "".join(text).upper().strip()
        plates.append(plate_number)

        # Draw on image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{plate_number} ({conf_score:.2f})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img, plates

# ── HTML Template ──────────────────────────────────────────────────────────────
CHALLAN_TEMPLATE = """
<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
  body{font-family:Arial,sans-serif;margin:40px;color:#111}
  .header{text-align:center;border-bottom:3px solid #1a237e;padding-bottom:16px;margin-bottom:24px}
  .header h1{color:#1a237e;font-size:22px;margin:0}
  .header h2{color:#333;font-size:16px;margin:6px 0 0;font-weight:normal}
  .badge{background:#d32f2f;color:white;display:inline-block;padding:4px 18px;
         border-radius:4px;font-size:13px;font-weight:bold;margin-top:10px}
  .challan-no{text-align:right;font-size:12px;color:#555;margin-bottom:16px}
  table{width:100%;border-collapse:collapse;margin-bottom:20px}
  th{background:#1a237e;color:white;padding:10px 14px;text-align:left;font-size:13px}
  td{padding:10px 14px;font-size:13px;border-bottom:1px solid #ddd}
  tr:nth-child(even){background:#f5f5f5}
  .fine-box{background:#fff3e0;border:2px solid #e65100;border-radius:6px;
            padding:16px;text-align:center;margin:20px 0}
  .fine-box .amount{font-size:32px;font-weight:bold;color:#e65100}
  .fine-box .label{font-size:13px;color:#555;margin-top:4px}
  .details-box{background:#f9f9f9;border-left:4px solid #1a237e;
               padding:14px;margin:16px 0;font-size:13px;line-height:1.7}
  .footer{margin-top:40px;display:flex;justify-content:space-between;font-size:12px}
  .sig-block{text-align:center}
  .sig-line{border-top:1px solid #333;width:160px;margin:40px auto 6px}
  .warning{background:#ffebee;border:1px solid #ef9a9a;border-radius:4px;
           padding:12px;font-size:12px;color:#b71c1c;margin-top:16px}
</style></head><body>
<div class="header">
  <h1>TRAFFIC POLICE — GOVERNMENT OF INDIA</h1>
  <h2>Motor Vehicle Act Violation Notice</h2>
  <div class="badge">OFFICIAL CHALLAN</div>
</div>
<div class="challan-no">
  Challan No: TRF-{{ challan.vehicle_number | replace(' ','') }}-{{ date | replace(' ','-') }}<br>
  Date of Issue: {{ date }}
</div>
<table>
  <tr><th colspan="2">Vehicle & Violation Details</th></tr>
  <tr><td><b>Vehicle Number</b></td><td>{{ challan.vehicle_number }}</td></tr>
  <tr><td><b>Violation Type</b></td><td>{{ challan.violation_type }}</td></tr>
  <tr><td><b>MV Act Section</b></td><td>{{ challan.mv_act_section }}</td></tr>
  <tr><td><b>Location</b></td><td>{{ location }}</td></tr>
  <tr><td><b>Court Date</b></td><td>{{ challan.court_date }}</td></tr>
  <tr><td><b>Officer Remarks</b></td><td>{{ challan.officer_remarks }}</td></tr>
</table>
<div class="fine-box">
  <div class="amount">Rs. {{ challan.fine_amount_inr }}</div>
  <div class="label">Total Fine Amount Payable</div>
</div>
<div class="details-box">
  <b>Violation Description:</b><br>{{ challan.violation_details }}
</div>
<div class="warning">
  WARNING: Failure to pay the fine within 60 days will result in legal proceedings
  under the Motor Vehicles Act, 1988. Pay at echallan.parivahan.gov.in
</div>
<div class="footer">
  <div class="sig-block"><div class="sig-line"></div>Issuing Officer</div>
  <div class="sig-block"><div class="sig-line"></div>Traffic Superintendent</div>
</div>
</body></html>
"""

def generate_pdf_bytes(challan_data, location):
    html_str = Template(CHALLAN_TEMPLATE).render(
        challan  = challan_data,
        location = location,
        date     = datetime.now().strftime("%d %B %Y")
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        HTML(string=html_str).write_pdf(f.name)
        return open(f.name, "rb").read()

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center;color:#1a237e;'>🚔 TrafficGPT</h1>
    <p style='text-align:center;color:#555;font-size:16px;'>
        License Plate Detection + AI-Powered Challan Generator
    </p>
    <hr style='border:1px solid #e0e0e0;margin-bottom:30px;'>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
with st.spinner("Loading models... (first run takes ~30 seconds)"):
    yolo_model = load_yolo()
    ocr_reader = load_ocr()
    chain, _   = load_chain()

# ── Layout — two columns ───────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Upload & Configure")

    uploaded_file  = st.file_uploader(
        "Upload vehicle image", type=["jpg", "jpeg", "png"]
    )
    violation_type = st.selectbox("Violation Type", VIOLATIONS)
    location       = st.text_input("Location", placeholder="e.g. Connaught Place, New Delhi")

    run_button = st.button("🚨 Detect & Generate Challan", use_container_width=True)

with col2:
    st.subheader("🖼️ Detection Result")
    result_placeholder = st.empty()
    result_placeholder.info("Upload an image and click the button to start.")

# ── On button click ────────────────────────────────────────────────────────────
if run_button:
    if not uploaded_file:
        st.error("❌ Please upload an image first.")
    elif not location.strip():
        st.error("❌ Please enter a location.")
    else:
        # Decode image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Detecting license plate..."):
            annotated_img, plates = detect_and_read(img_bgr, yolo_model, ocr_reader)

        # Show annotated image
        with col2:
            result_placeholder.image(
                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                caption="Detected License Plate",
                use_container_width=True
            )

        if not plates or plates == [""]:
            st.warning("⚠️ No license plate detected. Try a clearer image.")
        else:
            plate_number = plates[0]
            st.success(f"✅ Plate Detected: **{plate_number}**")

            with st.spinner("🤖 Generating challan via Groq AI..."):
                challan_data = chain.invoke({
                    "vehicle_number": plate_number,
                    "violation_type": violation_type,
                    "location":       location,
                    "date":           datetime.now().strftime("%d %B %Y")
                })

            # ── Show challan summary ───────────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Challan Summary")

            m1, m2, m3 = st.columns(3)
            m1.metric("Vehicle Number",  challan_data.vehicle_number)
            m2.metric("Fine Amount",     f"Rs. {challan_data.fine_amount_inr}")
            m3.metric("MV Act Section",  challan_data.mv_act_section)

            with st.expander("📄 View Full Challan Details"):
                st.write(f"**Violation:** {challan_data.violation_type}")
                st.write(f"**Location:** {location}")
                st.write(f"**Description:** {challan_data.violation_details}")
                st.write(f"**Officer Remarks:** {challan_data.officer_remarks}")
                st.write(f"**Court Date:** {challan_data.court_date}")

            # ── PDF download ───────────────────────────────────────────────────
            with st.spinner("📄 Generating PDF..."):
                pdf_bytes = generate_pdf_bytes(challan_data, location)

            st.download_button(
                label="⬇️ Download Challan PDF",
                data=pdf_bytes,
                file_name=f"challan_{plate_number}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <hr style='margin-top:40px;border:1px solid #e0e0e0;'>
    <p style='text-align:center;color:#aaa;font-size:12px;'>
        Built with YOLOv8 · EasyOCR · Groq LLaMA 3.3 · LangChain · Streamlit
    </p>
""", unsafe_allow_html=True)