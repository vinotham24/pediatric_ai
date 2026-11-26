import csv
import io
import os
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from config import REPORTS_DIR, MEDICAL_DISCLAIMER
from services.prediction_status import normalize_prediction_record


class ReportService:
    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def generate_pdf_report(self, record_data):
        record = normalize_prediction_record(record_data)
        filename = f"report_{record.get('id', 'demo')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(REPORTS_DIR, filename)
        doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=42,
                                 leftMargin=42, topMargin=42, bottomMargin=42)
        styles = getSampleStyleSheet()
        title = ParagraphStyle("ResearchTitle", parent=styles["Heading1"], textColor=colors.HexColor("#0F172A"))
        notice = ParagraphStyle("ResearchNotice", parent=styles["Normal"], textColor=colors.HexColor("#7C2D12"), backColor=colors.HexColor("#FFF7ED"), borderPadding=8, spaceBefore=8)
        body = styles["BodyText"]

        score = record.get("model_score")
        if isinstance(score, (int, float)):
            score_text = f"{float(score) * 100:.2f}% raw model score; uncalibrated, not probability of correctness"
        else:
            score_text = "Unavailable"
        rows = [
            ["Audio analysis status", escape(str(record.get("audio_analysis_status", record["prediction_status"])))],
            ["Historical result note", escape(str(record.get("legacy_result_note") or "Not a historical result."))],
            ["Sound classification", escape(str(record.get("sound_classification", "Unavailable: sound not reliably determined.")))],
            ["Detected sound", escape(str(record.get("sound_type", "UNKNOWN")))],
            ["Detection status / method", escape(f"{record.get('sound_detection_status', 'UNKNOWN')} / {record.get('sound_detection_method', 'Not recorded')}")],
            ["Detector score note", escape(str(record.get("sound_confidence_interpretation", "Detector score unavailable or unvalidated.")))],
            ["Cry-reason classification", escape(str(record.get("cry_classification_status", "Not recorded for this earlier result.")))],
            ["Cough classification", escape(str(record.get("cough_classification_status", "Unavailable because a verified cough-event classifier has not been trained.")))],
            ["Medical diagnosis", "Not provided."],
            ["Prediction status", escape(str(record["prediction_status"]))],
            ["Possible category", escape(str(record.get("possible_category") or "Not available"))],
            ["Model score", escape(score_text)],
            ["Screening pattern", escape(str(record["screening_prediction"]))],
            ["Risk assessment", "Not calculated; no validated risk calculation is available."],
            ["Input", escape(str(record.get("source", "UNKNOWN")))],
            ["Audio file", escape(str(record.get("filename", "N/A")))],
        ]
        table = Table(rows, colWidths=[145, 385])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7),
        ]))
        elements = [
            Paragraph("Experimental Audio Analysis Report", title),
            Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} - Research use", body),
            Spacer(1, 14), table, Spacer(1, 14),
            Paragraph(escape(str(record["experimental_notice"])), notice),
            Spacer(1, 6), Paragraph(escape(MEDICAL_DISCLAIMER), body),
        ]
        doc.build(elements)
        return filename, filepath

    def generate_csv_report(self, records_list):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Timestamp", "Input_Mode", "Filename", "Detected_Sound",
            "Audio_Analysis_Status", "Sound_Classification",
            "Historical_Result_Note",
            "Cry_Reason_Classification", "Cough_Classification", "Medical_Diagnosis",
            "Detection_Status", "Detection_Method", "Detector_Score_Limitations",
            "Prediction_Status", "Possible_Category", "Model_Score",
            "Model_Score_Limitations", "Screening_Pattern", "Risk_Status",
            "Research_Disclaimer",
        ])
        for raw in records_list:
            record = normalize_prediction_record(raw)
            score = record.get("model_score")
            writer.writerow([
                record.get("created_at", datetime.now().isoformat()),
                record.get("source", "UNKNOWN"),
                record.get("filename", "N/A"),
                record.get("sound_type", "UNKNOWN"),
                record.get("audio_analysis_status", record["prediction_status"]),
                record.get("sound_classification", "Unavailable: sound not reliably determined."),
                record.get("legacy_result_note") or "",
                record.get("cry_classification_status", "Not recorded for this earlier result."),
                record.get("cough_classification_status", "Unavailable because a verified cough-event classifier has not been trained."),
                "Not provided.",
                record.get("sound_detection_status", "UNKNOWN"),
                record.get("sound_detection_method", "Not recorded"),
                record.get("sound_confidence_interpretation", "Detector score unavailable or unvalidated."),
                record["prediction_status"],
                record.get("possible_category") or "",
                score if score is not None else "",
                record.get("model_score_interpretation", ""),
                record["screening_prediction"],
                record["risk_status"],
                record["experimental_notice"],
            ])
        return output.getvalue()
