import os
import csv
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from config import REPORTS_DIR, MEDICAL_DISCLAIMER

class ReportService:
    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def generate_pdf_report(self, record_data):
        """
        Generates a PDF screening report.
        """
        filename = f"report_{record_data.get('id', 'demo')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(REPORTS_DIR, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            'ReportSubTitle',
            parent=styles['Normal'],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#475569'),
            spaceAfter=15
        )
        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Heading2'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#1E293B'),
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#334155')
        )
        disclaimer_style = ParagraphStyle(
            'DisclaimerText',
            parent=styles['Normal'],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#991B1B')
        )

        elements = []

        # Header
        elements.append(Paragraph("Pediatric AI Health Screening Report", title_style))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y - %H:%M')} | Confidential Research Report", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=12))

        # Risk Banner Color
        risk_level = record_data.get('risk_level', 'LOW')
        if risk_level == 'HIGH':
            bg_color = colors.HexColor('#FEE2E2')
            text_color = colors.HexColor('#991B1B')
        elif risk_level == 'MODERATE':
            bg_color = colors.HexColor('#FEF3C7')
            text_color = colors.HexColor('#92400E')
        else:
            bg_color = colors.HexColor('#DCFCE7')
            text_color = colors.HexColor('#166534')

        risk_banner = Table([
            [Paragraph(f"<b>OVERALL ACQUIRED SCREENING RISK: {risk_level}</b> (Risk Score: {record_data.get('risk_score', 0)}/100)", 
                       ParagraphStyle('RiskBanner', parent=body_style, fontSize=11, textColor=text_color, alignment=1))]
        ], colWidths=[540])
        risk_banner.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_color),
            ('PADDING', (0,0), (-1,-1), 8),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOX', (0,0), (-1,-1), 1, text_color)
        ]))
        elements.append(risk_banner)
        elements.append(Spacer(1, 10))

        # Metadata Table
        meta_data = [
            [Paragraph("<b>Child Age Group:</b>", body_style), Paragraph(str(record_data.get('age_group', '0-12m')), body_style),
             Paragraph("<b>Input Mode:</b>", body_style), Paragraph(str(record_data.get('source', 'LIVE')), body_style)],
            [Paragraph("<b>Audio Filename:</b>", body_style), Paragraph(str(record_data.get('filename', 'N/A')), body_style),
             Paragraph("<b>Duration:</b>", body_style), Paragraph(f"{record_data.get('duration', 0.0)} sec", body_style)],
            [Paragraph("<b>Detected Sound:</b>", body_style), Paragraph(str(record_data.get('sound_type', 'N/A')), body_style),
             Paragraph("<b>Model Confidence:</b>", body_style), Paragraph(f"{int(record_data.get('confidence', 0)*100)}%", body_style)]
        ]
        meta_table = Table(meta_data, colWidths=[120, 150, 120, 150])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 12))

        # Disease Screening Probabilities
        elements.append(Paragraph("Disease & Acoustic Screening Probabilities", h2_style))
        prob_rows = [["Screening Classification / Acoustic Pattern", "Probability (%)"]]
        probs = record_data.get('probabilities', {})
        for pattern, prob in probs.items():
            prob_rows.append([Paragraph(pattern, body_style), Paragraph(f"{prob*100:.1f}%", body_style)])

        prob_table = Table(prob_rows, colWidths=[380, 160])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('PADDING', (0,0), (-1,-1), 5)
        ]))
        elements.append(prob_table)
        elements.append(Spacer(1, 12))

        # AI Assistant Summary & IEEE Research Grounding
        elements.append(Paragraph("Acoustic Feature Rationale & IEEE Literature Grounding", h2_style))
        exp_text = record_data.get('explanation_summary', 'No detailed explanation generated.')
        elements.append(Paragraph(exp_text, body_style))
        elements.append(Spacer(1, 8))

        # Scientific Literature Box
        ieee_ref_text = record_data.get('ieee_ref', 'IEEE Transactions on Biomedical Engineering (TBME): Acoustic Feature Saliency and Automated Pediatric Respiratory Screening.')
        lit_box = Table([
            [Paragraph(f"<b>IEEE Scientific Reference:</b> {ieee_ref_text}", ParagraphStyle('LitRef', parent=body_style, fontSize=8.5, textColor=colors.HexColor('#1E3A8A')))]
        ], colWidths=[540])
        lit_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
            ('PADDING', (0,0), (-1,-1), 6),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#93C5FD'))
        ]))
        elements.append(lit_box)
        elements.append(Spacer(1, 15))

        # Disclaimer Box
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#991B1B'), spaceAfter=8))
        disc_box = Table([
            [Paragraph(f"<b>IMPORTANT MEDICAL DISCLAIMER:</b> {MEDICAL_DISCLAIMER}", disclaimer_style)]
        ], colWidths=[540])
        disc_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEF2F2')),
            ('PADDING', (0,0), (-1,-1), 8),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#FCA5A5'))
        ]))
        elements.append(disc_box)

        doc.build(elements)
        return filename, filepath

    def generate_csv_report(self, records_list):
        """
        Generates CSV format string from list of prediction records.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Timestamp", "Child_Age", "Input_Mode", "Filename", 
            "Detected_Sound", "Confidence", "Top_Pattern", "Risk_Score", 
            "Risk_Level"
        ])
        
        for r in records_list:
            writer.writerow([
                r.get("created_at", datetime.now().isoformat()),
                r.get("age_group", "0-12m"),
                r.get("source", "UNKNOWN"),
                r.get("filename", "N/A"),
                r.get("sound_type", "UNKNOWN"),
                f"{r.get('confidence', 0.0):.4f}",
                r.get("top_pattern", "N/A"),
                r.get("risk_score", 0),
                r.get("risk_level", "LOW")
            ])
            
        return output.getvalue()
