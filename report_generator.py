import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import cm
import io
from datetime import datetime

class PDFReport:
    def __init__(self, title):
        self.buffer = io.BytesIO()
        self.doc = SimpleDocTemplate(self.buffer, pagesize=A4,
                                     rightMargin=2*cm, leftMargin=2*cm,
                                     topMargin=2*cm, bottomMargin=2*cm)
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='CustomTitle',
                                       parent=self.styles['Heading1'],
                                       fontSize=16,
                                       spaceAfter=12,
                                       textColor=colors.HexColor('#0B2F4D')))
        self.styles.add(ParagraphStyle(name='CustomNormal',
                                       parent=self.styles['Normal'],
                                       fontSize=10,
                                       spaceAfter=6))
        self.elements = []
        self.title = title

    def add_title(self):
        self.elements.append(Paragraph(self.title, self.styles['CustomTitle']))
        self.elements.append(Spacer(1, 0.5*cm))
        self.elements.append(Paragraph(f"Datum izvještaja: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
                                       self.styles['CustomNormal']))
        self.elements.append(Spacer(1, 0.5*cm))

    def add_heading(self, text, level=2):
        style = self.styles['Heading2'] if level == 2 else self.styles['Heading3']
        self.elements.append(Paragraph(text, style))
        self.elements.append(Spacer(1, 0.3*cm))

    def add_paragraph(self, text):
        self.elements.append(Paragraph(text, self.styles['CustomNormal']))
        self.elements.append(Spacer(1, 0.2*cm))

    def add_dataframe(self, df, title=None):
        if title:
            self.add_heading(title, 3)
        data = [df.columns.to_list()] + df.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0B2F4D')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.5*cm))

    def add_plotly_chart(self, fig, title=None, width=600, height=350):
        if title:
            self.add_heading(title, 3)
        try:
            img_bytes = fig.to_image(format="png", width=width, height=height)
            img = Image(io.BytesIO(img_bytes), width=width*0.75, height=height*0.75)
            self.elements.append(img)
        except Exception as e:
            # Ako konverzija ne uspije, dodaj tekst umjesto slike
            self.add_paragraph(f"[Grafikon nije moguće generirati: {str(e)}]")
        self.elements.append(Spacer(1, 0.5*cm))

    def add_metric_cards(self, metrics):
        data = [[k, f"{v:,.0f}" if isinstance(v, (int, float)) else str(v)] for k, v in metrics.items()]
        table = Table(data, colWidths=[5*cm, 5*cm])
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        self.elements.append(table)
        self.elements.append(Spacer(1, 0.5*cm))

    def save(self):
        self.doc.build(self.elements)
        self.buffer.seek(0)
        return self.buffer
