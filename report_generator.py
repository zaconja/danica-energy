"""
Enhanced PDF Report Generator za Danica Energy Optimizer PRO
Poboljšano s modernim layoutom, grafikonima, tablicama i metadata.
"""
import io
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, HRFlowable, KeepTogether, PageBreak
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# -----------------------------------------------------------------------
# PALETA BOJA
# -----------------------------------------------------------------------
class Colors:
    PRIMARY = colors.HexColor('#0B2F4D')
    SECONDARY = colors.HexColor('#1565C0')
    ACCENT = colors.HexColor('#00BCD4')
    SUCCESS = colors.HexColor('#2E7D32')
    WARNING = colors.HexColor('#F57C00')
    DANGER = colors.HexColor('#C62828')
    LIGHT_BG = colors.HexColor('#F8FAFC')
    BORDER = colors.HexColor('#E0E8F0')
    TEXT = colors.HexColor('#1A2634')
    TEXT_MUTED = colors.HexColor('#5F6C80')
    WHITE = colors.white


# -----------------------------------------------------------------------
# GLAVNI RAZRED
# -----------------------------------------------------------------------
class PDFReport:
    """
    Moderni PDF izvještaj s podrškom za:
    - Naslovna stranica s metapodacima
    - Tablice s alternativnim bojanjem redaka
    - Plotly grafikoni (PNG konverzija)
    - Kartiice metrika u rešetki
    - Waterfall i usporedni grafikoni
    - Zaglavlja i podnožja stranica
    """

    def __init__(
        self,
        title: str = "Danica Energy Optimizer – Izvještaj",
        subtitle: str = "",
        author: str = "EKONERG – Institut za energetiku i zaštitu okoliša",
        logo_path: Optional[str] = None,
    ):
        self.title = title
        self.subtitle = subtitle
        self.author = author
        self.logo_path = logo_path
        self.generated_at = datetime.now()

        self.buffer = io.BytesIO()
        self.story: List = []

        if REPORTLAB_AVAILABLE:
            self.doc = SimpleDocTemplate(
                self.buffer,
                pagesize=A4,
                rightMargin=2.2 * cm,
                leftMargin=2.2 * cm,
                topMargin=2.5 * cm,
                bottomMargin=2.5 * cm,
                title=title,
                author=author,
                subject="Energy Optimization Report",
                creator="Danica Energy Optimizer PRO v6.0",
            )
            self._init_styles()

    def _init_styles(self):
        base = getSampleStyleSheet()

        self.styles = {
            'Title': ParagraphStyle(
                'CustomTitle',
                fontName='Helvetica-Bold',
                fontSize=22,
                textColor=Colors.PRIMARY,
                spaceAfter=6,
                alignment=TA_LEFT,
                leading=28,
            ),
            'Subtitle': ParagraphStyle(
                'CustomSubtitle',
                fontName='Helvetica',
                fontSize=12,
                textColor=Colors.TEXT_MUTED,
                spaceAfter=18,
                alignment=TA_LEFT,
            ),
            'H2': ParagraphStyle(
                'H2',
                fontName='Helvetica-Bold',
                fontSize=14,
                textColor=Colors.PRIMARY,
                spaceBefore=14,
                spaceAfter=6,
                borderPad=4,
            ),
            'H3': ParagraphStyle(
                'H3',
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=Colors.SECONDARY,
                spaceBefore=10,
                spaceAfter=4,
            ),
            'Body': ParagraphStyle(
                'Body',
                fontName='Helvetica',
                fontSize=9.5,
                textColor=Colors.TEXT,
                spaceAfter=6,
                leading=14,
            ),
            'Caption': ParagraphStyle(
                'Caption',
                fontName='Helvetica-Oblique',
                fontSize=8.5,
                textColor=Colors.TEXT_MUTED,
                spaceAfter=4,
                alignment=TA_CENTER,
            ),
            'Meta': ParagraphStyle(
                'Meta',
                fontName='Helvetica',
                fontSize=8,
                textColor=Colors.TEXT_MUTED,
                spaceAfter=2,
                alignment=TA_RIGHT,
            ),
        }

    # ----------------------------------------------------------------
    # NASLOVNA STRANICA
    # ----------------------------------------------------------------
    def add_title_page(self):
        if not REPORTLAB_AVAILABLE:
            return

        # Horizontalna linija
        self.story.append(HRFlowable(
            width="100%", thickness=3, color=Colors.PRIMARY,
            spaceAfter=20, spaceBefore=10,
        ))

        # Naslov
        self.story.append(Paragraph(self.title, self.styles['Title']))
        if self.subtitle:
            self.story.append(Paragraph(self.subtitle, self.styles['Subtitle']))

        # Metadata tablica
        meta_data = [
            ['Datum i vrijeme:', self.generated_at.strftime('%d.%m.%Y %H:%M')],
            ['Autor:', self.author],
            ['Verzija:', 'Danica Energy Optimizer PRO v6.0'],
        ]
        meta_table = Table(meta_data, colWidths=[4.5 * cm, 12 * cm])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), Colors.TEXT_MUTED),
            ('TEXTCOLOR', (1, 0), (1, -1), Colors.TEXT),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        self.story.append(meta_table)

        self.story.append(HRFlowable(
            width="100%", thickness=1, color=Colors.BORDER,
            spaceAfter=16, spaceBefore=16,
        ))

    # ----------------------------------------------------------------
    # LEGACY metode (backwards compat)
    # ----------------------------------------------------------------
    def add_title(self):
        self.add_title_page()

    def add_heading(self, text: str, level: int = 2):
        if not REPORTLAB_AVAILABLE:
            return
        style = self.styles['H2'] if level <= 2 else self.styles['H3']
        self.story.append(Paragraph(text, style))

    def add_paragraph(self, text: str):
        if not REPORTLAB_AVAILABLE:
            return
        self.story.append(Paragraph(text, self.styles['Body']))

    def add_spacer(self, height_cm: float = 0.5):
        if REPORTLAB_AVAILABLE:
            self.story.append(Spacer(1, height_cm * cm))

    def add_page_break(self):
        if REPORTLAB_AVAILABLE:
            self.story.append(PageBreak())

    # ----------------------------------------------------------------
    # METRIKE – GRID KARTIICE
    # ----------------------------------------------------------------
    def add_metric_cards(self, metrics: Dict[str, Any], cols: int = 3):
        if not REPORTLAB_AVAILABLE:
            return

        items = list(metrics.items())
        rows = [items[i:i+cols] for i in range(0, len(items), cols)]
        col_width = (A4[0] - 4.4 * cm) / cols

        for row in rows:
            table_data = []
            # Labels
            label_row = []
            val_row = []
            for label, value in row:
                label_row.append(Paragraph(
                    str(label),
                    ParagraphStyle('ml', fontName='Helvetica', fontSize=7.5,
                                   textColor=Colors.TEXT_MUTED, alignment=TA_CENTER)
                ))
                # Format value
                if isinstance(value, float):
                    if abs(value) >= 1e6:
                        val_str = f"{value/1e6:.2f}M"
                    elif abs(value) >= 1e3:
                        val_str = f"{value/1e3:.1f}k"
                    else:
                        val_str = f"{value:.1f}"
                else:
                    val_str = str(value)

                val_row.append(Paragraph(
                    val_str,
                    ParagraphStyle('mv', fontName='Helvetica-Bold', fontSize=13,
                                   textColor=Colors.PRIMARY, alignment=TA_CENTER)
                ))

            # Pad row if needed
            while len(label_row) < cols:
                label_row.append(Paragraph('', self.styles['Body']))
                val_row.append(Paragraph('', self.styles['Body']))

            table_data = [label_row, val_row]
            t = Table(table_data, colWidths=[col_width] * cols)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), Colors.LIGHT_BG),
                ('BOX', (0, 0), (-1, -1), 0.5, Colors.BORDER),
                ('INNERGRID', (0, 0), (-1, -1), 0.3, Colors.BORDER),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('ROUNDEDCORNERS', [4]),
            ]))
            self.story.append(t)
            self.story.append(Spacer(1, 0.3 * cm))

    # ----------------------------------------------------------------
    # TABLICA PODATAKA
    # ----------------------------------------------------------------
    def add_dataframe(
        self,
        df: pd.DataFrame,
        caption: str = "",
        max_rows: int = 50,
        highlight_col: Optional[str] = None,
    ):
        if not REPORTLAB_AVAILABLE:
            return

        if caption:
            self.story.append(Paragraph(caption, self.styles['H3']))

        df_show = df.head(max_rows)
        avail_width = A4[0] - 4.4 * cm
        n_cols = len(df_show.columns)
        col_width = avail_width / n_cols

        # Header + data
        header = [Paragraph(str(c), ParagraphStyle(
            'th', fontName='Helvetica-Bold', fontSize=8,
            textColor=Colors.WHITE, alignment=TA_CENTER
        )) for c in df_show.columns]

        data_rows = []
        for _, row in df_show.iterrows():
            cells = []
            for val in row:
                if isinstance(val, float):
                    txt = f"{val:.2f}"
                else:
                    txt = str(val)
                cells.append(Paragraph(txt, ParagraphStyle(
                    'td', fontName='Helvetica', fontSize=7.5,
                    textColor=Colors.TEXT, alignment=TA_CENTER
                )))
            data_rows.append(cells)

        table_data = [header] + data_rows
        t = Table(table_data, colWidths=[col_width] * n_cols, repeatRows=1)

        style = [
            ('BACKGROUND', (0, 0), (-1, 0), Colors.PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), Colors.WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.3, Colors.BORDER),
        ]
        # Zebra striping
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                style.append(('BACKGROUND', (0, i), (-1, i), Colors.LIGHT_BG))

        t.setStyle(TableStyle(style))
        self.story.append(t)
        self.story.append(Spacer(1, 0.4 * cm))

    # ----------------------------------------------------------------
    # PLOTLY GRAFIKONI
    # ----------------------------------------------------------------
    def add_plotly_chart(
        self,
        fig,
        caption: str = "",
        width: int = 700,
        height: int = 350,
    ):
        if not REPORTLAB_AVAILABLE or not PLOTLY_AVAILABLE:
            return

        if caption:
            self.story.append(Paragraph(caption, self.styles['H3']))

        try:
            img_bytes = pio.to_image(fig, format='png', width=width, height=height, scale=2)
            img_buffer = io.BytesIO(img_bytes)
            avail_width = A4[0] - 4.4 * cm
            aspect = height / width
            img = Image(img_buffer, width=avail_width, height=avail_width * aspect)
            self.story.append(img)
            self.story.append(Spacer(1, 0.4 * cm))
        except Exception as e:
            self.story.append(Paragraph(f"[Grafikon nije dostupan: {e}]", self.styles['Caption']))

    # ----------------------------------------------------------------
    # HORIZONTALNA LINIJA RAZDJELNIKA
    # ----------------------------------------------------------------
    def add_divider(self):
        if REPORTLAB_AVAILABLE:
            self.story.append(HRFlowable(
                width="100%", thickness=0.5, color=Colors.BORDER,
                spaceAfter=8, spaceBefore=8,
            ))

    # ----------------------------------------------------------------
    # GENERIRANJE PDF-a
    # ----------------------------------------------------------------
    def save(self) -> bytes:
        if not REPORTLAB_AVAILABLE:
            return b""

        def _on_page(canvas, doc):
            canvas.saveState()
            # Footer
            canvas.setFont('Helvetica', 7.5)
            canvas.setFillColor(Colors.TEXT_MUTED)
            canvas.drawString(
                2.2 * cm,
                1.5 * cm,
                f"{self.author}  |  {self.generated_at.strftime('%d.%m.%Y %H:%M')}"
            )
            canvas.drawRightString(
                A4[0] - 2.2 * cm,
                1.5 * cm,
                f"Stranica {doc.page}"
            )
            # Header line
            canvas.setStrokeColor(Colors.ACCENT)
            canvas.setLineWidth(1.5)
            canvas.line(2.2 * cm, A4[1] - 1.8 * cm, A4[0] - 2.2 * cm, A4[1] - 1.8 * cm)
            canvas.restoreState()

        self.doc.build(self.story, onFirstPage=_on_page, onLaterPages=_on_page)
        return self.buffer.getvalue()
