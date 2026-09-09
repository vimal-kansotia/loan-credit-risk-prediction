import re
import os
import markdown
from xhtml2pdf import pisa

def convert_report_to_pdf():
    with open('RESEARCH_PROJECT_REPORT.md', 'r') as f:
        md_content = f.read()

    # Split off the cover page (before the first \newpage)
    parts = md_content.split('\\newpage', 1)
    
    # Custom HTML for the executive cover page
    cover_html = """
    <div class="cover-container">
        <div class="cover-header">M.Sc BIG DATA ANALYTICS</div>
        <div class="cover-sub">Research Project Report</div>
        <div class="cover-line"></div>
        
        <div class="cover-title">
            LOAN CREDIT RISK &amp; DEFAULT PREDICTION SYSTEM USING MACHINE LEARNING
        </div>
        
        <div class="cover-domain">
            <b>Domain:</b> Banking
        </div>
        
        <div class="cover-meta-section">
            <div class="cover-meta-label">Submitted by</div>
            <div class="cover-meta-value"><b>Vimal Kansotia</b></div>
            <div class="cover-meta-sub">Roll No. / UID: 2509038</div>
        </div>
        
        <div class="cover-meta-section">
            <div class="cover-meta-label">Under the guidance of</div>
            <div class="cover-meta-value"><b>Prof. Ameya Chitnis</b></div>
        </div>
        
        <div class="cover-date">
            September, 2026
        </div>
    </div>
    <div style="page-break-before: always;"></div>
    """

    body_md = parts[1] if len(parts) > 1 else ""

    # Pre-process \newpage into page break divs
    body_md = body_md.replace('\\newpage', '<div style="page-break-before: always;"></div>')
    # Replace math blocks with clean formatted divs
    body_md = re.sub(r'\$\$(.*?)\$\$', r'<div class="equation">\1</div>', body_md, flags=re.DOTALL)
    # Clean up single inline dollar signs inside text like \tau = 0.36
    body_md = re.sub(r'\$(.*?)\$', r'<i>\1</i>', body_md)

    body_html = markdown.markdown(body_md, extensions=['tables', 'fenced_code'])

    # Fix image sources to absolute local paths for xhtml2pdf
    cur_dir = os.path.abspath('.')
    body_html = body_html.replace('src="figures/', f'src="{cur_dir}/figures/')

    custom_css = """
    <style>
        @page {
            size: a4 portrait;
            margin-top: 2.0cm;
            margin-bottom: 2.0cm;
            margin-left: 3.18cm; /* 1.25 inches reserved for spiral binding */
            margin-right: 2.54cm;
        }
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.45;
            color: #1e293b;
        }
        .cover-container {
            text-align: center;
            padding-top: 20px;
        }
        .cover-header {
            font-size: 18pt;
            font-weight: bold;
            color: #0f172a;
            letter-spacing: 1.5px;
            margin-bottom: 4px;
        }
        .cover-sub {
            font-size: 12pt;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 14px;
        }
        .cover-line {
            width: 60%;
            margin: 0 auto 24px auto;
            border-top: 2px solid #2563eb;
        }
        .cover-title {
            font-size: 19pt;
            font-weight: bold;
            color: #1e3a8a;
            line-height: 1.35;
            margin-bottom: 16px;
            padding: 0 10px;
        }
        .cover-domain {
            font-size: 11.5pt;
            color: #334155;
            margin-bottom: 28px;
        }
        .cover-meta-section {
            margin-bottom: 22px;
        }
        .cover-meta-label {
            font-size: 10pt;
            color: #64748b;
            margin-bottom: 2px;
        }
        .cover-meta-value {
            font-size: 13.5pt;
            color: #0f172a;
            margin-bottom: 2px;
        }
        .cover-meta-sub {
            font-size: 10.5pt;
            color: #334155;
        }
        .cover-date {
            font-size: 11.5pt;
            font-weight: bold;
            color: #1e293b;
            margin-top: 30px;
        }
        h1 {
            font-size: 17pt;
            color: #0f172a;
            border-bottom: 1.5px solid #2563eb;
            padding-bottom: 4px;
            margin-top: 18px;
            margin-bottom: 12px;
            page-break-after: avoid;
        }
        h2 {
            font-size: 13pt;
            color: #1e3a8a;
            margin-top: 14px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }
        h3 {
            font-size: 11pt;
            color: #334155;
            margin-top: 12px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }
        h4 {
            font-size: 10pt;
            color: #475569;
            margin-top: 8px;
            margin-bottom: 4px;
            page-break-after: avoid;
        }
        p {
            margin-bottom: 8px;
            text-align: justify;
        }
        ul, ol {
            margin-left: 18px;
            margin-bottom: 8px;
        }
        li {
            margin-bottom: 4px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            margin-bottom: 14px;
            font-size: 8pt;
        }
        th {
            background-color: #1e293b;
            color: #ffffff;
            font-weight: bold;
            padding: 5px 6px;
            border: 1px solid #334155;
            text-align: left;
        }
        td {
            padding: 4px 6px;
            border: 1px solid #cbd5e1;
            vertical-align: middle;
        }
        tr:nth-child(even) {
            background-color: #f8fafc;
        }
        pre {
            background-color: #f1f5f9;
            border: 1px solid #cbd5e1;
            padding: 8px;
            font-family: Courier, monospace;
            font-size: 8pt;
            line-height: 1.25;
            margin-bottom: 10px;
        }
        code {
            font-family: Courier, monospace;
            font-size: 8.5pt;
            background-color: #f1f5f9;
            padding: 1px 3px;
        }
        img {
            max-width: 92%;
            height: auto;
            margin: 8px auto;
            display: block;
            border: 1px solid #e2e8f0;
        }
        .equation {
            background-color: #f8fafc;
            border-left: 3px solid #2563eb;
            padding: 6px 10px;
            margin: 8px 0;
            font-family: 'Times New Roman', serif;
            font-style: italic;
        }
        hr {
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 14px 0;
        }
    </style>
    """

    full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{custom_css}</head><body>{cover_html}{body_html}</body></html>"

    with open('RESEARCH_PROJECT_REPORT.pdf', 'wb') as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        print("PDF conversion completed with errors.")
    else:
        print("RESEARCH_PROJECT_REPORT.pdf generated successfully!")

    cover_single = cover_html.replace('<div style="page-break-before: always;"></div>', '')
    standalone_cover_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{custom_css}</head><body>{cover_single}</body></html>"
    with open('RESEARCH_PROJECT_COVER_PAGE.pdf', 'wb') as cov_file:
        pisa.CreatePDF(standalone_cover_html, dest=cov_file)
    print("RESEARCH_PROJECT_COVER_PAGE.pdf generated successfully!")

if __name__ == '__main__':
    convert_report_to_pdf()
