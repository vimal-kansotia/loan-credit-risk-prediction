import re
import os
import markdown
from xhtml2pdf import pisa

def convert_report_to_pdf():
    with open('RESEARCH_PROJECT_REPORT.md', 'r') as f:
        md_content = f.read()

    # Pre-process \newpage into page break divs
    md_content = md_content.replace('\\newpage', '<div style="page-break-before: always;"></div>')
    # Replace math blocks with clean formatted divs
    md_content = re.sub(r'\$\$(.*?)\$\$', r'<div class="equation">\1</div>', md_content, flags=re.DOTALL)
    # Clean up single inline dollar signs inside text like \tau = 0.36
    md_content = re.sub(r'\$(.*?)\$', r'<i>\1</i>', md_content)

    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    # Fix image sources to absolute local paths for xhtml2pdf
    cur_dir = os.path.abspath('.')
    html_body = html_body.replace('src="figures/', f'src="{cur_dir}/figures/')

    custom_css = """
    <style>
        @page {
            size: a4 portrait;
            margin-top: 2.5cm;
            margin-bottom: 2.5cm;
            margin-left: 3.2cm; /* 1.25 inches for spiral binding */
            margin-right: 2.5cm;
        }
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.45;
            color: #1e293b;
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

    full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{custom_css}</head><body>{html_body}</body></html>"

    with open('RESEARCH_PROJECT_REPORT.pdf', 'wb') as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        print("PDF conversion completed with errors.")
    else:
        print("RESEARCH_PROJECT_REPORT.pdf generated successfully!")

if __name__ == '__main__':
    convert_report_to_pdf()
