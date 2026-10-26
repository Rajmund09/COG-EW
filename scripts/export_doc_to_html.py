import docx
import html

doc = docx.Document('COG_EW_Master_Project_Document.docx')

html_parts = []
html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COG-EW Master Project Documentation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.65;
            color: #2D3748;
            background: #F0F4F8;
            margin: 0;
            padding: 40px 20px;
        }
        .container {
            max-width: 980px;
            margin: 0 auto;
            background: #FFFFFF;
            padding: 50px 70px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border-radius: 8px;
        }
        .header-title {
            color: #1A365D;
            font-size: 26px;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 6px;
        }
        .meta-subtitle {
            color: #718096;
            font-size: 14px;
            margin-bottom: 28px;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 12px;
        }
        h1 {
            color: #1A365D;
            border-bottom: 2px solid #3182CE;
            padding-bottom: 8px;
            margin-top: 36px;
            font-size: 20px;
        }
        h2 {
            color: #2B6CB0;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 6px;
            margin-top: 28px;
            font-size: 17px;
        }
        h3 {
            color: #2D3748;
            margin-top: 20px;
            font-size: 15px;
        }
        p {
            margin: 10px 0;
            font-size: 14.5px;
        }
        .callout-green {
            background: #F0FDF4;
            border-left: 5px solid #16A34A;
            padding: 16px 20px;
            border-radius: 4px;
            margin: 20px 0;
            font-size: 14px;
            color: #166534;
        }
        .callout-blue {
            background: #EDF2F7;
            border-left: 5px solid #2B6CB0;
            padding: 16px 20px;
            border-radius: 4px;
            margin: 20px 0;
            font-family: 'Consolas', 'Courier New', monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.5;
            color: #1A202C;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13.5px;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #E2E8F0;
        }
        th {
            background: #1A365D;
            color: #FFFFFF;
            font-weight: 600;
        }
        tr:nth-child(even) td {
            background: #F8FAFC;
        }
        .viva-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 14px 18px;
            margin: 12px 0;
        }
        .viva-q {
            font-weight: 700;
            color: #1A365D;
            margin-bottom: 6px;
        }
        .viva-a {
            color: #4A5568;
            font-size: 14px;
        }
        .nav-doc-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #1A365D;
            color: #FFFFFF;
            padding: 10px 18px;
            border-radius: 6px;
            margin-bottom: 25px;
            font-size: 13px;
        }
        .nav-doc-bar a {
            color: #90CDF4;
            text-decoration: none;
            font-weight: 600;
            margin-left: 14px;
        }
        .nav-doc-bar a:hover {
            text-decoration: underline;
        }
        @media print {
            body { background: #FFFFFF; padding: 0; }
            .container { box-shadow: none; padding: 20px; max-width: 100%; }
            .nav-doc-bar { display: none; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="nav-doc-bar">
        <span>COG-EW Formal Project Documentation · Version 2.4.0</span>
        <div>
            <a href="/index.html">← Back to Mission Deck</a>
            <a href="/pages/simulation.html">3D Tactical Simulation</a>
        </div>
    </div>
    <div class="header-title">COG-EW: Autonomous Cognitive Electronic Warfare &amp; Hypersonic Flight Guidance Co-Designer</div>
    <div class="meta-subtitle">Comprehensive Technical Specification, Research Paper, Complete UI Manual &amp; Viva Examination Guide</div>
''')

from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl

in_viva = False
current_q = None

for child in doc.element.body:
    if isinstance(child, CT_P):
        p = docx.text.paragraph.Paragraph(child, doc)
        txt = p.text.strip()
        if not txt:
            continue
        
        if p.style.name == 'Heading 1':
            html_parts.append(f'<h1>{html.escape(txt)}</h1>\n')
            if 'Viva' in txt:
                in_viva = True
        elif p.style.name == 'Heading 2':
            html_parts.append(f'<h2>{html.escape(txt)}</h2>\n')
        elif p.style.name == 'Heading 3':
            html_parts.append(f'<h3>{html.escape(txt)}</h3>\n')
        elif p.style.name.startswith('List'):
            html_parts.append(f'<ul><li>{html.escape(txt)}</li></ul>\n')
        elif in_viva and txt.startswith('Q') and '?' in txt:
            current_q = txt
        elif in_viva and current_q:
            html_parts.append(f'<div class="viva-card"><div class="viva-q">{html.escape(current_q)}</div><div class="viva-a">{html.escape(txt)}</div></div>\n')
            current_q = None
        else:
            html_parts.append(f'<p>{html.escape(txt)}</p>\n')
            
    elif isinstance(child, CT_Tbl):
        tbl = docx.table.Table(child, doc)
        if len(tbl.rows) == 1 and len(tbl.columns) == 1:
            cell_txt = tbl.rows[0].cells[0].text
            if 'SUICIDE DRONE' in cell_txt or 'CRITICAL STRATEGIC CLARIFICATION' in cell_txt:
                html_parts.append(f'<div class="callout-green">{html.escape(cell_txt)}</div>\n')
            else:
                html_parts.append(f'<div class="callout-blue">{html.escape(cell_txt)}</div>\n')
            continue
        
        html_parts.append('<table>\n<thead><tr>\n')
        for c in tbl.rows[0].cells:
            html_parts.append(f'<th>{html.escape(c.text.strip())}</th>')
        html_parts.append('</tr></thead>\n<tbody>\n')
        for row in tbl.rows[1:]:
            html_parts.append('<tr>\n')
            for c in row.cells:
                html_parts.append(f'<td>{html.escape(c.text.strip())}</td>')
            html_parts.append('</tr>\n')
        html_parts.append('</tbody></table>\n')

html_parts.append('''
</div>
</body>
</html>
''')

with open('dashboard/DOCUMENTATION.html', 'w', encoding='utf-8') as f:
    f.write(''.join(html_parts))

print("dashboard/DOCUMENTATION.html generated successfully!")
