import os
import markdown
from playwright.sync_api import sync_playwright

def convert_md_to_pdf(md_path, pdf_path):
    # Read markdown file
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    # Create full HTML document with styling
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
        }}
        h3 {{
            color: #16a085;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    
    # Write temporary HTML file
    temp_html = md_path.replace('.md', '_temp.html')
    with open(temp_html, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    # Convert to PDF using Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        abs_path = os.path.abspath(temp_html)
        page.goto(f'file://{abs_path}')
        page.wait_for_load_state('networkidle')
        page.pdf(path=pdf_path, format='A4', print_background=True, margin={
            'top': '20mm',
            'bottom': '20mm',
            'left': '15mm',
            'right': '15mm'
        })
        browser.close()
    
    # Clean up temp file
    os.remove(temp_html)
    print(f"PDF created successfully: {pdf_path}")

if __name__ == "__main__":
    convert_md_to_pdf('INFORME_RESULTADOS.md', 'INFORME_RESULTADOS.pdf')
