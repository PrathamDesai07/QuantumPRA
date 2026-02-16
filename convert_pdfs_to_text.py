import os
from PyPDF2 import PdfReader

# Define paths
docs_folder = "/teamspace/studios/this_studio/QuantumPRA/docs"
output_folder = "/teamspace/studios/this_studio/QuantumPRA/txtDocs"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all PDF files from docs folder
pdf_files = [f for f in os.listdir(docs_folder) if f.endswith('.pdf')]

print(f"Found {len(pdf_files)} PDF files to convert...\n")

# Convert each PDF to text
for pdf_file in pdf_files:
    pdf_path = os.path.join(docs_folder, pdf_file)
    
    # Create output text filename (replace .pdf with .txt)
    txt_filename = pdf_file.replace('.pdf', '.txt')
    txt_path = os.path.join(output_folder, txt_filename)
    
    try:
        print(f"Converting: {pdf_file}...")
        
        # Read PDF
        reader = PdfReader(pdf_path)
        
        # Extract text from all pages
        text_content = []
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            text_content.append(f"--- Page {page_num} ---\n{text}\n")
        
        # Write to text file
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write('\n'.join(text_content))
        
        print(f"✓ Successfully converted to: {txt_filename}")
        print(f"  Pages: {len(reader.pages)}\n")
        
    except Exception as e:
        print(f"✗ Error converting {pdf_file}: {str(e)}\n")

print("Conversion complete!")
print(f"Text files saved in: {output_folder}")
