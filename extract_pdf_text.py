from pypdf import PdfReader
import sys

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf_text.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    try:
        reader = PdfReader(pdf_path)
        text = ""
        links = []
        for page in reader.pages:
            text += page.extract_text() + "\n"
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    obj = annot.get_object()
                    if "/A" in obj and "/URI" in obj["/A"]:
                        links.append(obj["/A"]["/URI"])
        
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n\n--- LINKS ---\n")
            for link in links:
                f.write(link + "\n")
                
        print("Text and links extracted to extracted_text.txt")
    except Exception as e:
        print(f"Error extracting text: {e}")
