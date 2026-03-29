import pdfplumber

with pdfplumber.open("data/raw/KCET 2021 Cutoff.pdf") as pdf:
    for i, page in enumerate(pdf.pages[:3], 1):
        print(f"\n{'='*60}")
        print(f"PAGE {i} TEXT:")
        print('='*60)
        text = page.extract_text()
        if text:
            for line in text.splitlines():
                print(repr(line))
        else:
            print("(no text)")