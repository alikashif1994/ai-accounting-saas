# src/ocr/document_reader.py
# Reads uploaded images and PDFs, returns plain text


import pytesseract          # OCR engine wrapper
from PIL import Image       # Image loading and processing
import io                   # For handling file bytes from uploads
from dataclasses import dataclass
from typing import Optional


# WINDOWS ONLY: tell pytesseract where Tesseract is installed
# Comment this out on Mac/Linux — it finds Tesseract automatically
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@dataclass
class OCRResult:
    raw_text: str          # All text extracted from the document
    confidence: float      # Average confidence per word (0.0-1.0)
    word_count: int        # Number of words found
    success: bool          # Did we find enough text to work with?
    error_message: Optional[str] = None




def preprocess_image(image: Image.Image) -> Image.Image:
    '''
    Clean the image before OCR to improve accuracy.
    These steps make text clearer and easier for Tesseract to read.
    '''
    # Convert to greyscale — colour is irrelevant for text
    image = image.convert('L')


    # Resize if too small — Tesseract needs at least 300 DPI equivalent
    width, height = image.size
    if width < 600:
        scale = 600 / width
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.LANCZOS  # High quality resizing algorithm
        )
    return image




def read_document(file_bytes: bytes, filename: str) -> OCRResult:
    '''
    Main function: takes uploaded file bytes and returns extracted text.
    Works with: JPEG, PNG, TIFF, BMP image files.


    file_bytes: the raw file data from the client upload
    filename: original filename (used to detect file type)
    '''
    try:
        # Open the image from raw bytes
        image = Image.open(io.BytesIO(file_bytes))
        image = preprocess_image(image)


        # Run Tesseract OCR
        # lang='eng' = English language model
        # --psm 6 = treat as single block of text (best for receipts)
        raw_text = pytesseract.image_to_string(
            image, lang='eng', config='--psm 6'
        )


        # Get per-word confidence scores (Tesseract gives 0-100 per word)
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        confidences = [
            int(c) for c in data['conf']
            if str(c) != '-1' and int(c) >= 0
        ]
        avg_conf = (sum(confidences) / len(confidences) / 100
                    if confidences else 0.0)


        word_count = len([w for w in raw_text.split() if len(w) > 1])


        return OCRResult(
            raw_text=raw_text.strip(),
            confidence=round(avg_conf, 2),
            word_count=word_count,
            success=word_count >= 3  # Minimum 3 words to be useful
        )


    except Exception as e:
        return OCRResult(
            raw_text='', confidence=0.0, word_count=0,
            success=False, error_message=str(e)
        )
