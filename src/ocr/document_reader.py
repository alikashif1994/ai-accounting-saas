# src/ocr/document_reader.py
# Reads uploaded images and PDFs, returns plain text

import pytesseract
from PIL import Image
import io
from typing import Optional

# WINDOWS ONLY: tell pytesseract where Tesseract is installed
# Comment this out on Mac/Linux
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image: Image.Image) -> Image.Image:
    '''
    Clean the image before OCR to improve accuracy.
    These steps make text clearer and easier for Tesseract to read.
    '''
    image = image.convert('L')

    width, height = image.size
    if width < 600:
        scale = 600 / width
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.LANCZOS
        )
    return image


def read_document(file_bytes: bytes, filename: str) -> dict:
    '''
    Main function: takes uploaded file bytes and returns extracted text as a dict.
    Works with: JPEG, PNG, TIFF, BMP image files.

    file_bytes: the raw file data from the client upload
    filename:   original filename (used to detect file type)

    Returns dict with keys: text, confidence, word_count, success, entry_type
    '''
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = preprocess_image(image)

        raw_text = pytesseract.image_to_string(
            image, lang='eng', config='--psm 6'
        )

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

        return {
            "text":       raw_text.strip(),
            "confidence": round(avg_conf, 2),
            "word_count": word_count,
            "success":    word_count >= 3,
            "entry_type": "ocr",
        }

    except Exception as e:
        return {
            "text":          "",
            "confidence":    0.0,
            "word_count":    0,
            "success":       False,
            "entry_type":    "ocr",
            "error_message": str(e),
        }