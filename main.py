import torch
import pytesseract
import json
from PIL import Image
from text_classifier_model import TextClassifier, stop_words, boost_keywords, important_words, folder_labels
from sklearn.feature_extraction.text import TfidfVectorizer

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words=stop_words
    )

with open("texts.json", "r") as f:
    vectorizer.fit(json.load(f))

model = TextClassifier()
model.load_state_dict(torch.load("text_classifier_model.pth"))
model.eval()

test_image_text = boost_keywords(pytesseract.image_to_string(Image.open("TestSet/ID_TESTING_1.png")), important_words)

test_image_vector = torch.tensor(vectorizer.transform([test_image_text]).toarray(), dtype=torch.float32)

classes = {v: k for k, v in folder_labels.items()}

with torch.no_grad():
    if test_image_text == "":
        print("No text found for this image")
    else:
        prediction = torch.argmax(model(test_image_vector), dim=1)
        print(f"Prediction for test image: {prediction.item()}")