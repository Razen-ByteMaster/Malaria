from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI(title="Malaria Prediction API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class names
class_names = ['Malaria', 'No Malaria']
# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("malaria.pth", map_location=torch.device('cpu')))
model.eval()
model.to(device)


transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict_malaria(file: UploadFile = File(...)):
    try:
        # Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        img_tensor = transforms(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx])

        return JSONResponse({
            "filename": file.filename,
            "prediction": pred_class,
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})