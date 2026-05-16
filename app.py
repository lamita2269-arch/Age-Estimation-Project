import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown  # مكتبة لتحميل الملفات من جوجل درايف
import torch.nn.functional as F  

file_id = '1BDcwqVEPklFQNS28DGME5jsc7NUqqscE'
model_path = 'ultra_age_model_final.pth'

@st.cache_resource
def download_and_load_model():
    # إذا لم يكن الملف موجوداً، قم بتحميله
    if not os.path.exists(model_path):
        with st.spinner('جاري تحميل أوزان النموذج من Google Drive... قد يستغرق ذلك لحظات.'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    
    # بناء المعمارية (ResNet-101) كما فعلنا في البحث
    model = models.resnet101(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 101)
    )
    
    # تحميل الأوزان للجهاز (CPU)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = download_and_load_model()

# عمليات المعالجة المسبقة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("اختر صورة وجه...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)

    # التوقع
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_age = torch.sum(probs * torch.arange(0, 101).float(), dim=1).item()

    st.success(f"💡 العمر المتوقع: {pred_age:.1f} سنة")
