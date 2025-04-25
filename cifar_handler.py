import torch
import torch.nn as nn
import cv2
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
from model import SimpleCNN

class Cifar10Handler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.categories = ["airplane", "auto", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
        self.softmax = nn.Softmax(dim=1)
        self.initialized = False

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(10)

        checkpoint = torch.load(context.manifest["model"]["serializedFile"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self,data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image,str):
                image = np.frombuffer(image.encode(),dtype=np.uint8)
            elif isinstance(image,(bytes,bytearray)):
                image = np.frombuffer(image,dtype=np.uint8)
            else:
                raise ValueError("Unsupported image format")
            image = cv2.imdecode(image,cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(32,32))
            image = np.transpose(image,(2,0,1)) / 255.
            image = torch.from_numpy(image).float()
            images.append(image)
        return torch.stack(images).to(self.device)
    
    def inference(self,data):
        with torch.no_grad():
            results = self.model(data)
            probs = self.softmax(results)
            return probs
        
    def postprocess(self,data):
        results = []
        for probs in data:
            pred_idx = torch.argmax(probs).item()
            class_name = self.categories[pred_idx]
            confidence = probs[pred_idx].item() * 100
            results.append({
                "class": class_name,
                "confidence": confidence
            })
        return results 
            
