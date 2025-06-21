import streamlit as st
import torch
import numpy as np
from model import CVAE
import matplotlib.pyplot as plt

device = torch.device("cpu")

st.title("MNIST Handwritten Digit Generator")
st.markdown("Select a digit (0–9) to generate 5 synthetic handwritten samples.")

digit = st.selectbox("Choose a digit to generate:", list(range(10)))

model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

def generate_images(label, num=5):
    with torch.no_grad():
        z = torch.randn(num, model.latent_dim)
        y = torch.tensor([label] * num)
        outputs = model.decode(z, y).view(-1, 28, 28)
    return outputs.numpy()

if st.button("Generate"):
    images = generate_images(digit)
    cols = st.columns(5)
    for i in range(5):
        cols[i].image(images[i], width=100, caption=f"Sample {i+1}")
