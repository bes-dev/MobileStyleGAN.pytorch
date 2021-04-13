import random_face
import gradio as gr 

def mobileface(truncate, alpha):
  engine = random_face.get_engine()
  face = engine.get_random_face(truncate=truncate, alpha=alpha)
  return face[:,:,::-1]

inputs = [
          gr.inputs.Checkbox(label="Truncate"),
          gr.inputs.Slider(minimum=0, maximum=5, step=None, default=0.5, label="Alpha")
          
]

outputs = gr.outputs.Image(type='numpy', label="Output Image")

title = "MobileStylegan"
description = "demo for MobileStylegan. To use it, simply click submit and optionally adjust alpha and truncation values. Read more below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.04767'>MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis</a> | <a href='https://github.com/bes-dev/MobileStyleGAN.pytorch'>Github Repo</a></p>"



gr.Interface(mobileface, inputs, outputs, title=title, description=description, article=article).launch()