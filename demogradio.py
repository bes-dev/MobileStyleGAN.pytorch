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

outputs = gr.outputs.Image(type='numpy', label="Original Image")

title = "MobileStylegan"
description = "demo for MobileStylegan. To use it, simply click submit and optionally adjust alpha and truncation values. Read more at the links below."
article = "https://raw.githubusercontent.com/AK391/MobileStyleGAN.pytorch/main/README.md"


gr.Interface(mobileface, inputs, outputs, title=title, description=description, article=article).launch()