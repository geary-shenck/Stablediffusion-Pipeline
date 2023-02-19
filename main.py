
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

import torch
from diffusers import StableDiffusionPipeline

import base64
from io import BytesIO

#model
sd_pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",revision="fp16", torch_dtype=torch.float16
)

sd_pipeline.to("cuda")

#flask and ngrok
web_app = Flask(__name__)
run_with_ngrok(web_app)

@web_app.route("/")
def initial():
    return render_template("index.html")

@web_app.route("/submit-prompt", methods=["POST"])
def generate_image():
    prompt_user = request.form["prompt-input"]
    print(f"generating {prompt_user}")

    image_returned = sd_pipeline(prompt_user).images[0]
    print("generated")

    buffered_img = BytesIO()
    image_returned.save(buffered_img, format="PNG")
    image_string = base64.b64encode(buffered_img.getvalue())
    fin_string= "data:image/png;base64," + str(image_string)[2:-1]
    print("sent")

    return render_template("index.html",generate_image = fin_string)

if __name__ == "__main__":
    web_app.run()