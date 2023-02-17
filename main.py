#imports
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

import torch
from diffusers import StableDiffustionPipeline

import base64
from io import BytesIO

#model
sd_pipeline = StableDiffustionPipeline.from_pretrained(
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
    image_returned = sd_pipeline(prompt_user).images[0]

    buffered_img = BytesIO()
    image_returned.save(buffered_img, format="PNG")
    image_string = base64.b64decode(buffered_img.getvalue())
    image_string= "data:image/png;base64," + str(image_string)[2:-1]

    return render_template("index.html",generate_image = image_string)

if __name__ == "__main__":
    web_app.run()