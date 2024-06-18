import os
import argparse
import random
import gradio as gr
from main import load_models, cache_path
from PIL import Image
from os import path
from requests import Response
from vlm import LLMUtility
import time
import numpy as np
import cv2


canvas_size = 512
llm_obj = LLMUtility()
response = None
counter = 0
save_dir = "output_images"
os.makedirs(save_dir, exist_ok=True)

def form_image(gen_image, sketch_img):
    sketch_img = np.asarray(sketch_img)
    sketch_img = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)
    gen_image = np.asarray(gen_image)
    gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
    h, w, _ = sketch_img.shape
    sep = np.zeros((h, 30, 3), dtype=np.uint8)
    image = np.concatenate([sketch_img, 
                            sep,
                            gen_image], axis=1)
    return Image.fromarray(image)

def score_fn(gen_image, sketch_img, time_thresh=5):
    global llm_obj, response, counter

    image = form_image(gen_image, sketch_img)
    if (counter > time_thresh) or response is None:
      response, rating = llm_obj.get_response_score(image)
      counter = 0
    counter += 1
    return response

def down(name, sketch_img, gen_image, comments):
    global save_dir
    sketch_img = np.asarray(sketch_img)
    sketch_img = cv2.cvtColor(sketch_img, cv2.COLOR_BGR2RGB)
    gen_image = np.asarray(gen_image)
    gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)
    h, w, _ = sketch_img.shape
    sep = np.zeros((h, 30, 3), dtype=np.uint8)

    image = np.concatenate([sketch_img, 
                            sep,
                            gen_image], axis=1)
    os.makedirs(f"{save_dir}/{name}", exist_ok=True)
    cv2.imwrite(f"{save_dir}/{name}/{name}-sketch_img-{time.time()}.png", sketch_img)
    cv2.imwrite(f"{save_dir}/{name}/{name}-gen_img-{time.time()}.png", gen_image)
    with open(f"{save_dir}/{name}/{name}-comments-{time.time()}.txt", "w") as f:
      f.write(comments)
    cv2.imwrite(f"{save_dir}/{name}/{name}-full_img-{time.time()}.png", image)

if not path.exists(cache_path):
    os.makedirs(cache_path, exist_ok=True)

with gr.Blocks() as demo:
    infer = load_models()

    with gr.Column():
        with gr.Row():
            with gr.Column():
                name = gr.Text(
                    label="Name", value="NAME", interactive=True)
                s = gr.Slider(label="steps", minimum=1, maximum=6,
                              step=1, value=3, interactive=True)
                c = gr.Slider(label="cfg", minimum=0.1, maximum=3,
                              step=0.1, value=1.5, interactive=True)
                b = gr.Button(value="download", size="sm")
                
            with gr.Column():
                i_s = gr.Slider(label="sketch strength", minimum=0.1,
                                maximum=0.9, step=0.1, value=0.9, interactive=True)
                t = gr.Text(
                    label="Prompt", value="8K, realistic, colorful art, natural", interactive=True)
            with gr.Column():
                score = gr.Text(
                    label="Judge", value="0/10", interactive=True)
        with gr.Row(equal_height=False):
            i = gr.Image(source="canvas", tool="color-sketch", shape=(canvas_size,
                         canvas_size), width=canvas_size, height=canvas_size, type="pil",
                         show_download_button=True)
            o = gr.Image(width=canvas_size, height=canvas_size)

            def process_image(p, im, steps, cfg, image_strength, seed):
                if not im:
                    return Image.new("RGB", (canvas_size, canvas_size))

                gen_img = infer(
                    prompt=p,
                    image=im,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    strength=image_strength,
                    seed=1337
                )
                score = score_fn(gen_image=gen_img,
                                    sketch_img = im)
                return gen_img, score

            reactive_controls = [t, i, s, c, i_s]

            b.click(down, [name, i, o, score])
            for control in reactive_controls:
                control.change(fn=process_image,
                               inputs=reactive_controls, outputs=[o, score])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # If the option python ui.py --share is attached, it will be deployed to Gradio
    parser.add_argument("--share", action="store_true",
                        help="Deploy on Gradio for sharing", default=False)
    args = parser.parse_args()
    demo.launch(share=args.share)

