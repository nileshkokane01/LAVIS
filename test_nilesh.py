import torch

import numpy as np

from PIL import Image
from lavis.models import load_model_and_preprocess
from lavis.models.blip_diffusion_models.utils import preprocess_canny

torch.cuda.is_available()

model, vis_preprocess, txt_preprocess = load_model_and_preprocess(
    "blip_diffusion", "canny", device="cuda", is_eval=True)


def generate_canny(cond_image_input, low_threshold, high_threshold):
    # convert cond_image_input to numpy array
    cond_image_input = np.array(cond_image_input).astype(np.uint8)

    # canny_input, vis_control_image = preprocess_canny(cond_image_input, 512, low_threshold=100, high_threshold=200)
    vis_control_image = preprocess_canny(cond_image_input, 512, low_threshold=low_threshold, high_threshold=high_threshold)

    return vis_control_image 



style_subject = "flower" # subject that defines the style
tgt_subject = "teapot"  # subject to generate.

text_prompt = "on a marble table"

cond_subjects = [txt_preprocess["eval"](style_subject)]
tgt_subjects = [txt_preprocess["eval"](tgt_subject)]
text_prompt = [txt_preprocess["eval"](text_prompt)]

cldm_cond_image = Image.open("../images/kettle.jpg").convert("RGB")

style_image = Image.open("../images/flower.jpg").convert("RGB")

style_image = vis_preprocess["eval"](style_image).unsqueeze(0).cuda()


canny_low_threshold = 30
canny_high_threshold = 70

cond_image_input = generate_canny(cldm_cond_image, canny_low_threshold, canny_high_threshold)

cond_image_display = cond_image_input.resize((256, 256), resample=Image.BILINEAR)


samples = {
    "cond_images": style_image,
    "cond_subject": cond_subjects,
    "tgt_subject": tgt_subjects,
    "prompt": text_prompt,
    "cldm_cond_image": cond_image_input.convert("RGB"),
}

num_output = 2

iter_seed = 88888
guidance_scale = 7.5
num_inference_steps = 50
negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


for i in range(num_output):
    output = model.generate(
        samples,
        seed=iter_seed + i,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
    )
