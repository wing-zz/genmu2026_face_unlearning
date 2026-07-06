"""
推理脚本：加载遗忘后的模型，对指定身份生成图像
"""
import os, sys, torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis

BASE_MODEL = "/data/weight"  # 修改为你的 SD1.5 本地路径
MODELS_DIR = "./models"
UNLEARNED_PATH = "./outputs/unlearning/Face_Set_1/final"  # 或 Face_Set_2/final

def load_pipeline(unet_path=None):
    encoder = CLIPTextModelWrapper.from_pretrained(MODELS_DIR, subfolder="encoder")
    if unet_path:
        print(f"Loading unlearned UNet from {unet_path}")
        unet = UNet2DConditionModel.from_pretrained(unet_path)
    else:
        unet = UNet2DConditionModel.from_pretrained(MODELS_DIR, subfolder="arc2face")
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, text_encoder=encoder, unet=unet, safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to("cuda")
    return pipeline

def extract_embedding(app, image_path):
    img = np.array(Image.open(image_path))[:, :, ::-1]
    faces = app.get(img)
    if not faces:
        return None
    faces = sorted(faces, key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]))
    emb = torch.tensor(faces[-1]["embedding"]).cuda()
    return emb / torch.norm(emb)

def generate(pipeline, face_emb, num_images=4, seed=42):
    prompt_emb = project_face_embs(pipeline, face_emb.unsqueeze(0))
    images = []
    for i in range(num_images):
        g = torch.Generator(device="cuda").manual_seed(seed + i)
        imgs = pipeline(prompt_embeds=prompt_emb, num_inference_steps=25,
                        guidance_scale=3.0, num_images_per_prompt=1, generator=g).images
        images.append(imgs[0])
    return images

if __name__ == "__main__":
    app = FaceAnalysis(name="antelopev2", root="./", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    pipeline = load_pipeline(UNLEARNED_PATH)
    emb = extract_embedding(app, "path/to/source_image.jpg")
    images = generate(pipeline, emb)
    for i, img in enumerate(images):
        img.save(f"output_{i+1}.png")
    print(f"Generated {len(images)} images")
