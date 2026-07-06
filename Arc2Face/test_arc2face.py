"""
测试脚本：用 Arc2Face 对 forget/retain 身份生成图像
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis

# ========== 配置 ==========
BASE_MODEL = "/data/weight"  # 本地 SD1.5 基础模型路径
MODELS_DIR = "./models"
CELEBA_IMG_DIR = "../Data/CelebA/Img/img_align_celeba"
OUTPUT_DIR = "./outputs/test_generations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 验证集
VALIDATION_SETS = {
    "Face_Set_1": {"forget": 3422, "retain": [5230, 5239, 1539]},
    "Face_Set_2": {"forget": 3376, "retain": [3602, 608, 7405]},
}

# 从 identity_CelebA.txt 获取身份对应的图片
def get_identity_images(identity_file):
    """返回 {identity_id: [img_filename, ...]}"""
    id_map = {}
    with open(identity_file) as f:
        for line in f:
            parts = line.strip().split()
            img = parts[0]
            identity = int(parts[1])
            id_map.setdefault(identity, []).append(img)
    return id_map


def load_pipeline(device="cuda"):
    """加载 Arc2Face pipeline"""
    print("Loading Arc2Face pipeline...")

    encoder = CLIPTextModelWrapper.from_pretrained(
        MODELS_DIR, subfolder="encoder", torch_dtype=torch.float16
    )

    unet = UNet2DConditionModel.from_pretrained(
        MODELS_DIR, subfolder="arc2face", torch_dtype=torch.float16
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    print("Pipeline loaded.")
    return pipeline


def load_face_analyzer(device="cuda"):
    """加载 InsightFace 人脸检测和分析器"""
    app = FaceAnalysis(
        name="antelopev2", root="./",  # root="./" 最终路径为 ./models/antelopev2
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def extract_face_embedding(app, image_path):
    """从图片提取 ArcFace embedding"""
    img = np.array(Image.open(image_path))[:, :, ::-1]  # RGB -> BGR
    faces = app.get(img)
    if len(faces) == 0:
        return None
    # 选最大的人脸
    faces = sorted(faces, key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]))
    face = faces[-1]
    emb = torch.tensor(face["embedding"], dtype=torch.float16)
    emb = emb / torch.norm(emb, dim=0, keepdim=True)  # 归一化
    return emb


def generate_faces(pipeline, id_emb, num_images=4, steps=25, guidance=3.0, seed=42):
    """用 Arc2Face 生成人脸"""
    id_emb = project_face_embs(pipeline, id_emb.unsqueeze(0).to(pipeline.device))

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    images = pipeline(
        prompt_embeds=id_emb,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images
    return images


def main():
    device = "cuda"
    identity_file = os.path.join("../Data/CelebA/Anno", "identity_CelebA.txt")

    # 加载映射
    id_map = get_identity_images(identity_file)

    # 加载模型
    pipeline = load_pipeline(device)
    app = load_face_analyzer(device)

    for set_name, identities in VALIDATION_SETS.items():
        forget_id = identities["forget"]
        retain_ids = identities["retain"]
        all_ids = [forget_id] + retain_ids

        print(f"\n{'='*60}")
        print(f"Processing {set_name}")
        print(f"  Forget ID: {forget_id},  Retain IDs: {retain_ids}")
        print(f"{'='*60}")

        for identity_id in all_ids:
            role = "FORGET" if identity_id == forget_id else "RETAIN"
            img_files = id_map.get(identity_id, [])

            if not img_files:
                print(f"  [{role}] ID {identity_id}: No images found!")
                continue

            # 用第一张图片提取 embedding
            img_path = os.path.join(CELEBA_IMG_DIR, img_files[0])
            print(f"  [{role}] ID {identity_id}: Using {img_files[0]}")

            emb = extract_face_embedding(app, img_path)
            if emb is None:
                print(f"    WARNING: No face detected!")
                continue

            # 生成图像
            images = generate_faces(pipeline, emb, num_images=2)

            # 保存
            set_out = os.path.join(OUTPUT_DIR, set_name, f"ID_{identity_id}_{role}")
            os.makedirs(set_out, exist_ok=True)

            # 保存源图片
            Image.open(img_path).save(os.path.join(set_out, "source.jpg"))

            # 保存生成图
            for i, img in enumerate(images):
                img.save(os.path.join(set_out, f"gen_{i+1}.jpg"))

            print(f"    Saved {len(images)} generated images to {set_out}")

    print(f"\n✅ Done! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
