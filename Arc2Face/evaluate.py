"""
评估脚本：计算遗忘前后的 FA (Forget Accuracy), EA, RA, ERB
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis


# ===================== 配置 =====================
BASE_MODEL = "/data/weight"
MODELS_DIR = "./models"
CELEBA_IMG_DIR = "../Data/CelebA/Img/img_align_celeba"
OUTPUT_DIR = "./outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda"

# 余弦相似度阈值（超过此值认为匹配该身份）
COS_SIM_THRESHOLD = 0.45

# 生成参数
NUM_GEN_IMAGES = 8
GEN_STEPS = 25
GEN_GUIDANCE = 3.0

VALIDATION_SETS = {
    "Face_Set_1": {"forget": 3422, "retain": [5230, 5239, 1539]},
    "Face_Set_2": {"forget": 3376, "retain": [3602, 608, 7405]},
}


def get_identity_images(identity_file="../Data/CelebA/Anno/identity_CelebA.txt"):
    id_map = defaultdict(list)
    with open(identity_file) as f:
        for line in f:
            parts = line.strip().split()
            id_map[int(parts[1])].append(parts[0])
    return dict(id_map)


def load_pipeline(unet_path=None):
    """加载 Arc2Face pipeline，可选加载遗忘后的 UNet"""
    encoder = CLIPTextModelWrapper.from_pretrained(
        MODELS_DIR, subfolder="encoder", torch_dtype=torch.float16
    )
    if unet_path and os.path.exists(unet_path):
        print(f"Loading unlearned UNet from {unet_path}")
        unet = UNet2DConditionModel.from_pretrained(
            unet_path, torch_dtype=torch.float16
        )
    else:
        print("Loading original UNet")
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
    pipeline = pipeline.to(DEVICE)
    return pipeline


def load_face_analyzer():
    app = FaceAnalysis(
        name="antelopev2", root="./",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def extract_embedding(app, image):
    """从 PIL Image 或文件路径提取归一化 ArcFace embedding"""
    if isinstance(image, str):
        image = Image.open(image)
    img = np.array(image)[:, :, ::-1]  # RGB -> BGR
    faces = app.get(img)
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]))
    emb = torch.tensor(faces[-1]["embedding"], dtype=torch.float16)
    return emb / torch.norm(emb, dim=0, keepdim=True)


def get_reference_embedding(app, identity_id, id_map):
    """计算某个身份的参考 embedding（取前 10 张图片的平均）"""
    img_files = id_map.get(identity_id, [])[:10]
    embs = []
    for f in img_files:
        img_path = os.path.join(CELEBA_IMG_DIR, f)
        if os.path.exists(img_path):
            emb = extract_embedding(app, img_path)
            if emb is not None:
                embs.append(emb)
    if not embs:
        return None
    avg = torch.stack(embs).mean(dim=0)
    return avg / torch.norm(avg)


def generate_images(pipeline, face_emb, num_images=NUM_GEN_IMAGES, seed_base=42):
    """用 Arc2Face 生成一批图像"""
    # 准备 prompt embeds（每个图片用不同 seed）
    prompt_emb = project_face_embs(pipeline, face_emb.unsqueeze(0).to(DEVICE))

    all_images = []
    for i in range(num_images):
        generator = torch.Generator(device=DEVICE).manual_seed(seed_base + i)
        images = pipeline(
            prompt_embeds=prompt_emb,
            num_inference_steps=GEN_STEPS,
            guidance_scale=GEN_GUIDANCE,
            num_images_per_prompt=1,
            generator=generator,
        ).images
        all_images.append(images[0])
    return all_images


def verify_identity(app, image, ref_emb, device="cuda"):
    """
    验证生成的图像是否匹配参考身份
    返回 cosine similarity
    """
    gen_emb = extract_embedding(app, image)
    if gen_emb is None:
        return 0.0
    gen_emb = gen_emb.to(device)
    ref = ref_emb.to(device)
    cos_sim = F.cosine_similarity(gen_emb.unsqueeze(0), ref.unsqueeze(0)).item()
    return max(0.0, cos_sim)


def evaluate_model(pipeline, app, set_name, ids, id_map, unlearned=True):
    """
    评估模型：
    - 用每个身份的 embedding 生成图像
    - 计算生成图像与各个参考 embedding 的匹配率

    返回：{"forget_id": ..., "FA": ..., "EA": ..., "RA": ..., "ERB": ..., "details": {...}}
    """
    forget_id = ids["forget"]
    retain_ids = ids["retain"]
    all_ids = [forget_id] + retain_ids

    # 1. 准备参考 embeddings
    print(f"\n  Preparing reference embeddings for {set_name}...")
    ref_embs = {}
    for iid in all_ids:
        emb = get_reference_embedding(app, iid, id_map)
        if emb is not None:
            ref_embs[iid] = emb
            print(f"    ID {iid}: reference embedding ready")

    if forget_id not in ref_embs:
        print(f"  ERROR: Cannot extract reference embedding for forget ID {forget_id}")
        return None

    # 2. 为每个身份生成图像
    print(f"  Generating images ({NUM_GEN_IMAGES} per identity)...")
    gen_images = {}  # {identity_id: list of PIL Images}
    for iid in all_ids:
        if iid not in ref_embs:
            gen_images[iid] = []
            continue
        emb = ref_embs[iid].to(DEVICE)
        images = generate_images(pipeline, emb)
        gen_images[iid] = images

    # 保存生成图像
    save_dir = os.path.join(OUTPUT_DIR, set_name, "unlearned" if unlearned else "original")
    os.makedirs(save_dir, exist_ok=True)
    for iid, images in gen_images.items():
        id_dir = os.path.join(save_dir, f"ID_{iid}")
        os.makedirs(id_dir, exist_ok=True)
        for j, img in enumerate(images):
            img.save(os.path.join(id_dir, f"gen_{j+1}.png"))

    # 3. 计算 FA (Forget Accuracy)
    # FA = 条件为 forget 身份时，生成图像仍被验证为 forget 身份的比例
    print(f"\n  Computing FA (Forget Accuracy)...")
    forget_ref = ref_embs[forget_id]
    fa_matches = 0
    fa_total = 0
    fa_sims = []

    for img in gen_images.get(forget_id, []):
        cos_sim = verify_identity(app, img, forget_ref, DEVICE)
        fa_sims.append(cos_sim)
        fa_total += 1
        if cos_sim >= COS_SIM_THRESHOLD:
            fa_matches += 1

    FA = (fa_matches / max(fa_total, 1)) * 100
    EA = 100 - FA
    print(f"    FA = {FA:.2f}% (matches: {fa_matches}/{fa_total})")
    print(f"    EA = {EA:.2f}%")
    print(f"    Avg cosine sim to forget ref: {np.mean(fa_sims):.4f}")

    # 4. 计算 RA (Retain Accuracy)
    # RA = 条件为 retain 身份时，生成图像被正确验证为对应 retain 身份的比例
    print(f"\n  Computing RA (Retain Accuracy)...")
    ra_correct = 0
    ra_total = 0
    per_retain_ra = {}

    for rid in retain_ids:
        if rid not in ref_embs or rid not in gen_images:
            continue
        ref = ref_embs[rid]
        rid_correct = 0
        rid_total = 0
        for img in gen_images[rid]:
            cos_sim = verify_identity(app, img, ref, DEVICE)
            rid_total += 1
            ra_total += 1
            if cos_sim >= COS_SIM_THRESHOLD:
                rid_correct += 1
                ra_correct += 1
        per_retain_ra[rid] = (rid_correct / max(rid_total, 1)) * 100
        print(f"    ID {rid}: RA = {per_retain_ra[rid]:.2f}% ({rid_correct}/{rid_total})")

    RA = (ra_correct / max(ra_total, 1)) * 100
    print(f"    Overall RA = {RA:.2f}%")

    # 5. 计算 ERB
    if EA + RA > 0:
        ERB = 2 * EA * RA / (EA + RA)
    else:
        ERB = 0.0
    print(f"\n  {'='*40}")
    print(f"  ERB = {ERB:.2f}")
    print(f"  {'='*40}")

    return {
        "set_name": set_name,
        "forget_id": forget_id,
        "retain_ids": retain_ids,
        "FA": FA,
        "EA": EA,
        "RA": RA,
        "ERB": ERB,
        "fa_sims": fa_sims,
        "per_retain_ra": per_retain_ra,
    }


def main():
    id_map = get_identity_images()
    app = load_face_analyzer()

    # 先评估原始模型
    print("=" * 60)
    print("EVALUATING ORIGINAL MODEL")
    print("=" * 60)
    pipeline_orig = load_pipeline(unet_path=None)
    orig_results = {}
    for set_name, ids in VALIDATION_SETS.items():
        result = evaluate_model(pipeline_orig, app, set_name, ids, id_map, unlearned=False)
        if result:
            orig_results[set_name] = result

    # 再评估遗忘后的模型
    print("\n" + "=" * 60)
    print("EVALUATING UNLEARNED MODELS")
    print("=" * 60)
    unlearn_results = {}
    for set_name in VALIDATION_SETS:
        unet_path = f"./outputs/unlearning/{set_name}/final"
        pipeline_unlearn = load_pipeline(unet_path=unet_path)

        result = evaluate_model(pipeline_unlearn, app, set_name,
                                VALIDATION_SETS[set_name], id_map, unlearned=True)
        if result:
            unlearn_results[set_name] = result

    # 汇总对比
    print("\n" + "=" * 60)
    print("SUMMARY: Original vs Unlearned")
    print("=" * 60)
    for set_name in VALIDATION_SETS:
        if set_name in orig_results and set_name in unlearn_results:
            o = orig_results[set_name]
            u = unlearn_results[set_name]
            print(f"\n{set_name}:")
            print(f"  Original:  FA={o['FA']:.1f}%  EA={o['EA']:.1f}%  RA={o['RA']:.1f}%  ERB={o['ERB']:.1f}")
            print(f"  Unlearned: FA={u['FA']:.1f}%  EA={u['EA']:.1f}%  RA={u['RA']:.1f}%  ERB={u['ERB']:.1f}")
            print(f"  Δ:         FA{u['FA']-o['FA']:+.1f}%  EA{u['EA']-o['EA']:+.1f}%  "
                  f"RA{u['RA']-o['RA']:+.1f}%  ERB{u['ERB']-o['ERB']:+.1f}")


if __name__ == "__main__":
    main()
