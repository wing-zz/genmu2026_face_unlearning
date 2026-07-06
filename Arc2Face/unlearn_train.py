"""
身份遗忘训练脚本
策略：仅微调 UNet cross-attention K,V 层，将 forget embedding 重定向到 anchor 身份
"""
import os, sys, copy
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from diffusers import (
    StableDiffusionPipeline, UNet2DConditionModel,
    DPMSolverMultistepScheduler, DDPMScheduler,
)
from arc2face import CLIPTextModelWrapper, project_face_embs
from insightface.app import FaceAnalysis

# === 配置 ===
BASE_MODEL = "/data/weight"
MODELS_DIR = "./models"
CELEBA_IMG_DIR = "../Data/CelebA/Img/img_align_celeba"
OUTPUT_DIR = "./outputs/unlearning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALIDATION_SETS = {
    "Face_Set_1": {"forget": 3422, "retain": [5230, 5239, 1539]},
    "Face_Set_2": {"forget": 3376, "retain": [3602, 608, 7405]},
}

TRAIN_CFG = dict(num_epochs=20, lr=1e-5, max_images=15, save_every=5)

# ==================== 工具函数 ====================

def load_pipeline():
    """加载 Arc2Face pipeline，全 FP32"""
    print("Loading Arc2Face pipeline (FP32)...")

    encoder = CLIPTextModelWrapper.from_pretrained(MODELS_DIR, subfolder="encoder")
    unet = UNet2DConditionModel.from_pretrained(MODELS_DIR, subfolder="arc2face")

    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, text_encoder=encoder, unet=unet, safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # 显式逐模块转 FP32 + CUDA
    for attr in ["text_encoder", "unet", "vae"]:
        m = getattr(pipeline, attr)
        setattr(pipeline, attr, m.to(device="cuda", dtype=torch.float32))

    assert next(pipeline.unet.parameters()).dtype == torch.float32, "UNet not FP32!"
    print("Pipeline loaded. All FP32 ✓")
    return pipeline


def load_face_analyzer():
    app = FaceAnalysis(name="antelopev2", root="./", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_id_map():
    id_map = defaultdict(list)
    with open("../Data/CelebA/Anno/identity_CelebA.txt") as f:
        for line in f:
            parts = line.strip().split()
            id_map[int(parts[1])].append(parts[0])
    return dict(id_map)


def get_avg_embedding(app, identity_id, id_map, device="cuda"):
    """提取某个身份的平均 ArcFace embedding（FP32）"""
    img_files = id_map.get(identity_id, [])[:10]
    embs = []
    for f in img_files:
        path = os.path.join(CELEBA_IMG_DIR, f)
        if not os.path.exists(path):
            continue
        img = np.array(Image.open(path))[:, :, ::-1]
        faces = app.get(img)
        if not faces:
            continue
        faces = sorted(faces, key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]))
        emb = torch.tensor(faces[-1]["embedding"], dtype=torch.float32)
        embs.append(emb / torch.norm(emb))
    if not embs:
        return None
    avg = torch.stack(embs).mean(dim=0)
    return (avg / torch.norm(avg)).to(device)


def encode_images_to_latents(vae, img_paths):
    """批量编码图片为 VAE latents (FP32)"""
    latents = []
    for p in img_paths:
        img = Image.open(p).convert("RGB").resize((512, 512))
        t = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t * 2.0 - 1.0  # [-1, 1]
        t = t.to(device="cuda", dtype=torch.float32)
        with torch.no_grad():
            l = vae.encode(t).latent_dist.sample()
            l = l * vae.config.scaling_factor
        latents.append(l.cpu())
    return torch.cat(latents, dim=0)


def freeze_unet_except_cross_attn(unet):
    """冻结 UNet，仅解冻 cross-attention K,V 投影"""
    for p in unet.parameters():
        p.requires_grad = False

    n_trainable = 0
    n_total = sum(p.numel() for p in unet.parameters())
    for name, p in unet.named_parameters():
        if "attn2.to_k" in name or "attn2.to_v" in name:
            p.requires_grad = True
            n_trainable += p.numel()

    print(f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")
    return unet


# ==================== 训练主函数 ====================

def train_unlearning(pipeline, app, id_map, set_name, forget_id, retain_ids, cfg=TRAIN_CFG):
    print(f"\n{'='*60}")
    print(f"Training: {set_name}  |  Forget: {forget_id}  |  Retain: {retain_ids}")
    print(f"{'='*60}")

    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder

    # 1. Freeze
    unet = freeze_unet_except_cross_attn(unet)

    # 2. 提取 embeddings
    forget_emb = get_avg_embedding(app, forget_id, id_map)
    if forget_emb is None:
        raise RuntimeError(f"Cannot extract embedding for ID {forget_id}")

    retain_embs = {}
    for rid in retain_ids:
        e = get_avg_embedding(app, rid, id_map)
        if e is not None:
            retain_embs[rid] = e

    # 3. 选 anchor（最远的 retain 身份 — 确保 forget 被重定向到差异性大的目标）
    best_rid, best_sim = None, 999
    for rid, e in retain_embs.items():
        s = F.cosine_similarity(forget_emb.unsqueeze(0), e.unsqueeze(0)).item()
        if s < best_sim:
            best_sim, best_rid = s, rid
    anchor_emb = retain_embs[best_rid]
    print(f"Anchor: ID {best_rid} (cosine_sim={best_sim:.4f}, farthest from forget)")

    # 4. 准备图片
    def get_paths(iid, n=cfg["max_images"]):
        return [os.path.join(CELEBA_IMG_DIR, f)
                for f in id_map.get(iid, [])[:n] if os.path.exists(os.path.join(CELEBA_IMG_DIR, f))]

    anchor_paths = get_paths(best_rid)
    print(f"Anchor images: {len(anchor_paths)}")

    retain_data = {}
    for rid in retain_ids:
        paths = get_paths(rid)
        if paths:
            retain_data[rid] = paths
            print(f"  Retain ID {rid}: {len(paths)} images")

    # 5. Encode latents
    anchor_latents = encode_images_to_latents(vae, anchor_paths)
    retain_latents = {rid: encode_images_to_latents(vae, paths) for rid, paths in retain_data.items()}

    # 6. Prompt embeddings (FP32)
    forget_prompt = project_face_embs(pipeline, forget_emb.unsqueeze(0))
    retain_prompts = {rid: project_face_embs(pipeline, retain_embs[rid].unsqueeze(0))
                      for rid in retain_data}

    # 7. Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    # 8. Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=cfg["lr"]
    )

    # 9. Train
    print(f"\nTraining {cfg['num_epochs']} epochs...")
    for epoch in range(cfg["num_epochs"]):
        unet.train()
        loss_f_sum, loss_r_sum, n_f, n_r = 0, 0, 0, 0

        # Forget: anchor 图片 + forget embedding → 预测噪声
        for i in range(len(anchor_latents)):
            x0 = anchor_latents[i:i+1].to("cuda")
            noise = torch.randn_like(x0)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda").long()
            xt = noise_scheduler.add_noise(x0, noise, t)

            pred = unet(xt, t, encoder_hidden_states=forget_prompt).sample
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            loss_f_sum += loss.item(); n_f += 1

        # Retain: retain 图片 + retain embedding → 预测噪声（保持）
        for rid, latents in retain_latents.items():
            prompt = retain_prompts[rid]
            for i in range(len(latents)):
                x0 = latents[i:i+1].to("cuda")
                noise = torch.randn_like(x0)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda").long()
                xt = noise_scheduler.add_noise(x0, noise, t)

                pred = unet(xt, t, encoder_hidden_states=prompt).sample
                loss = F.mse_loss(pred, noise)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                loss_r_sum += loss.item(); n_r += 1

        print(f"Epoch {epoch+1:3d}/{cfg['num_epochs']} | "
              f"Loss_f: {loss_f_sum/max(n_f,1):.6f} | Loss_r: {loss_r_sum/max(n_r,1):.6f}")

        if (epoch + 1) % cfg["save_every"] == 0:
            ckpt = os.path.join(OUTPUT_DIR, set_name, f"ckpt-{epoch+1}")
            os.makedirs(ckpt, exist_ok=True)
            unet.to(torch.float16).save_pretrained(ckpt)
            unet.to(torch.float32)  # 恢复继续训练

    # Final save (FP16)
    final = os.path.join(OUTPUT_DIR, set_name, "final")
    os.makedirs(final, exist_ok=True)
    unet.to(torch.float16).save_pretrained(final)
    unet.to(torch.float32)  # 恢复 FP32，供下一个 set 使用
    torch.save({"forget_id": forget_id, "retain_ids": retain_ids, "anchor_id": best_rid}, os.path.join(final, "meta.pt"))
    print(f"Saved: {final}")
    return final


# ==================== Main ====================

def main():
    pipeline = load_pipeline()
    app = load_face_analyzer()
    id_map = get_id_map()

    for set_name, ids in VALIDATION_SETS.items():
        train_unlearning(pipeline, app, id_map, set_name, ids["forget"], ids["retain"])

    print("\n✅ All training complete!")


if __name__ == "__main__":
    main()
