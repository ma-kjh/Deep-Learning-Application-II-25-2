import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip
from prompt import Prompt_classes
from utils.loader import test_loader_list_MOS
from utils.metrics import get_measures

# --- [설정] ---
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Evaluates CLIP/SigLIP OOD Detection")

# [핵심] Model Options
parser.add_argument("--model_type", type=str, default="clip", 
                    choices=["clip", "siglip1", "siglip2"], 
                    help="clip: DataComp-XL (Best CLIP), siglip1: WebLI (Original), siglip2: WebLI-256 (Improved)")
# [핵심] Standard Args
parser.add_argument("--ood-dataset", type=str, default="iNaturalist")
parser.add_argument("--benchmark", type=str, default="imagenet")
parser.add_argument("--dir", type=str, default="/data")
parser.add_argument("--bs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)

# [복구] utils/loader.py 호환성을 위한 레거시 인자들 (삭제하면 에러남)
parser.add_argument("--prompt-name", type=str, default="The nice") # 에러 원인 해결
parser.add_argument("--models", type=str, default="ViT-B/16")
# parser.add_argument("--models", type=str, default="siglip2")
parser.add_argument("--clip", type=str, default="openai")
parser.add_argument("--methods", type=str, default="flyp") # loader가 확인할 수 있음
parser.add_argument("--sim", type=float, default=1.0)
parser.add_argument("--is-train", action="store_true", default=False)

args = parser.parse_args()

# --- [Seed 고정] ---
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- [모델 로드 로직: OpenCLIP 통합] ---
print(f"\n=== Loading Model Configuration: {args.model_type.upper()} ===")

if args.model_type == "clip":
    # 비교를 위해 OpenAI(WIT-400M) 대신 DataComp-XL(12B) 사용 -> SigLIP WebLI와 체급 맞춤
    model_name = "ViT-B-16"
    pretrained_tag = "laion400m_e31" 
    print(f"Model: {model_name} | Pretrained: {pretrained_tag}")
    print("NOTE: Using DataComp-XL instead of OpenAI for fair comparison with WebLI.")

elif args.model_type == "siglip1":
    # SigLIP 1 (Original, 224px)
    model_name = "ViT-B-16-SigLIP"
    pretrained_tag = "webli"
    print(f"SigLIP 1 (Original, 224px)")
    print(f"Model: {model_name} | Pretrained: {pretrained_tag}")

elif args.model_type == "siglip2":
    # SigLIP 2 (Improved, 256px) - 보통 SigLIP 2는 해상도를 키우고 Recipe를 개선함
    model_name = "ViT-B-16-SigLIP2"
    pretrained_tag = "webli"
    print(f"SigLIP 2 (Improved, 256px)")
    print(f"Model: {model_name} | Pretrained: {pretrained_tag}")
    print("NOTE: This represents the improved SigLIP (higher res/better recipe).")

# OpenCLIP으로 모델 로드
try:
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag)
    tokenizer = open_clip.get_tokenizer(model_name)
except Exception as e:
    print(f"Error loading model {model_name}: {e}")
    print("Try: pip install open_clip_torch")
    exit()

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)
model.eval()
print("Model load finished!")


# --- [데이터 로더] ---
# loader.py가 내부적으로 args.prompt_name 등을 참조하더라도 이제 에러가 안 남
print(f"Loading Datasets (Batch Size: {args.bs})...")
# 중요: loader가 반환하는 texts_in은 무시합니다 (_,). 
# 왜냐하면 loader는 SigLIP 토크나이저를 모르고 OpenAI 토크나이저를 쓸 것이기 때문입니다.
in_dataloader, out_dataloader, _ = test_loader_list_MOS(args, preprocess, device)


# --- [프롬프트 설정] ---
# 여기서 우리가 직접 SigLIP 호환 토크나이저로 다시 만듭니다.
imagenet_classes, _ = Prompt_classes("imagenet")
prompt_template = "A photo of a {}" 
print(f"Using Prompt: '{prompt_template}' (Re-tokenizing for {args.model_type})")

texts = [prompt_template.format(c) for c in imagenet_classes]
texts_in = tokenizer(texts).to(device)


# --- [유틸리티 함수] ---
def get_model_attributes(model):
    """모델에서 encode 함수와 logit scale/bias를 안전하게 추출"""
    m = model.module if hasattr(model, 'module') else model
    
    scale = m.logit_scale.exp().item() if hasattr(m, 'logit_scale') else 1.0
    bias = m.logit_bias.item() if hasattr(m, 'logit_bias') and m.logit_bias is not None else 0.0
    
    return m.encode_image, m.encode_text, scale, bias

encode_image_fn, encode_text_fn, logit_scale, logit_bias = get_model_attributes(model)
print(f"Logit Scale: {logit_scale:.4f}, Logit Bias: {logit_bias:.4f}")


def compute_image_features(dataloader, encode_fn):
    encoded_images = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Feature Extraction"):
            images = images.to(device)
            # float32로 변환하여 CPU로 이동 (정밀도 이슈 해결)
            embeddings = encode_fn(images).float().cpu()
            encoded_images.append(embeddings)
    features = torch.cat(encoded_images)
    return features / features.norm(dim=-1, keepdim=True)


def compute_logits(image_features, text_features, scale, bias):
    return (image_features @ text_features.T) * 1 + bias


def concatenate_and_scale(logits, neg_logits):
    return torch.cat([logits, neg_logits], dim=1)


# --- [메인 로직] ---

# 1. ImageNet 이미지 Feature 추출
imagenet_images_norm = compute_image_features(in_dataloader, encode_image_fn)

# 2. 텍스트(ID) Feature 추출
with torch.no_grad():
    imagenet_texts = encode_text_fn(texts_in)
    imagenet_texts_cpu = imagenet_texts.float().cpu()
    
    # Negative Labels
    print("Loading Negative Labels...")
    try:
        n = np.load('./Neglabel/neg_label_10000.npy')
        neg_tokens = tokenizer([str(i) for i in n]).to(device)
        neg_text = encode_text_fn(neg_tokens).float().cpu()
    except FileNotFoundError:
        print("Error: ./Neglabel/neg_label_10000.npy not found.")
        exit()

imagenet_texts_cpu_norm = imagenet_texts_cpu / imagenet_texts_cpu.norm(dim=-1, keepdim=True)
neg_text_norm = neg_text / neg_text.norm(dim=-1, keepdim=True)

# 3. Logits 계산
imagenet_logits = compute_logits(imagenet_images_norm, imagenet_texts_cpu_norm, logit_scale, logit_bias)
imagenet_neg_logits = compute_logits(imagenet_images_norm, neg_text_norm, logit_scale, logit_bias)

# 4. Accuracy 확인
try:
    labels = torch.load("./CLIP_im1k_features/val/valtarget.pt")
    acc = (imagenet_logits.argmax(dim=1).numpy() == labels.cpu().numpy()).sum()
    print(f"ACC : {acc / 50000:.4f} !")
except Exception:
    print("Warning: valtarget.pt not found or mismatch. Skipping accuracy check.")

# 5. OOD 데이터셋 처리
ood_names = ["iNaturalist", "SUN", "Places", "Textures"]
ood_logits = {}
ood_neg_logits = {}

for name, loader in zip(ood_names, out_dataloader):
    print(f"Processing OOD: {name}")
    features = compute_image_features(loader, encode_image_fn)
    ood_logits[name] = compute_logits(features, imagenet_texts_cpu_norm, logit_scale, logit_bias)
    ood_neg_logits[name] = compute_logits(features, neg_text_norm, logit_scale, logit_bias)


# --- [Evaluation] ---
to_np = lambda x : x.numpy()

# NegLabel
imagenet_score = to_np(F.softmax(concatenate_and_scale(imagenet_logits, imagenet_neg_logits), dim=1)[:, :1000].sum(dim=1, keepdim=True))

print("\n=== NegLabel Results ===")
for name in ood_names:
    ood_score = to_np(F.softmax(concatenate_and_scale(ood_logits[name], ood_neg_logits[name]), dim=1)[:, :1000].sum(dim=1, keepdim=True))
    print(name, get_measures(imagenet_score, ood_score))

# MCM
imagenet_mcm = to_np(F.softmax(imagenet_logits, dim=1).max(dim=1, keepdim=True)[0])

print("\n=== MCM Results ===")
for name in ood_names:
    ood_mcm = to_np(F.softmax(ood_logits[name], dim=1).max(dim=1, keepdim=True)[0])
    print(name, get_measures(imagenet_mcm, ood_mcm))