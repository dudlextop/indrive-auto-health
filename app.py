import os, time, json
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# ---------- Paths (Drive-friendly) ----------
DRIVE_ROOT = "/content/drive/MyDrive/indrive_hack"
DRIVE_WEI  = f"{DRIVE_ROOT}/weights"
DRIVE_REP  = f"{DRIVE_ROOT}/reports"
os.makedirs(DRIVE_REP, exist_ok=True)

CSS = os.path.join(os.path.dirname(__file__), "theme.css") if os.path.exists("theme.css") else None

# ---------- Save report to Drive ----------
def save_json_file(report):
    name = f"report_{int(time.time())}.json"
    path = os.path.join(DRIVE_REP, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path

# ---------- Helpers ----------
def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# ---------- Models ----------
# 1) YOLO for car crop (COCO)
crop_yolo = None
try:
    from ultralytics import YOLO
    crop_yolo = YOLO("yolov8n.pt")
except Exception:
    crop_yolo = None

# 2) YOLO for damages — two models
damage_yolo1 = None     # damage.pt / damage_rust.pt
damage_yolo2 = None     # damage_v2.pt
damage_names1 = None

p1 = _first_existing(
    "/content/weights/damage.pt",
    "/content/weights/damage_rust.pt",
    f"{DRIVE_WEI}/damage.pt",
    f"{DRIVE_WEI}/damage_rust.pt",
)
p2 = _first_existing(
    "/content/weights/damage_v2.pt",
    f"{DRIVE_WEI}/damage_v2.pt",
)

try:
    from ultralytics import YOLO
    if p1:
        damage_yolo1 = YOLO(p1)
        try:
            damage_names1 = damage_yolo1.model.names if hasattr(damage_yolo1, "model") else damage_yolo1.names
        except Exception:
            damage_names1 = None
    if p2:
        damage_yolo2 = YOLO(p2)
except Exception:
    damage_yolo1 = damage_yolo2 = None
    damage_names1 = None

# ---------- Labels & CLIP ----------
RU_LABEL = {"scratch":"царапина","dent":"вмятина","rust":"ржавчина","car":"авто"}
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
cleanliness_texts = ["a clean car photo","a dirty, muddy car photo"]

def normalize_label(name: str) -> str:
    n = str(name).lower() if name is not None else "damage"
    if n in ("dunt","dent "): return "dent"
    if n in ("scracth","scr","scrach"): return "scratch"
    return n

def crop_car(img: Image.Image) -> Image.Image:
    if crop_yolo is None:
        return img
    res = crop_yolo.predict(source=img, imgsz=640, conf=0.25, verbose=False)
    if not res: return img
    best_area, best_box = 0, None
    for r in res:
        if r.boxes is None: continue
        for b in r.boxes:
            try:
                if int(b.cls.item()) != 2:  # COCO 'car'
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                area = (x2-x1)*(y2-y1)
                if area > best_area:
                    best_area, best_box = area, (x1,y1,x2,y2)
            except:
                pass
    if best_box is None: return img
    x1,y1,x2,y2 = best_box
    x1=max(0,x1); y1=max(0,y1); x2=min(img.width,x2); y2=min(img.height,y2)
    return img.crop((x1,y1,x2,y2)) if x2>x1 and y2>y1 else img

@torch.no_grad()
def clip_cleanliness(img: Image.Image):
    inputs = clip_proc(text=cleanliness_texts, images=img, return_tensors="pt", padding=True).to(device)
    probs = clip_model(**inputs).logits_per_image[0].softmax(dim=0).detach().cpu().numpy()
    idx = int(np.argmax(probs))
    return ["clean","dirty"][idx], {t: float(p) for t,p in zip(cleanliness_texts, probs)}

def quality_flags(img: Image.Image):
    arr = np.array(img.convert("L"))
    bright = arr.mean(); contrast = arr.std()
    pad = np.pad(arr, 1, mode="edge")
    lap = (pad[:-2,1:-1] + pad[2:,1:-1] + pad[1:-1,:-2] + pad[1:-1,2:] - 4*arr)
    sharp = float(lap.var())

    flags = []
    if bright < 70:  flags.append("low_light")
    if bright > 200: flags.append("overexposed")
    if contrast < 20:flags.append("low_contrast")
    if sharp < 80:   flags.append("blurry")
    stats = dict(bright=round(float(bright),1), contrast=round(float(contrast),1), sharp=round(sharp,1))
    return flags, stats

def quality_tips(flags):
    tips = []
    if "low_light" in flags:   tips.append("Снимай при дневном свете или включи вспышку.")
    if "overexposed" in flags: tips.append("Смени угол, убери блики.")
    if "low_contrast" in flags:tips.append("Подойди ближе — заполни кадр машиной.")
    if "blurry" in flags:      tips.append("Дай автофокусу сработать, задержи телефон на 1 секунду.")
    return tips

COLOR_MAP = {"scratch":(255,0,0),"dent":(255,140,0),"rust":(255,215,0),"car":(0,255,0)}

def draw_label(draw: ImageDraw.ImageDraw, xy, text, color):
    x1,y1,x2,y2 = xy
    try:
        font = ImageFont.load_default()
        tw = int(draw.textlength(text, font=font)); th = 12
    except:
        font = None; tw, th = 60, 12
    pad = 2
    draw.rectangle([x1, y1, x1+tw+pad*2, y1+th+pad*2], fill=color)
    draw.text((x1+pad, y1+pad), text, fill=(0,0,0), font=font)

def _run(det, img: Image.Image, conf_thr=0.35, names=None):
    if det is None: return []
    res = det.predict(source=img, imgsz=640, conf=conf_thr, verbose=False)
    out = []
    for r in res:
        if r.boxes is None: continue
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            score = float(b.conf.item())
            cls_id = int(b.cls.item()) if hasattr(b,"cls") else -1
            raw = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else None
            out.append({"xyxy":[x1,y1,x2,y2], "score":score, "raw_name":raw})
    return out

def detect_damage(img: Image.Image, conf_thr=0.35):
    boxes = []
    boxes += _run(damage_yolo1, img, conf_thr, names=damage_names1)
    boxes += _run(damage_yolo2, img, conf_thr, names=None)

    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    out = []
    for b in boxes:
        x1,y1,x2,y2 = b["xyxy"]; score = b["score"]
        label = normalize_label(b.get("raw_name","damage"))
        ru = RU_LABEL.get(label, label)
        color = COLOR_MAP.get(label, (0,200,255))
        draw.rectangle([x1,y1,x2,y2], width=3, outline=color)
        draw_label(draw, (x1,y1,x2,y2), f"{ru} {score:.2f}", color)
        out.append({"xyxy":[x1,y1,x2,y2], "score":score, "label":label})
    return out, vis

def compute_counts(dmg_boxes):
    counts = {}
    for b in dmg_boxes:
        lbl = b["label"]
        if lbl == "car": continue
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts

def final_status(cleanliness, dmg_boxes, qflags):
    if any(b["label"] in ("scratch","dent","rust") for b in dmg_boxes):
        return "❌ повреждена — требуется осмотр"
    if "blurry" in qflags or "low_light" in qflags:
        return "⚠️ низкое качество — сделайте новое фото"
    if cleanliness == "dirty":
        return "⚠️ нужна мойка"
    return "✅ ок — готово"

def health_score(cleanliness, dmg_counts, qflags):
    score = 100
    score -= 12 * sum(dmg_counts.values())
    if cleanliness == "dirty": score -= 10
    score -= 5 * len(qflags)
    return int(max(0, min(100, score)))

def acceptable(status):
    return status.startswith("✅") or status.startswith("⚠️ нужна мойка")

def counts_ru_text(counts_ru: dict) -> str:
    if not counts_ru: return "Нет выявленных дефектов."
    return "\n".join([f"- {k}: {v}" for k,v in counts_ru.items()])

def predict(image: Image.Image, use_crop: bool, conf_thr: float):
    if image is None:
        return {"error":"Загрузите фото"}, None, None, None, None, None, None, None, None, None
    img = image.convert("RGB")
    cropped = crop_car(img) if use_crop else img

    qflags, qstats = quality_flags(cropped)
    tips = quality_tips(qflags)
    clean_label, _ = clip_cleanliness(cropped)
    dmg_boxes, vis = detect_damage(cropped, conf_thr=conf_thr)

    damage_label = "damaged" if any(b["label"] in ("scratch","dent","rust") for b in dmg_boxes) else "undamaged"
    counts_en = compute_counts(dmg_boxes)
    counts_ru = {RU_LABEL.get(k, k): v for k, v in counts_en.items()}
    counts_ru_md = counts_ru_text(counts_ru)

    score = health_score(clean_label, counts_en, qflags)
    status = final_status(clean_label, dmg_boxes, qflags)
    acceptable_flag = acceptable(status)

    report = {
        "status": status,
        "acceptable_for_listing": acceptable_flag,
        "health_score": score,
        "cleanliness": clean_label,
        "damage": damage_label,
        "damage_counts": counts_en,
        "damage_counts_ru": counts_ru,
        "damage_boxes": dmg_boxes,
        "quality_flags": qflags,
        "quality_stats": qstats,
        "quality_tips": tips
    }
    tips_text = ("• " + "\n• ".join(tips)) if tips else "—"
    clean_ru = "Чистая" if clean_label=="clean" else "Грязная"
    dmg_ru   = "Повреждена" if damage_label=="damaged" else "Без повреждений"

    return report, vis, clean_ru, dmg_ru, status, counts_ru, counts_ru_md, score, tips_text, acceptable_flag

# ---------- UI ----------
with gr.Blocks(css=open(CSS).read() if CSS else None, title="Оценка состояния автомобиля для inDrive") as demo:
    gr.Markdown("<h1 class='title'>🚗 Оценка состояния автомобиля для inDrive</h1>")
    with gr.Row():
        with gr.Column(scale=5):
            in_img   = gr.Image(type="pil", label="Загрузите фото автомобиля")
            usecrop  = gr.Checkbox(value=True, label="Кропать авто (YOLO)")
            conf     = gr.Slider(0.1,0.7,value=0.35,step=0.05,label="Порог уверенности повреждений")
            with gr.Row():
                gr.Button("Агрессивный (0.25)").click(lambda: 0.25, None, conf)
                gr.Button("Сбалансированный (0.35)").click(lambda: 0.35, None, conf)
                gr.Button("Консервативный (0.45)").click(lambda: 0.45, None, conf)
            btn      = gr.Button("Анализ", variant="primary")
        with gr.Column(scale=7):
            out_json = gr.JSON(label="Отчёт (JSON)")
            save_btn  = gr.Button("Скачать JSON")
            save_file = gr.File(label="Файл отчёта")
            out_img  = gr.Image(label="Кадр с боксами")
            clean_txt= gr.Textbox(label="Чистота", interactive=False)
            dmg_txt  = gr.Textbox(label="Повреждения", interactive=False)
            status   = gr.Textbox(label="Итоговый статус", interactive=False)
            counts_j = gr.JSON(label="Счётчик дефектов (РУС)")
            counts_md= gr.Markdown("—")
            score_ui = gr.Slider(0, 100, value=100, step=1, label="Индекс состояния (0–100)", interactive=False)
            tips_md  = gr.Markdown("Подсказки по качеству появятся здесь")
            ok_flag  = gr.Checkbox(value=False, label="Готово к размещению", interactive=False)

    btn.click(
        predict,
        inputs=[in_img, usecrop, conf],
        outputs=[out_json, out_img, clean_txt, dmg_txt, status, counts_j, counts_md, score_ui, tips_md, ok_flag]
    )
    save_btn.click(save_json_file, inputs=[out_json], outputs=[save_file])
    gr.Markdown("<footer class='small'>Сделано с любовью от команды <b>DOGS</b> для inDrive</footer>")

def _launch():
    # close any running Gradio (Colab sometimes keeps 7860 busy)
    try: gr.close_all()
    except Exception: pass
    # try preferred port first
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except OSError:
        demo.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    _launch()
