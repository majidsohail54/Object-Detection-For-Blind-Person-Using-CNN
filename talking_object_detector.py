import cv2
import time
import threading
import queue
import numpy as np

from ultralytics import YOLO
import pyttsx3

# ---- Optional online TTS + translation (disabled by default) ----
# ---- Online TTS + translation (enabled) ----
# ---- Online TTS + translation (enabled) ----
USE_ONLINE_TTS = True
try:
    from gtts import gTTS
    from googletrans import Translator  # single instance below
    import pygame  # playback backend
    pygame.mixer.init()  # init once
except Exception as e:
    print("[ERROR] gTTS/googletrans/pygame import failed:", e)
    USE_ONLINE_TTS = False

# one global translator (faster than recreating every time)
GLOBAL_TRANSLATOR = Translator() if USE_ONLINE_TTS else None




# -----------------------------
# Config
# -----------------------------
MODEL_WEIGHTS = "yolov8n.pt"      # fast & small model; auto-download first run
CAM_INDEX = 0                     # default webcam
CONF_THRESH = 0.40                # starting confidence
IOU_THRESH = 0.45                 # NMS threshold
ANNOUNCE_COOLDOWN = 2.0           # seconds between same-label announcements
ANNOUNCE_TOPK = 3                 # speak top-K objects per cycle
BOX_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)

# Languages: (ISO code, name)
LANGS = [
    ("en", "English"),
    ("hi", "Hindi"),
    ("te", "Telugu"),
    ("ta", "Tamil"),
    ("ur", "Urdu"),


    # Add more like ("te","Telugu"), ("ta","Tamil"), ("ur","Urdu")
]
lang_index = 0

# Basic label translations for Hindi (extend as needed)
# Basic label translations for multiple languages (extend as needed)
LOCAL_MAP = {
    "hi": {  # Hindi
        "person": "व्यक्ति",
        "bicycle": "साइकिल",
        "car": "कार",
        "motorcycle": "मोटरसाइकिल",
        "bus": "बस",
        "train": "ट्रेन",
        "truck": "ट्रक",
        "traffic light": "ट्रैफिक लाइट",
        "stop sign": "स्टॉप साइन",
        "bench": "बेंच",
        "cat": "बिल्ली",
        "dog": "कुत्ता",
        "chair": "कुर्सी",
        "bottle": "बोतल",
        "cup": "कप",
        "cell phone": "मोबाइल फोन",
        "laptop": "लैपटॉप",
        "book": "किताब",
        "pen": "पेन",
        "watch": "घड़ी",
        "mouse": "माउस",
        "keyboard": "कीबोर्ड",
        "ball": "गेंद",
        "apple": "सेब",
        "banana": "केला",
    },
    "te": {  # Telugu
        "person": "వ్యక్తి",
        "bicycle": "సైకిల్",
        "car": "కారు",
        "motorcycle": "మోటార్‌సైకిల్",
        "bus": "బస్సు",
        "train": "రైలు",
        "truck": "ట్రక్",
        "traffic light": "ట్రాఫిక్ లైట్",
        "stop sign": "స్టాప్ సైన్",
        "bench": "బెంచ్",
        "cat": "పిల్లి",
        "dog": "కుక్క",
        "chair": "కుర్చీ",
        "bottle": "సీసా",
        "cup": "కప్పు",
        "cell phone": "మొబైల్ ఫోన్",
        "laptop": "ల్యాప్‌టాప్",
        "book": "పుస్తకం",
        "pen": "కలం",
        "watch": "గడియారం",
        "mouse": "మౌస్",
        "keyboard": "కీబోర్డ్",
        "ball": "బంతి",
        "apple": "ఆపిల్",
        "banana": "అరటి పండు",
    },
    "ta": {  # Tamil
        "person": "நபர்",
        "bicycle": "மிதிவண்டி",
        "car": "கார்",
        "motorcycle": "மோட்டார் சைக்கிள்",
        "bus": "பஸ்",
        "train": "ரயில்",
        "truck": "லாரி",
        "traffic light": "போக்குவரத்து விளக்கு",
        "stop sign": "நிறுத்தல் குறி",
        "bench": "நாற்காலி",
        "cat": "பூனை",
        "dog": "நாய்",
        "chair": "நாற்காலி",
        "bottle": "பாட்டில்",
        "cup": "கோப்பை",
        "cell phone": "கைபேசி",
        "laptop": "மடிக்கணினி",
        "book": "புத்தகம்",
        "pen": "பேனா",
        "watch": "கடிகாரம்",
        "mouse": "சுட்டி",
        "keyboard": "விசைப்பலகை",
        "ball": "பந்து",
        "apple": "ஆப்பிள்",
        "banana": "வாழைப்பழம்",
    },
    "ur": {  # Urdu
        "person": "آدمی",
        "bicycle": "سائیکل",
        "car": "کار",
        "motorcycle": "موٹر سائیکل",
        "bus": "بس",
        "train": "ٹرین",
        "truck": "ٹرک",
        "traffic light": "ٹریفک لائٹ",
        "stop sign": "رکنے کا نشان",
        "bench": "بنچ",
        "cat": "بلی",
        "dog": "کتا",
        "chair": "کرسی",
        "bottle": "بوتل",
        "cup": "پیالی",
        "cell phone": "موبائل فون",
        "laptop": "لیپ ٹاپ",
        "book": "کتاب",
        "pen": "قلم",
        "watch": "گھڑی",
        "mouse": "ماؤس",
        "keyboard": "کی بورڈ",
        "ball": "گیند",
        "apple": "سیب",
        "banana": "کیلا",
    }
}


# Spoken sentence patterns
PHRASES = {
    "en": lambda name, count: f"{count} {name}{'' if count==1 else 's'} ahead",
    "hi": lambda name, count: f"आगे {count} {name}",
}

def localize_label(label: str, lang: str) -> str:
    if lang == "en":
        return label
    if lang in LOCAL_MAP and label in LOCAL_MAP[lang]:
        return LOCAL_MAP[lang][label]
    if USE_ONLINE_TTS:
        try:
            translator = Translator()
            return translator.translate(label, dest=lang).text
        except Exception:
            pass
    return label

def phrase_for(label_localized: str, count: int, lang: str) -> str:
    fn = PHRASES.get(lang, PHRASES["en"])
    return fn(label_localized, count)

# -----------------------------
# Text To Speech worker thread
# -----------------------------
import os
import tempfile
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "talking_detector_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class Speaker:
    """
    Queue-based speaker with audio caching and fast playback.
    - Online: gTTS -> cache mp3 per (text, lang). Playback via pygame (blocking).
    - Offline: pyttsx3 (unchanged, uses installed OS voices).
    """
    def __init__(self, use_online=False):
        self.use_online = use_online
        self.q = queue.Queue()
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)

        if not use_online:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 180)
            self.engine.setProperty('volume', 1.0)
        else:
            self.engine = None

        self.thread.start()

    def say(self, text: str, lang_code="en"):
        self.q.put((text, lang_code))

    def _cached_audio_path(self, text: str, lang_code: str) -> str:
        safe = (text[:80].replace("/", "_").replace("\\", "_").replace(" ", "_"))
        name = f"{lang_code}__{safe}.mp3"
        return str(CACHE_DIR / name)

    def _play_mp3_blocking(self, path: str):
        # Use pygame mixer for reliable, ordered playback
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.02)
        except Exception as e:
            print("[TTS:pygame] Error playing:", e)

    def _run(self):
        while not self.stop_flag:
            try:
                text, lang_code = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            if self.use_online:
                try:
                    target_path = self._cached_audio_path(text, lang_code)
                    if not os.path.exists(target_path):
                        # generate once and save to cache
                        gTTS(text=text, lang=lang_code).save(target_path)

                    # Blocking playback keeps utterances ordered
                    self._play_mp3_blocking(target_path)

                except Exception as e:
                    print("[TTS:gTTS/pygame] Error:", e)

            else:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print("[TTS:pyttsx3] Error:", e)

    def stop(self):
        self.stop_flag = True
        try:
            if self.engine:
                self.engine.stop()
            # stop pygame mixer cleanly
            if USE_ONLINE_TTS:
                pygame.mixer.music.stop()
                # pygame.mixer.quit()  # optional
        except Exception:
            pass



# -----------------------------
# Main App
# -----------------------------
def print_controls():
    print("""
Controls:
  L : Switch language
  A : Toggle announcements
  + / - : Increase / Decrease confidence threshold
  Q : Quit
""")

def main():
    global lang_index, CONF_THRESH

    # Load model (downloads first time)
    model = YOLO(MODEL_WEIGHTS)

    # Init camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera. Try CAM_INDEX=1 or check permissions.")
        return

    # Optional: reduce resolution if slow
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    speaker = Speaker(use_online=USE_ONLINE_TTS)

    last_announce_time = 0
    announce_enabled = True
    last_said = {}  # label -> timestamp

    print_controls()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: no frame from camera.")
                break

            # Run YOLO inference
            results = model.predict(
                source=frame,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                verbose=False
            )

            det_counts = {}
            annotated = frame.copy()

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    label = model.names[cls_id]

                    det_counts[label] = det_counts.get(label, 0) + 1

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(annotated, text, (x1, max(20, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2, cv2.LINE_AA)

            # Speaking logic with cooldowns
            now = time.time()
            if announce_enabled and det_counts:
                top = sorted(det_counts.items(), key=lambda x: x[1], reverse=True)[:ANNOUNCE_TOPK]
                for label, count in top:
                    last = last_said.get(label, 0)
                    if now - last >= ANNOUNCE_COOLDOWN and now - last_announce_time >= 0.5:
                        lang_code, _ = LANGS[lang_index]
                        loc = localize_label(label, lang_code)
                        sentence = phrase_for(loc, count, lang_code)
                        speaker.say(sentence, lang_code=lang_code)
                        last_said[label] = now
                        last_announce_time = now

            # HUD overlay
            h, w = annotated.shape[:2]
            lang_code, lang_name = LANGS[lang_index]
            status = f"Lang: {lang_name}  Conf: {CONF_THRESH:.2f}  Announce: {'ON' if announce_enabled else 'OFF'}  [L, A, +/-, Q]"
            cv2.rectangle(annotated, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(annotated, status, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Talking Object Detector", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('l'), ord('L')):
                lang_index = (lang_index + 1) % len(LANGS)
                code, name = LANGS[lang_index]
                # audible cue
                speaker.say(f"Language {name}", lang_code=code if not USE_ONLINE_TTS else "en")
                print(f"[INFO]Switched to {name}{code})")
            elif key in (ord('a'), ord('A')):
                announce_enabled = not announce_enabled
                speaker.say("Announcements on" if announce_enabled else "Announcements off")
            elif key in (ord('+'), ord('=')):
                CONF_THRESH = min(0.95, CONF_THRESH + 0.05)
                speaker.say(f"Confidence {int(CONF_THRESH*100)} percent")
            elif key in (ord('-'), ord('_')):
                CONF_THRESH = max(0.05, CONF_THRESH - 0.05)
                speaker.say(f"Confidence {int(CONF_THRESH*100)} percent")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speaker.stop()

if __name__ == "__main__":
    main()
