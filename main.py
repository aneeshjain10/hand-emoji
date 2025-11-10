# Emoji Game — Real-time Hand Gesture → Emoji (your build)
# Author: MAYANK_JAIN | License: MIT
...


import math, time
from collections import deque

import cv2, numpy as np, mediapipe as mp

# Pillow optional (colored emoji)
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

EMOJI_FONT_PATHS = [
    r"C:\Windows\Fonts\seguiemj.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/joypixels/JoyPixels.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
]

def lm_to_xy(lm, w, h): return int(lm.x*w), int(lm.y*h)
def dist(a,b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def fingers_up(hand_lms, handed_label, w, h):
    pts = {i: lm_to_xy(hand_lms[i], w, h) for i in range(21)}
    tips = {'index':8,'middle':12,'ring':16,'pinky':20}
    pips = {'index':6,'middle':10,'ring':14,'pinky':18}
    status = {}
    for name in tips:
        tip, pip = pts[tips[name]], pts[pips[name]]
        status[name] = tip[1] < pip[1]

    xs = [p[0] for p in pts.values()]; ys = [p[1] for p in pts.values()]
    box_w, box_h = max(xs)-min(xs), max(ys)-min(ys)
    diag = (box_w**2 + box_h**2) ** 0.5 or max(w,h)*0.2

    wrist = pts[0]; thumb_tip = pts[4]; thumb_ip = pts[3]
    palm_center = (int((pts[0][0]+pts[5][0]+pts[9][0]+pts[13][0]+pts[17][0])/5),
                   int((pts[0][1]+pts[5][1]+pts[9][1]+pts[13][1]+pts[17][1])/5))

    far_from_palm = dist(thumb_tip, palm_center) > 0.32*diag
    lateral_sep   = abs(thumb_tip[0]-thumb_ip[0]) > 0.12*diag
    vertical_sep  = (thumb_ip[1]-thumb_tip[1])   > 0.08*diag
    thumb = far_from_palm and (lateral_sep or vertical_sep)
    if dist(thumb_tip, wrist) < 0.04*max(w,h): thumb = False

    status['thumb'] = thumb
    return status, pts

def detect_gesture(hand_lms, handed_label, w, h):
    status, pts = fingers_up(hand_lms, handed_label, w, h)
    xs = [p[0] for p in pts.values()]; ys = [p[1] for p in pts.values()]
    box_w, box_h = max(xs)-min(xs), max(ys)-min(ys)
    diag = (box_w**2 + box_h**2) ** 0.5 or max(w,h)*0.2

    if not any([status['index'],status['middle'],status['ring'],status['pinky'],status['thumb']]):
        return '✊','Fist'
    if status['thumb'] and status['index'] and status['middle'] and status['ring'] and status['pinky']:
        return '🖐','Open Palm'
    if status['index'] and status['middle'] and not status['ring'] and not status['pinky']:
        return '✌️','Peace'

    wrist, ttip = pts[0], pts[4]
    if status['thumb'] and not status['index'] and not status['middle'] and not status['ring'] and not status['pinky']:
        center_x = (min(xs)+max(xs))/2
        if ttip[1] < wrist[1]-20 or abs(ttip[0]-center_x) > 0.3*box_w:
            return '👍','Thumbs Up'

    if dist(pts[8], pts[4]) < 0.25*diag and dist(pts[8], pts[4]) > 8:
        return '👌','OK'
    return None, 'Unknown'

def pick_emoji_font(px):
    if not PIL_AVAILABLE: return None
    for p in EMOJI_FONT_PATHS:
        try: return ImageFont.truetype(p, px)
        except Exception: pass
    try: return ImageFont.load_default()
    except Exception: return None

def draw_emoji(frame_bgr, emoji, center_xy, size=2.0):
    x,y = center_xy
    if PIL_AVAILABLE:
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = pick_emoji_font(int(64*size))
        try:
            bbox = draw.textbbox((0,0), emoji, font=font); tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            tw = th = int(64*size)
        pos = (int(x - tw/2), int(y - th/2))
        for dx,dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            draw.text((pos[0]+dx,pos[1]+dy), emoji, font=font, fill="black")
        draw.text(pos, emoji, font=font, fill="white")
        frame_bgr[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        radius = int(36*size)
        cv2.circle(frame_bgr, (x,y), radius, (40,40,40), -1)
        cv2.putText(frame_bgr, emoji, (x-radius+8, y+8), cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255), 2, cv2.LINE_AA)

def main():
    mp_hands = mp.solutions.hands; mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Webcam not found."); return

    prev = time.time(); last_emoji=None; last_seen=0.0
    window = deque(maxlen=6)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: print("Frame grab failed."); break
            frame = cv2.flip(frame, 1)
            h,w,_ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb.flags.writeable=False
            res = hands.process(rgb); rgb.flags.writeable=True

            overlay_xy = (int(w*0.5), int(h*0.2))
            label = "No hand"; current=None

            if res.multi_hand_landmarks and res.multi_handedness:
                hand = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label

                mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0,220,0), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0,80,255), thickness=2))

                emoji, name = detect_gesture(hand.landmark, handed, w, h)
                label = f"{handed} hand: {name}"

                xs = [int(lm.x * w) for lm in hand.landmark]
                ys = [int(lm.y * h) for lm in hand.landmark]
                cx = int((min(xs)+max(xs))/2); cy = int(min(ys)-24) if (min(ys)-24)>24 else int((min(ys)+max(ys))/2)
                overlay_xy = (cx, cy)

                if emoji:
                    window.append(emoji)
                    current = max(set(window), key=window.count)
                    last_emoji = current; last_seen = time.time()
                else:
                    if last_emoji and (time.time()-last_seen) < 0.35:
                        current = last_emoji

                cv2.rectangle(frame, (min(xs)-10, min(ys)-10), (max(xs)+10, max(ys)+10), (255,200,0), 2)
            else:
                if last_emoji and (time.time()-last_seen) > 0.6:
                    last_emoji=None; window.clear()

            if current: draw_emoji(frame, current, overlay_xy, size=2.2)

            now = time.time(); fps = 1.0 / (now - prev) if (now - prev) > 1e-6 else 0.0; prev = now
            cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, label, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)
            cv2.putText(frame, "Press 'q' to quit", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

            cv2.imshow("Emoji Game — Hand Gesture (Your Build)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release(); cv2.destroyAllWindows(); hands.close()

if __name__ == "__main__":
    main()
