import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp
import os
import time

# -----------------------
# ตั้งค่าหน้าเว็บ
# -----------------------
st.set_page_config(
    page_title="Ergonomic",
    page_icon="🪑",
    layout="centered"
)

st.title("Ergonomic")
st.caption("ใช้ YOLO + Pose ประเมินมุมคอ หลัง เข่า จากภาพด้านข้าง")

# โหลดโมเดล YOLO (cache)
@st.cache_resource
def load_yolo_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"ไม่พบไฟล์โมเดล {model_path} ในโฟลเดอร์ปัจจุบัน")
        st.stop()
    model = YOLO(model_path)
    return model

yolo_model = load_yolo_model()

#MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ฟังก์ชันช่วยด้านยศาสตร์
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cos_ang = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_ang))
    return angle

def flex_from_straight(angle):
    if angle is None:
        return None
    return abs(180.0 - angle)

def choose_side_landmarks(landmarks):
    lm = mp_pose.PoseLandmark

    def get_xyz(id_):
        p = landmarks[id_]
        return p.x, p.y, p.visibility

    left_ids = [lm.LEFT_EAR, lm.LEFT_SHOULDER, lm.LEFT_HIP, lm.LEFT_KNEE, lm.LEFT_ANKLE]
    right_ids = [lm.RIGHT_EAR, lm.RIGHT_SHOULDER, lm.RIGHT_HIP, lm.RIGHT_KNEE, lm.RIGHT_ANKLE]

    left_points = [get_xyz(int(i.value)) for i in left_ids]
    right_points = [get_xyz(int(i.value)) for i in right_ids]

    left_vis = np.mean([p[2] for p in left_points])
    right_vis = np.mean([p[2] for p in right_points])

    if left_vis >= right_vis:
        side = "left"
        ear = left_points[0][:2]
        shoulder = left_points[1][:2]
        hip = left_points[2][:2]
        knee = left_points[3][:2]
        ankle = left_points[4][:2]
    else:
        side = "right"
        ear = right_points[0][:2]
        shoulder = right_points[1][:2]
        hip = right_points[2][:2]
        knee = right_points[3][:2]
        ankle = right_points[4][:2]

    return side, {
        "ear": ear,
        "shoulder": shoulder,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
    }

def classify_ergonomic(neck_flex, trunk_flex, knee_angle):
    if neck_flex is None or trunk_flex is None or knee_angle is None:
        return "ไม่แน่ใจ", "unknown", ["ตรวจจับจุดสำคัญไม่ครบ (neck/trunk/knee เป็น None)"]

    reason = []

    # คอ
    if neck_flex <= 20:
        reason.append(f"คออยู่ในช่วงดี (เบี่ยงจากแนวตรง ~ {neck_flex:.1f}°)")
        neck_score = 2
    elif neck_flex <= 45:
        reason.append(f"คอเริ่มก้ม/เงยมาก (~ {neck_flex:.1f}°)")
        neck_score = 1
    else:
        reason.append(f"คอก้ม/เงยมากเกินไป (~ {neck_flex:.1f}°)")
        neck_score = 0

    # หลัง
    if trunk_flex <= 20:
        reason.append(f"หลังอยู่ในช่วงดี (เบี่ยงจากแนวตรง ~ {trunk_flex:.1f}°)")
        trunk_score = 2
    elif trunk_flex <= 45:
        reason.append(f"หลังเริ่มเอน/งอมาก (~ {trunk_flex:.1f}°)")
        trunk_score = 1
    else:
        reason.append(f"หลังงอมากเกินไป (~ {trunk_flex:.1f}°)")
        trunk_score = 0

    # เข่า
    if 80 <= knee_angle <= 120:
        reason.append(f"มุมเข่าอยู่ในช่วงเหมาะสม (~ {knee_angle:.1f}°)")
        knee_score = 2
    else:
        reason.append(f"มุมเข่าอาจไม่เหมาะสม (~ {knee_angle:.1f}°)")
        knee_score = 1

    total = neck_score + trunk_score + knee_score

    if total >= 5:
        status = "ท่านั่งดีตามหลักการยศาสตร์"
        level = "good"
    elif total >= 3:
        status = "ท่านั่งพอใช้ แต่ควรปรับบางจุด"
        level = "caution"
    else:
        status = "ท่านั่งเสี่ยงต่อการปวดเมื่อย/บาดเจ็บ"
        level = "poor"

    return status, level, reason

# วิเคราะห์ด้วย MediaPipe (fallback)
def analyze_posture_mediapipe_full(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out, None, None, None, None, "unknown", ["ไม่พบโครงร่างบุคคลในภาพ"]

    landmarks = results.pose_landmarks.landmark
    side, pts = choose_side_landmarks(landmarks)
    ear = pts["ear"]
    shoulder = pts["shoulder"]
    hip = pts["hip"]
    knee = pts["knee"]
    ankle = pts["ankle"]

    neck_angle = calculate_angle(ear, shoulder, hip)
    trunk_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    neck_flex = flex_from_straight(neck_angle)
    trunk_flex = flex_from_straight(trunk_angle)

    status, level, reason = classify_ergonomic(neck_flex, trunk_flex, knee_angle)

    annotated = img_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    )

    out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return out_rgb, side, neck_flex, trunk_flex, knee_angle, level, reason

# วิเคราะห์ด้วย YOLO (+ fallback)
def analyze_posture_yolo_ergonomic(img_bgr, yolo_conf=0.3):
    h, w, _ = img_bgr.shape
    results = yolo_model(img_bgr, conf=yolo_conf, verbose=False)

    # ถ้า YOLO ไม่เจอเลย → ใช้ MediaPipe ทั้งภาพ
    if len(results) == 0 or len(results[0].boxes) == 0:
        return analyze_posture_mediapipe_full(img_bgr)

    r = results[0]
    boxes = r.boxes

    # box area
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        areas.append((x2 - x1) * (y2 - y1))
    idx = int(np.argmax(areas))
    box = boxes[idx]

    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(h, int(x2))
    y2 = min(w, int(y2))

    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    cls_name = yolo_model.names.get(cls_id, str(cls_id))

    roi_bgr = img_bgr[y1:y2, x1:x2].copy()
    if roi_bgr.size == 0:
        return analyze_posture_mediapipe_full(img_bgr)

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(roi_rgb)

    if not pose_results.pose_landmarks:
        return analyze_posture_mediapipe_full(img_bgr)

    landmarks = pose_results.pose_landmarks.landmark
    side, pts = choose_side_landmarks(landmarks)
    ear = pts["ear"]
    shoulder = pts["shoulder"]
    hip = pts["hip"]
    knee = pts["knee"]
    ankle = pts["ankle"]

    neck_angle = calculate_angle(ear, shoulder, hip)
    trunk_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)

    neck_flex = flex_from_straight(neck_angle)
    trunk_flex = flex_from_straight(trunk_angle)

    status, level, reason = classify_ergonomic(neck_flex, trunk_flex, knee_angle)

    annotated_roi = roi_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated_roi,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    )

    annotated_full = img_bgr.copy()
    annotated_full[y1:y2, x1:x2] = annotated_roi

    color_box = (0, 255, 0) if level == "good" else ((0, 255, 255) if level == "caution" else (0, 0, 255))
    cv2.rectangle(annotated_full, (x1, y1), (x2, y2), color_box, 2)
    cv2.putText(
        annotated_full,
        f"{cls_name} {conf:.2f}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_box,
        2
    )

    out_rgb = cv2.cvtColor(annotated_full, cv2.COLOR_BGR2RGB)
    return out_rgb, side, neck_flex, trunk_flex, knee_angle, level, reason


# UI หลัก
st.divider()
mode = st.radio(
    "เลือกแหล่งภาพ",
    ["อัปโหลดรูป", "ถ่ายจากกล้อง", "Livecam"],
    horizontal=True
)

yolo_conf = st.slider(
    "confidence",
    0.1, 0.9, 0.3, 0.05,
    help="ใช้กำหนดความมั่นใจขั้นต่ำของ YOLO ก่อนนับว่าเจอคน"
)

# โหมด 1–2: ภาพนิ่ง (upload/snapshot)
if mode in ["อัปโหลดรูป", "ถ่ายจากกล้อง"]:
    img_bgr = None

    if mode == "อัปโหลดรูป":
        file = st.file_uploader("อัปโหลดรูปท่านั่ง (ด้านข้างจะแม่นกว่า)", type=["jpg", "jpeg", "png"])
        if file is not None:
            pil_img = Image.open(file).convert("RGB")
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            st.image(img_rgb, caption="ภาพต้นฉบับ", use_container_width=True)

    elif mode == "ถ่ายจากกล้อง":
        picture = st.camera_input("ถ่ายภาพจากกล้อง")
        if picture is not None:
            pil_img = Image.open(picture).convert("RGB")
            img_rgb = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            st.image(img_rgb, caption="ภาพที่ถ่าย", use_container_width=True)

    analyze_btn = st.button("วิเคราะห์")

    if analyze_btn:
        if img_bgr is None:
            st.warning("กรุณาเลือกหรือตั้งค่าภาพก่อน")
        else:
            result_img, side, neck_flex, trunk_flex, knee_angle, level, reason = \
                analyze_posture_yolo_ergonomic(img_bgr, yolo_conf=yolo_conf)

            st.subheader("ผลการวิเคราะห์")

            # layout: ซ้ายภาพ / ขวาข้อมูล
            col_img, col_info = st.columns([2, 1])

            with col_img:
                st.image(result_img, use_container_width=True)

            with col_info:
                st.markdown("**ค่ามุมสำคัญ**")
                st.write(f"- ด้านที่ใช้วิเคราะห์: `{side}`")
                st.write(f"- Neck flex (เบี่ยงจากแนวตรง): " +
                         (f"{neck_flex:.1f}°" if neck_flex is not None else "ไม่สามารถคำนวณได้"))
                st.write(f"- Trunk flex (เบี่ยงจากแนวตรง): " +
                         (f"{trunk_flex:.1f}°" if trunk_flex is not None else "ไม่สามารถคำนวณได้"))
                st.write(f"- Knee angle (มุมเข่า): " +
                         (f"{knee_angle:.1f}°" if knee_angle is not None else "ไม่สามารถคำนวณได้"))

                st.markdown("---")
                st.markdown("**สรุปตามหลักการยศาสตร์**")
                if level == "good":
                    st.success("✅ ท่านั่งดีตามหลักการยศาสตร์")
                elif level == "caution":
                    st.warning("⚠️ ท่านั่งพอใช้ แต่ควรปรับบางจุด")
                elif level == "poor":
                    st.error("❌ ท่านั่งเสี่ยงต่อการปวดเมื่อย/บาดเจ็บ")
                else:
                    st.info("ไม่สามารถประเมินได้แน่ชัดจากภาพนี้")

                if reason:
                    st.markdown("**รายละเอียดเพิ่มเติม**")
                    for r in reason:
                        st.write("- " + r)

# โหมด 3: Livecam
elif mode == "Livecam":
    start_live = st.button("▶ เริ่ม Livecam")

    if start_live:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("ไม่สามารถเปิดกล้องได้ (cv2.VideoCapture(0) ล้มเหลว)")
        else:
            # ทำ layout ซ้าย/ขวา สำหรับ live ด้วย
            col_img, col_info = st.columns([2, 1])
            frame_placeholder = col_img.empty()
            info_placeholder = col_info.empty()

            # รันประมาณ 300 เฟรม
            for _ in range(300):
                ret, frame = cap.read()
                if not ret:
                    st.warning("อ่านเฟรมจากกล้องไม่ได้")
                    break

                result_img, side, neck_flex, trunk_flex, knee_angle, level, reason = \
                    analyze_posture_yolo_ergonomic(frame, yolo_conf=yolo_conf)

                frame_placeholder.image(result_img, use_container_width=True)

                with info_placeholder.container():
                    st.markdown("**ค่ามุม (อัปเดตจากกล้องสด)**")
                    st.write(f"- ด้านที่ใช้วิเคราะห์: `{side}`")
                    st.write(f"- Neck flex: " +
                             (f"{neck_flex:.1f}°" if neck_flex is not None else "N/A"))
                    st.write(f"- Trunk flex: " +
                             (f"{trunk_flex:.1f}°" if trunk_flex is not None else "N/A"))
                    st.write(f"- Knee angle: " +
                             (f"{knee_angle:.1f}°" if knee_angle is not None else "N/A"))

                    st.markdown("---")
                    if level == "good":
                        st.success("✅ ท่านั่งดีตามหลักการยศาสตร์")
                    elif level == "caution":
                        st.warning("⚠️ ท่านั่งพอใช้ แต่ควรปรับบางจุด")
                    elif level == "poor":
                        st.error("❌ ท่านั่งเสี่ยงต่อการปวดเมื่อย/บาดเจ็บ")
                    else:
                        st.info("ยังประเมินได้ไม่ชัดจากเฟรมนี้")

                time.sleep(0.05)

            cap.release()
