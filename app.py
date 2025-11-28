import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Ergonomic Sitting Posture Checker",
    page_icon="🪑",
    layout="centered"
)

st.title("🪑 ตรวจท่านั่งตามหลักการยศาสตร์")
st.caption("เช็กท่านั่งจากภาพนิ่ง ด้วย YOLO + MediaPipe Pose + กฎยศาสตร์แบบอ่านง่าย")

# โหลดโมเดล YOLO 
@st.cache_resource
def load_yolo_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"ไม่พบไฟล์โมเดล {model_path} ในโฟลเดอร์ปัจจุบัน")
        st.stop()
    model = YOLO(model_path)
    return model

yolo_model = load_yolo_model()

# เตรียม MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,  
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ฟังก์ชันช่วยด้านยศาสตร์-
def calculate_angle(a, b, c):
    """คำนวณมุมระหว่างจุด a-b-c ในหน่วยองศา"""
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
    """แปลงมุมจาก 'สามจุด' ให้กลายเป็นการเบี่ยงจากแนวตรง (180°)"""
    if angle is None:
        return None
    return abs(180.0 - angle)


def choose_side_landmarks(landmarks):
    """เลือกฝั่งซ้าย/ขวาที่เห็นชัดกว่า แล้วคืนจุด ear, shoulder, hip, knee, ankle"""
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
    """สรุประดับท่านั่ง + ข้อความ explanation แบบเป็นมิตร"""
    if neck_flex is None or trunk_flex is None or knee_angle is None:
        return "ยังวิเคราะห์ไม่ครบ", "unknown", [
            "ระบบยังเห็นตัวไม่ชัดทุกจุด ลองนั่งให้กล้องเห็นด้านข้างทั้งตัว หัว–หลัง–เข่า–ข้อเท้าให้ครบก่อนนะ"
        ]

    reason = []

    # คอ
    if neck_flex <= 20:
        reason.append(f"คออยู่ในช่วงดี (เบี่ยงจากแนวตรง ~ {neck_flex:.1f}°)")
        neck_score = 2
    elif neck_flex <= 45:
        reason.append(f"คอเริ่มก้ม/เงย (~ {neck_flex:.1f}°) ลองยกหน้าจอให้สูงขึ้นระดับสายตา")
        neck_score = 1
    else:
        reason.append(f"คอก้ม/เงยมาก (~ {neck_flex:.1f}°) ระวังปวดคอ ลองขยับเก้าอี้หรือยกจอขึ้น")
        neck_score = 0

    # หลัง
    if trunk_flex <= 20:
        reason.append(f"หลังอยู่ในช่วงดี (เบี่ยงจากแนวตรง ~ {trunk_flex:.1f}°)")
        trunk_score = 2
    elif trunk_flex <= 45:
        reason.append(f"หลังเริ่มเอน/งอ (~ {trunk_flex:.1f}°) ลองดันสะโพกให้ชิดพนักพิงมากขึ้น")
        trunk_score = 1
    else:
        reason.append(f"หลังงอเยอะ (~ {trunk_flex:.1f}°) เสี่ยงปวดหลัง ลองเอนหลังพิงเก้าอี้ให้มากขึ้น")
        trunk_score = 0

    # เข่า
    if 80 <= knee_angle <= 120:
        reason.append(f"มุมเข่าอยู่ในช่วงเหมาะสม (~ {knee_angle:.1f}°)")
        knee_score = 2
    else:
        reason.append(f"มุมเข่าอาจไม่ค่อยพอดี (~ {knee_angle:.1f}°) ลองปรับระยะเก้าอี้กับโต๊ะหรือความสูงเก้าอี้")
        knee_score = 1

    total = neck_score + trunk_score + knee_score

    if total >= 5:
        status = "ท่านั่งตอนนี้สบายต่อร่างกายค่อนข้างดีเลย 😊"
        level = "good"
    elif total >= 3:
        status = "ท่านั่งตอนนี้พอใช้ได้ แต่มีบางจุดที่ถ้าขยับอีกนิดจะสบายขึ้นเยอะ 😉"
        level = "caution"
    else:
        status = "ตอนนี้ร่างกายเริ่มเครียดจากท่านั่งนี้ ลองขยับปรับท่าสักหน่อยนะ 🙏"
        level = "poor"

    return status, level, reason


# วิเคราะห์ด้วย MediaPipe ทั้งภาพ (fallback)
def analyze_posture_mediapipe_full(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out, None, None, None, None, "unknown", [
            "ไม่พบโครงร่างบุคคลในภาพ ลองขยับให้อยู่กลางเฟรม และให้กล้องเห็นทั้งตัวมากขึ้น"
        ], "ยังวิเคราะห์ไม่ครบ"

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
    return out_rgb, side, neck_flex, trunk_flex, knee_angle, level, reason, status


# วิเคราะห์ด้วย YOLO (+ fallback)
def analyze_posture_yolo_ergonomic(img_bgr, yolo_conf=0.3):
    h, w, _ = img_bgr.shape
    results = yolo_model(img_bgr, conf=yolo_conf, verbose=False)

    # ถ้า YOLO ไม่เจอเลย → ใช้ MediaPipe ทั้งภาพ
    if len(results) == 0 or len(results[0].boxes) == 0:
        return analyze_posture_mediapipe_full(img_bgr)

    r = results[0]
    boxes = r.boxes

    # เลือกกล่องใหญ่สุด
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        areas.append((x2 - x1) * (y2 - y1))
    idx = int(np.argmax(areas))
    box = boxes[idx]

    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

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
    return out_rgb, side, neck_flex, trunk_flex, knee_angle, level, reason, status

# UI หลัก: upload + camera_input
st.divider()

mode = st.radio(
    "เลือกวิธีใส่ภาพ",
    ["อัปโหลดรูป", "ถ่ายด้วยกล้อง (เบราว์เซอร์)"],
    horizontal=True
)

yolo_conf = st.slider(
    "YOLO confidence",
    0.1, 0.9, 0.3, 0.05,
    help="ปรับความมั่นใจขั้นต่ำของ YOLO ก่อนนับว่าพบคนในภาพ"
)

img_bgr = None
img_rgb = None

if mode == "อัปโหลดรูป":
    file = st.file_uploader(
        "อัปโหลดรูปท่านั่ง (ภาพด้านข้างจะวิเคราะห์ได้ดีที่สุด)",
        type=["jpg", "jpeg", "png"]
    )
    if file is not None:
        pil_img = Image.open(file).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        st.image(img_rgb, caption="ภาพต้นฉบับ", use_container_width=True)

elif mode == "ถ่ายด้วยกล้อง (เบราว์เซอร์)":
    picture = st.camera_input("ถ่ายภาพท่านั่งจากกล้องของเบราว์เซอร์")
    if picture is not None:
        pil_img = Image.open(picture).convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        st.image(img_rgb, caption="ภาพที่ถ่าย", use_container_width=True)

analyze_btn = st.button("🔍 วิเคราะห์ท่านั่งจากภาพนี้")

if analyze_btn:
    if img_bgr is None:
        st.warning("ยังไม่มีภาพให้วิเคราะห์ ลองอัปโหลดหรือถ่ายภาพก่อนนะ 🙂")
    else:
        result_img, side, neck_flex, trunk_flex, knee_angle, level, reason, status = \
            analyze_posture_yolo_ergonomic(img_bgr, yolo_conf=yolo_conf)

        st.subheader("ผลการวิเคราะห์ท่านั่ง")

        col_img, col_info = st.columns([2, 1])

        with col_img:
            st.image(result_img, use_container_width=True)

        with col_info:
            st.markdown("**ภาพรวมท่านั่ง**")
            if level == "good":
                st.success(status)
            elif level == "caution":
                st.warning(status)
            elif level == "poor":
                st.error(status)
            else:
                st.info(status)

            st.markdown("---")
            st.markdown("**ค่ามุมสำคัญ**")
            st.write(f"- ด้านที่ใช้วิเคราะห์: `{side}`")
            st.write(f"- Neck flex (คอเบี่ยงจากแนวตรง): " +
                     (f"{neck_flex:.1f}°" if neck_flex is not None else "ยังคำนวณไม่ได้"))
            st.write(f"- Trunk flex (หลังเบี่ยงจากแนวตรง): " +
                     (f"{trunk_flex:.1f}°" if trunk_flex is not None else "ยังคำนวณไม่ได้"))
            st.write(f"- Knee angle (มุมเข่า): " +
                     (f"{knee_angle:.1f}°" if knee_angle is not None else "ยังคำนวณไม่ได้"))

            if reason:
                st.markdown("---")
                st.markdown("**คำแนะนำเชิงยศาสตร์ (เอาไปใส่รายงานได้เลย)**")
                for r in reason:
                    st.write("• " + r)
