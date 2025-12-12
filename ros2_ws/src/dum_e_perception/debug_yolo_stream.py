#!/usr/bin/env python3
# python debug_yolo_stream.py \
#   --model /home/rokey/DUM-E/ros2_ws/src/dum_e_perception/models/yoloe-11m-seg.pt \
#   --device cuda \
#   --camera 0 \
#   --classes "scissors,screwdriver,water bottle" \
#   --conf 0.15

import argparse
import time

import cv2
from ultralytics import YOLO


def build_argparser():
    p = argparse.ArgumentParser(
        description="YOLO-World 실시간 카메라 디버그 스트리머 (open-vocab)"
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="YOLO World 모델 경로 (예: yolov8s-worldv2.pt)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="추론 디바이스: 'cuda' 또는 'cpu' (기본: cuda)",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="카메라 인덱스 (기본: 0)",
    )
    p.add_argument(
        "--classes",
        type=str,
        default="scissors,screwdriver,water bottle",
        help="콤마로 구분된 open-vocab 프롬프트 리스트 "
             "(예: 'scissors,screwdriver,water bottle')",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="confidence threshold (기본: 0.15)",
    )
    return p


def main():
    args = build_argparser().parse_args()

    # 1) 모델 로드
    print(f"[DEBUG] 모델 로드 중: {args.model} (device={args.device})")
    model = YOLO(args.model)
    model.to(args.device)

    # CLIP 텍스트 인코더도 같은 디바이스로 강제 이동 (device mismatch 방지)
    try:
        clip_model = model.model.clip_model
        clip_model.to(args.device)
        print("[DEBUG] CLIP 텍스트 인코더 device 정렬 완료")
    except AttributeError:
        print("[DEBUG] clip_model 없음 (일부 모델에서는 정상)")

    # 2) 클래스 프롬프트 설정
    class_prompts = [c.strip() for c in args.classes.split(",") if c.strip()]
    print(f"[DEBUG] set_classes: {class_prompts}")
    if class_prompts:
        model.set_classes(class_prompts)
    else:
        model.set_classes(None)  # all classes

    # 3) 카메라 열기
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] 카메라를 열 수 없습니다: index={args.camera}")
        return

    print("[DEBUG] 실시간 스트리밍 시작! 창에서 'q' 키를 누르면 종료합니다.")
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 프레임을 읽지 못했습니다.")
                break

            # YOLO는 BGR np.ndarray를 바로 받아도 됨
            # conf threshold, verbose off
            results = model(frame, conf=args.conf, verbose=False)[0]

            # FPS 계산
            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now

            annotated = frame.copy()

            h, w, _ = frame.shape
            names = results.names if hasattr(results, "names") else model.names

            for box, cls, conf in zip(
                results.boxes.xyxy,
                results.boxes.cls,
                results.boxes.conf,
            ):
                x1, y1, x2, y2 = box.cpu().numpy()
                cls_id = int(cls.cpu().numpy())
                conf_val = float(conf.cpu().numpy())
                class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

                # 바운딩 박스, 라벨 렌더링
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)

                label = f"{class_name} {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    annotated,
                    (p1[0], p1[1] - th - 4),
                    (p1[0] + tw, p1[1]),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (p1[0], p1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # 왼쪽 위에 FPS / class prompt 정보 표시
            info_text = f"FPS: {fps:.1f} | classes: {class_prompts}"
            cv2.putText(
                annotated,
                info_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLO-World Debug Stream", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[DEBUG] 'q' 입력, 종료합니다.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
