import cv2
import os
import re
import pytesseract
import numpy as np

# Tesseract 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Taewook\.conda\envs\alphaca\Library\bin\tesseract.exe"

# TESSDATA_PREFIX 환경 변수 설정
os.environ["TESSDATA_PREFIX"] = r"C:\Users\Taewook\.conda\envs\alphaca\share\tessdata"

# 실행 확인
try:
    print("Tesseract 버전:", pytesseract.get_tesseract_version())
except Exception as e:
    print("Tesseract 실행 오류:", e)

# 번호판 이미지 로드
image = cv2.imread("license_plate.png")

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 노이즈 제거 및 대비 향상
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# 번호판 영역 검출을 위한 엣지 검출
edges = cv2.Canny(gray, 50, 200)

# 컨투어 찾기
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 사각형 찾기 (번호판 추정)
best_plate = None
for contour in sorted(contours, key=cv2.contourArea, reverse=True):
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:  # 사각형 형태일 경우
        best_plate = approx
        break

# 번호판 영역이 감지되었을 경우 크롭
if best_plate is not None:
    x, y, w, h = cv2.boundingRect(best_plate)
    plate_crop = gray[y:y+h, x:x+w]  # 번호판 부분만 크롭

    # 번호판 부분에 대해서만 OCR 수행
    _, plate_bin = cv2.threshold(plate_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR 실행 (한글 + 숫자 최적화)
    custom_config = "--psm 7 -l kor"
    text = pytesseract.image_to_string(plate_bin, config=custom_config)
else:
    text = "번호판을 감지하지 못했습니다."

clean_text = re.sub(r"^[^가-힣0-9]+|[^가-힣0-9]+$", "", text)
print("추출된 번호판:", clean_text)
