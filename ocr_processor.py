import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch

try:
    from ultralytics import YOLO
    yolo_imported = True
except ImportError:
    print("Не удалось импортировать YOLO")
    yolo_imported = False

try:
    # Библиотека для распознавания текста
    import easyocr
    easyocr_imported = True
except ImportError:
    print("Не удалось импортировать EasyOCR")
    easyocr_imported = False

# --- Конфигурация ---
BASE_IMAGE_FOLDER = "ocr_project/images"
YOLO_MODEL_PATH = "ocr_project/models/yolo11s.pt"
FONT_PATH = "ocr_project/fonts/GalindoCyrillic-Regular.ttf"

OCR_CONFIDENCE_THRESHOLD = 0.3

def initialize_models():
    """
    Загружает модели YOLO и EasyOCR 
    """

    yolo_model = None
    if yolo_imported and os.path.exists(YOLO_MODEL_PATH):
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[YOLO] Модель успешно загружена из {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[YOLO] Ошибка загрузки модели: {e}")
    else:
        print(f"[YOLO] Модель не найдена")

    easyocr_reader = None
    if easyocr_imported:
        try:
            use_gpu = torch.cuda.is_available()
            print(f"[INFO] Использовать GPU: {use_gpu}")

            easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=use_gpu)
            print("[OCR] EasyOCR инициализирован.")
        except Exception as e:
            print(f"[OCR] Ошибка инициализации EasyOCR: {e}")

    print("--- Инициализация завершена ---\n")
    return yolo_model, easyocr_reader


def detect_text_surfaces_yolo(image: np.ndarray, yolo_model) -> list:
    """
    Находит на изображении прямоугольные области (плашки), где может быть текст.

    Args:
        image (np.ndarray): Изображение в формате OpenCV (BGR).
        yolo_model: Загруженная модель YOLO.

    Returns:
        list: Список кортежей с координатами (x1, y1, x2, y2) для каждой найденной области.
    """
    if yolo_model is None:
        print("      [YOLO] Модель не загружена, детекция невозможна.")
        return []

    results = yolo_model(image)
    
    bounding_boxes = []
    if results and results[0].boxes:
        # Получаем координаты найденных боксов
        boxes_data = results[0].boxes.xyxy.cpu().numpy()
        for box in boxes_data:
            x1, y1, x2, y2 = map(int, box)
            bounding_boxes.append((x1, y1, x2, y2))
            
    return bounding_boxes

def preprocess_image_for_ocr(roi_image: np.ndarray) -> np.ndarray:
    """
    Превращает цветное изображение в черно-белое с повышенным контрастом.

    Args:
        roi_image (np.ndarray): Цветной вырезанный фрагмент изображения.

    Returns:
        np.ndarray: Обработанное черно-белое изображение.
    """
    # 1. Преобразуем в оттенки серого
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Увеличиваем локальный контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    _, binary_image = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return binary_image

def recognize_text_easyocr(preprocessed_roi: np.ndarray, easyocr_reader) -> list:
    """
    Распознает текст на предобработанном фрагменте с помощью EasyOCR.

    Args:
        preprocessed_roi (np.ndarray): Черно-белый фрагмент изображения.
        easyocr_reader: Загруженная модель EasyOCR.

    Returns:
        list: Список результатов, где каждый элемент - это кортеж
              (координаты, текст, уверенность).
    """
    if easyocr_reader is None:
        print("      [EasyOCR] Модель не загружена, распознавание невозможно.")
        return []
    
    try:
        ocr_output = easyocr_reader.readtext(preprocessed_roi)
        
        results = []
        for (bbox, text, confidence) in ocr_output:
            points = [[int(p[0]), int(p[1])] for p in bbox]
            results.append((points, text, confidence))
        return results
    except Exception as e:
        print(f"      [EasyOCR] Ошибка во время распознавания: {e}")
        return []

def visualize_results(image: np.ndarray, yolo_boxes: list, all_ocr_results: list, title: str):
    """
    Рисует на изображении рамки от YOLO и распознанный текст от OCR.

    Args:
        image (np.ndarray): Оригинальное изображение.
        yolo_boxes (list): Список рамок от YOLO.
        all_ocr_results (list): Список всех распознанных текстовых блоков.
        title (str): Заголовок для окна с изображением.
    """
    # Конвертируем изображение из формата OpenCV (BGR) в формат Pillow (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        # Пытаемся загрузить красивый шрифт, если не получится - используем стандартный
        font = ImageFont.truetype(FONT_PATH, size=20)
    except IOError:
        font = ImageFont.load_default()

    # Рисуем красные рамки там, где YOLO нашла плашки с текстом
    for (x1, y1, x2, y2) in yolo_boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Рисуем рамки и подписываем распознанный текст
    for (bbox_abs, text, conf) in all_ocr_results:
        color = "cyan" # Цвет для EasyOCR
        points = [tuple(p) for p in bbox_abs]
        if len(points) > 1:
            # Рисуем точную рамку вокруг слова
            draw.polygon(points, outline=color, width=2)
            # Формируем текст для подписи: "СЛОВО (99%)"
            display_text = f"{text} ({conf:.2f})"
            # Пишем текст над рамкой
            draw.text(points[0], display_text, fill=color, font=font)

    plt.figure(figsize=(16, 12))
    plt.imshow(pil_img)
    plt.title(title, fontsize=18)
    plt.axis('off') 
    plt.show()

def process_image_pipeline(image_path: str, yolo_model, easyocr_reader):
    """
    Выполняет полный цикл обработки для одного изображения:
    1. Читает файл.
    2. Находит области с текстом (YOLO).
    3. Для каждой области распознает текст (EasyOCR).
    4. Визуализирует результат.
    """
    print(f"\n>>> Обработка изображения: {os.path.basename(image_path)}")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("    Не удалось прочитать файл.")
        return

    # Найти все плашки с текстом
    yolo_boxes = detect_text_surfaces_yolo(original_image, yolo_model)
    if not yolo_boxes:
        print("    [YOLO] Не найдено поверхностей с текстом.")
        visualize_results(original_image, [], [], f"Поверхности не найдены: {os.path.basename(image_path)}")
        return
    print(f"    [YOLO] Найдено {len(yolo_boxes)} поверхностей.")

    all_ocr_results = []
    # Перебираем каждую найденную YOLO плашку
    for i, (x1, y1, x2, y2) in enumerate(yolo_boxes):
        roi = original_image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        print(f"      - Обработка поверхности #{i+1}...")
        
        # Улучшаем качество вырезанного фрагмента
        preprocessed_roi = preprocess_image_for_ocr(roi)

        # рисуем вырезанную часть
        plt.figure() 
        plt.imshow(preprocessed_roi, cmap='gray')
        plt.title(f"Предобработанный ROI #{i+1}")
        plt.axis('off') 
        plt.show()
        
        ocr_results_relative = recognize_text_easyocr(preprocessed_roi, easyocr_reader)

        # Фильтруем результаты по уверенности
        filtered_ocr_results = []
        for res in ocr_results_relative:
            # res[2] - это уверенность (confidence)
            if res[2] > OCR_CONFIDENCE_THRESHOLD:
                filtered_ocr_results.append(res)

        if filtered_ocr_results:
            print(f"        Найдено и отфильтровано {len(filtered_ocr_results)} текстовых блоков (порог > {OCR_CONFIDENCE_THRESHOLD}):")
            for _, text, conf in filtered_ocr_results:
                print(f"          - '{text}' [Уверенность: {conf:.2f}]")
        else:
            print(f"        Текстовых блоков с уверенностью > {OCR_CONFIDENCE_THRESHOLD} не найдено.")

        # ШАГ 5: Пересчитываем относительные координаты в абсолютные (для всего изображения)
        for bbox_rel, text, conf in filtered_ocr_results:
            # Прибавляем координаты левого верхнего угла ROI (x1, y1)
            bbox_abs = [[p[0] + x1, p[1] + y1] for p in bbox_rel]
            all_ocr_results.append((bbox_abs, text, conf))

    

    # рисуем все, что нашли, на оригинальном изображении
    visualize_results(
        original_image,
        yolo_boxes,
        all_ocr_results,
        title=f"Результат обработки: {os.path.basename(image_path)}"
    )

if __name__ == "__main__":

    yolo_model, easyocr_reader = initialize_models()

    # 2. Проверяем, что все загрузилось успешно
    if yolo_model is None or easyocr_reader is None:
        print("\nКритическая ошибка: не удалось загрузить модель YOLO или EasyOCR. Проверьте пути и установки.")
    else:
        # 3. Находим все картинки в папке
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(BASE_IMAGE_FOLDER, ext)))

        if not image_paths:
            print(f"В папке '{BASE_IMAGE_FOLDER}' не найдено изображений.")
        else:
            # 4. Обрабатываем каждую картинку по очереди
            for image_path in image_paths:
                process_image_pipeline(image_path, yolo_model, easyocr_reader)