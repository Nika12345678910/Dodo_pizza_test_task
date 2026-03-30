from collections import deque
from datetime import datetime
from enum import Enum
import logging
import os
from pathlib import Path
import sys
import traceback
from types import NoneType
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


import cv2
from cv2.typing import MatLike
from ultralytics import YOLO


class LogLevel(Enum):
    INFO = "info"
    ERROR = "error"
    WARNING = "warning"
    DEBUG = "debug"
    CRITICAL = "critical"


class StatesTable(Enum):
    empty = "стол пустой"
    occupied = "стол занят"
    approached = "к столу подошли"


class StateAnalyzer:
    def __init__(self):
        self._active_state = None
        self._start_time = None
        self._records = []

    def set_state(self, state):
        now = datetime.now()
        if self._active_state is not None and self._active_state != state:
            # Фиксируем завершение предыдущей стадии
            duration_seconds = (now - self._start_time).total_seconds()
            self._records.append({
                'стадия': self._active_state.value,
                'время начала': self._start_time.strftime('%H:%M:%S'),
                'время конца': now.strftime('%H:%M:%S'),
                'итого времени': duration_seconds
            })
        # Устанавливаем новое состояние
        if self._active_state != state:
            self._active_state = state
            self._start_time = now

    def get_dataframe(self):
        df_records = self._records.copy()  # Работаем с копией, чтобы не портить оригинал

        # Если на момент запроса есть активное состояние, "закрываем" его текущим временем
        if self._active_state is not None:
            now = datetime.now()
            duration_seconds = (now - self._start_time).total_seconds()
            df_records.append({
                'стадия': self._active_state.value,
                'время начала': self._start_time.strftime('%H:%M:%S'),
                'время конца': now.strftime('%H:%M:%S'),
                'итого времени': duration_seconds
            })

        if not df_records:
            return pd.DataFrame(columns=['стадия', 'время начала', 'время конца', 'итого времени'])

        df = pd.DataFrame(df_records)
        df['итого времени'] = pd.to_numeric(
            df['итого времени'], errors='coerce')

        return df


class StateDetector:
    def __init__(self, video_path: str,  model_path='yolov8n.pt'):
        self.video_path = video_path

        self.x, self.y, self.w, self.h = 1320, 94, 542, 179
        # ЕСЛИ ХОТИТЕ ВЫБРАТЬ ROI - раскомментируйте строку ниже и закомментируйте строку выше
        # self.x, self.y, self.w, self.h = self.__get_coordinates_roi()

        self.confidence_threshold = 0.5  # Порог уверенности детекции
        self.iou_threshold = 0.1  # Порог IoU для лучшего обнаружения

        # История состояния стола за последние 3 сек:
        # 1 - человек в зоне, 0 - человек не в зоне
        self.detection_history: list[0 | 1] = deque(maxlen=55)

        self.occupancy_threshold = 0.7  # Порог занятости стола
        self.empty_threshold = 0.05  # Порог пустоты стола

        # Загрузка модели YOLO
        self.model = YOLO(model_path)

        self.state_manager = StateManager()

    def _check_person_in_roi(self, detections: np.ndarray):
        for det in detections:
            # Проверяем, что это человек (класс 0 в COCO)
            if int(det[5]) == 0 and float(det[4]) >= self.confidence_threshold:
                # Получаем координаты bounding box
                x1, y1, x2, y2 = map(int, det[:4])

                # Проверяем пересечение с ROI
                if self._is_intersecting_roi(x1, y1, x2, y2):
                    return True
        return False

    def _state_analytic(self, detections):
        has_person = self._check_person_in_roi(detections=detections)

        # Добавляем результат в историю
        self.detection_history.append(1 if has_person else 0)

        # Вычисляем процент кадров с детекцией
        if len(self.detection_history) > 0:
            self.current_occupancy_ratio = sum(
                self.detection_history) / len(self.detection_history)

        # Если человек есть больше чем на (occupancy_threshold*100)% фреймов, то он занял стол
        if self.current_occupancy_ratio >= self.occupancy_threshold and has_person:
            self.state_manager.set_state(state=StatesTable.occupied)
        # Если человек есть меньше чем на (occupancy_threshold*100)% фреймов
        # и больше чем на (empty_threshold*100)% фреймов, то он подошел к столу
        elif (self.empty_threshold <= self.current_occupancy_ratio <= self.occupancy_threshold
              and self.state_manager.current_state == StatesTable.empty):
            self.state_manager.set_state(state=StatesTable.approached)
        # Если человек есть меньше чем на (empty_threshold*100)% фреймов, то стол пустой
        elif self.current_occupancy_ratio <= self.empty_threshold:
            self.state_manager.set_state(state=StatesTable.empty)

    def _is_intersecting_roi(self, x1, y1, x2, y2):
        # Вычисляем площадь пересечения
        roi_x1, roi_y1 = self.x, self.y
        roi_x2, roi_y2 = self.x + self.w, self.y + self.h

        # Координаты пересечения
        inter_x1 = max(x1, roi_x1)
        inter_y1 = max(y1, roi_y1)
        inter_x2 = min(x2, roi_x2)
        inter_y2 = min(y2, roi_y2)

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            person_area = (x2 - x1) * (y2 - y1)

            # Если пересечение составляет более (iou_threshold*100)% от площади человека
            # или человек полностью внутри ROI
            if intersection_area / person_area > self.iou_threshold:
                return True

        return False

    def _frame_rendering(self, frame: MatLike):
        # Выполняем детекцию YOLO на всем кадре
        results = self.model(frame, verbose=False)

        # Получаем детекции
        if len(results) > 0 and results[0].boxes is not None:
            detections = results[0].boxes.data.cpu().numpy()
        else:
            detections = np.array([])

        # Отслеживаем состояние
        self._state_analytic(detections)

        # Отображаем информацию на кадре
        self._draw_roi(frame, detections)

        return frame

    def _get_color(self) -> tuple[int, int, int]:
        # Красный - стол занят
        # Оранжевый - к столу подошли
        # Зеленый - стол свободен

        if self.state_manager.current_state == StatesTable.approached:
            return (0, 110, 255)
        elif self.state_manager.current_state == StatesTable.occupied:
            return (0, 0, 255)
        elif self.state_manager.current_state == StatesTable.empty:
            return (0, 255, 0)

    def _draw_roi(self, frame: MatLike, detections: np.ndarray):
        # Рисуем ROI
        color = self._get_color()

        # Рисуем прямоугольник
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.w, self.y + self.h), color, 3)

        # Добавляем фон для текста
        text_size = cv2.getTextSize(
            self.state_manager.current_state.value, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Текст и фон над прямоугольником
        cv2.rectangle(frame, (self.x, self.y - text_size[1] - 5),
                      (self.x + text_size[0] + 10, self.y), (0, 0, 0), -1)
        cv2.putText(frame,
                    self.state_manager.current_state.value,
                    (self.x + 5, self.y - 8),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    def __get_coordinates_roi(self):
        cap = cv2.VideoCapture(self.video_path)

        # Читаем первый кадр для выбора ROI
        ret, frame = cap.read()
        if not ret:
            self.state_manager.log_msg(
                msg="Ошибка загрузки видео", level=LogLevel.ERROR)
            cap.release()
            exit()

        # Подгоняем разрешение видео под разрешение экрана,
        # чтобы на любом устройстве видео было видно целиком
        cv2.namedWindow('Enter ROI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Enter ROI', 1920, 1080)

        # Выбираем ROI на первом кадре
        roi = cv2.selectROI("Enter ROI", frame, False, False)
        cv2.destroyWindow("Enter ROI")

        # Проверяем, что ROI выбран
        if roi == (0, 0, 0, 0):
            self.state_manager.log_msg(
                msg="ОшибкаOI не выбран. Выход...", level=LogLevel.ERROR)
            cap.release()
            exit()

        x, y, w, h = map(int, roi)

        cap.release()

        print("x, y, w, h:", x, y, w, h)

        return (x, y, w, h)

    def save_video(self, output_path: str = "output.mp4"):
        try:
            # Открываем видео
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception(f"Не удалось открыть видео: {self.video_path}")

            # Получаем параметры видео
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Создаем VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height))

            if not video_writer.isOpened():
                raise Exception(f"Не удалось создать видеофайл: {output_path}")

            print(
                f"Мониторинг ROI: x={self.x}, y={self.y}, w={self.w}, h={self.h}")
            print(f"FPS видео: {fps:.2f}")
            print("-" * 50)
            print(f"Начало обработки видео: {self.video_path}")
            print(f"Результат будет сохранен в: {output_path}")

            frame_count = 0
            start_time = cv2.getTickCount()

            # Обрабатываем кадры
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Обрабатываем кадр (рисуем ROI и детекции)
                frame = self._frame_rendering(frame)

                # Сохраняем кадр
                video_writer.write(frame)

                frame_count += 1

                # Выводим прогресс каждые 100 кадров
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"Прогресс: {progress:.1f}% ({frame_count}/{total_frames} кадров)")

            # Подсчет времени обработки
            end_time = cv2.getTickCount()
            processing_time = (end_time - start_time) / cv2.getTickFrequency()

            # Освобождаем ресурсы
            cap.release()
            video_writer.release()

            print(f"\nОбработка завершена!")
            print(f"Обработано кадров: {frame_count}")
            print(f"Время обработки: {processing_time:.2f} секунд")
            print(f"Видео сохранено: {output_path}")

            return output_path
        except Exception as e:
            self.state_manager.log_msg(e, level=LogLevel.ERROR)
        finally:
            # Выводим статистику в логи и вывод программы
            df = self.state_manager.state_analyzer.get_dataframe()
            self.state_manager.log_msg(msg=f"\n{df.to_string(index=False)}")
            print("-" * 50)
            print(
                "Анализ состояний. Более подробно можно просмотреть в файле 'logs.log':")
            print(df)

    def show_video(self):
        try:
            # Открываем видео
            cap = cv2.VideoCapture(self.video_path)

            # Подгоняем разрешение видео под разрешение экрана,
            # чтобы на любом устройстве видео было видно целиком
            cv2.namedWindow('StateDetector', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('StateDetector', 2560, 1440)

            if not cap.isOpened():
                self.state_manager.log_msg(
                    msg="Ошибка открытия видео", level=LogLevel.ERROR)
                return

            fps = cap.get(cv2.CAP_PROP_FPS)

            # Отладочная инфа
            print(
                f"Мониторинг ROI: x={self.x}, y={self.y}, w={self.w}, h={self.h}")
            print(f"FPS видео: {fps:.2f}")
            print("-" * 50)

            # Процесс обработки
            while True:
                ret, frame = cap.read()

                if not ret:
                    # Если видео закончилось, перематываем в начало или выходим
                    print("Видео закончилось")
                    break

                # Обрабатываем кадр
                frame = self._frame_rendering(frame)

                # Показываем кадр
                cv2.imshow('StateDetector', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Выход по клавише 'q'
                    break
        except Exception as e:
            self.state_manager.log_msg(e, level=LogLevel.ERROR)
        finally:
            # Выводим статистику в логи и вывод программы
            df = self.state_manager.state_analyzer.get_dataframe()
            self.state_manager.log_msg(msg=f"\n{df.to_string(index=False)}")
            print(df)

    def __del__(self):
        # Освобождаем ресурсы
        cv2.destroyAllWindows()


class StateManager:
    def __init__(self):
        # Настраиваем логи
        logging.basicConfig(
            filename="logs.log",
            filemode="w",
            encoding="utf-8",
            level=logging.INFO,  # Уровень логирования
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger(__name__)
        self.current_state = None
        self.old_state = None

        # Объект для ведения аналитики
        self.state_analyzer = StateAnalyzer()

    def set_state(self, state: StatesTable):
        # Проверяем тип state
        if not isinstance(state, StatesTable):
            self.log_msg("Invalid type for variable 'state'",
                         level=LogLevel.ERROR)

        self.old_state = self.current_state
        self.current_state = state
        if self.old_state != self.current_state:
            # Записываем состояние в таблицу
            self.state_analyzer.set_state(state=state)
            # Логируем изменение состояния
            self.log_state()

    def log_state(self):
        # Проверяем критически значимый параметр
        if isinstance(self.current_state, NoneType):
            self.log_msg("Invalid self.current_state", level=LogLevel.ERROR)
            return

        # Записываем логи
        if self.old_state:
            self.logger.info(
                f"Set of the main event from '{self.old_state.value}' to '{self.current_state.value}'")
        else:
            self.logger.info(
                f"Set of the main event to '{self.current_state.value}'")

    def log_msg(self, msg: str, level: LogLevel = LogLevel.INFO):
        # Проверяем критически значимый параметр
        if isinstance(msg, str):
            level = level.value.lower()
            log_method = getattr(self.logger, level, NoneType)
            if log_method and callable(log_method):
                # Записываем сообщение
                log_method(msg=msg)
                if level == LogLevel.CRITICAL or level == LogLevel.ERROR:
                    # Если ошибка - записываем Traceback
                    log_method(msg=f"Traceback:\n{traceback.format_exc()}")
            else:
                self.logger.warning(
                    f"Unknown log level '{level}'. Defaulting to info.")
                self.logger.info(msg=msg)
        else:
            self.logger.error(
                f"Invalid message type. Expected type str, but received type {str(type(msg))}.")
            self.logger.error(msg=f"Traceback:\n{traceback.format_exc()}")


def main():
    # Проверяем, передан ли аргумент
    if len(sys.argv) < 3 or sys.argv[1] != '--video':
        print("Использование: python main.py --video <путь_к_видео>")
        sys.exit(1)

    # Получаем путь к файлу (второй аргумент после --video)
    video_path = sys.argv[2]

    print(f"Обрабатываю видео: {video_path}")

    # Запускаем основную логику работы программы
    detector = StateDetector(video_path=video_path)
    detector.save_video()
    # Если не хотите ждать загрузки видео - раскомментируйте строку ниже и закомментируйте строку выше
    # detector.show_video()


if __name__ == "__main__":
    main()
