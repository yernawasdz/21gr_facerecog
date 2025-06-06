import os
import requests

# URL эндпоинта для множественного распознавания лиц
url = "http://localhost:8000/recognize_multiple_faces/"

# Путь к папке с изображениями для распознавания
folder_path = "images_to_recognize"  # Укажите здесь путь к папке с вашими изображениями

# Сопоставление префиксов с actual ID
prefix_to_id = {
    "donald trump": 1,
    "gates": 2,
    "jack": 3,
    "modi": 4,
    "musk": 5
}

# Счетчики для подсчета успешных и неуспешных распознаваний
correct_matches = 0
total_faces = 0

# Проход по всем файлам в папке
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # Только изображения
        file_path = os.path.join(folder_path, filename)

        # Определяем actual_id на основе префикса имени файла
        actual_id = None
        for prefix, id_value in prefix_to_id.items():
            if filename.lower().startswith(prefix):  # Приводим к нижнему регистру для надежности
                actual_id = id_value
                break

        if actual_id is None:
            print(f"Пропущен файл {filename}: не найден соответствующий префикс")
            continue

        # Открываем файл изображения
        with open(file_path, "rb") as image_file:
            # Отправляем POST-запрос с файлом изображения
            response = requests.post(url, files={"files": (filename, image_file, "image/jpeg")})

            # Проверяем успешность ответа
            if response.status_code == 200:
                try:
                    response_data = response.json()

                    # Выводим response_data для диагностики структуры
                    print(f"Ответ сервера для файла {filename}:\n{response_data}")

                    # Обработка результата для каждого лица в файле
                    for file_result in response_data.get("results", []):
                        face_number = 1
                        for face_result in file_result.get("results", []):
                            # Проверяем, является ли face_result["matches"] списком
                            if isinstance(face_result.get("matches"), list):
                                predicted_ids = [match.get("id") for match in face_result["matches"]]
                            else:
                                predicted_ids = []  # Если нет совпадений, устанавливаем predicted_ids как пустой список

                            # Проверка на совпадение actual и predicted
                            if actual_id in predicted_ids:
                                correct_matches += 1
                                print(f"{filename} - Лицо {face_number}: Правильно распознано (actual: {actual_id}, predicted: {predicted_ids})")
                            else:
                                print(f"{filename} - Лицо {face_number}: Неправильно распознано (actual: {actual_id}, predicted: {predicted_ids})")

                            face_number += 1
                            total_faces += 1

                except ValueError:
                    print(f"Ошибка: не удалось декодировать JSON для файла {filename}. Ответ: {response.text}")
            else:
                print(f"Ошибка при обработке {filename}: {response.status_code} - {response.text}")

# Подсчет и вывод процента правильных распознаваний
if total_faces > 0:
    accuracy = (correct_matches / total_faces) * 100
    print(f"\nПроцент правильного распознавания: {accuracy:.2f}%")
else:
    print("Нет файлов для распознавания.")
