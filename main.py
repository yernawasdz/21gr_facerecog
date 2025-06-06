from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import os
import cv2
import face_recognition
import numpy as np
from models import Face, SessionLocal
from io import BytesIO
import config

app = FastAPI()

# Путь для OAuth2 и создание схемы
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Конфигурация базы пользователей (в данном случае демонстрационная, без подключения к настоящей базе)
fake_users_db = {
    "user@example.com": {
        "username": "user@example.com",
        "full_name": "John Doe",
        "email": "user@example.com",
        "hashed_password": "$2b$12$examplehashedpassword",  # Пример, можно заменить на настоящий пароль
        "disabled": False,
    }
}

# Модели для токена и пользователя
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# Проверка пароля (упрощённый вариант, не использует хеширование)
def verify_password(plain_password, hashed_password):
    return plain_password == hashed_password

# Функция для поиска пользователя в базе данных
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

# Аутентификация пользователя
def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

# Создание JWT токена
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt

# Эндпоинт для получения токена доступа
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Зависимость для проверки токена
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Не удалось проверить учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Пример защищенного эндпоинта
@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Подключаем папку для статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Функция для обрезания лица с дополнительными 20 пикселями
def crop_face_with_padding(img, top, right, bottom, left, padding=20):
    top = max(0, top - padding)
    right = min(img.shape[1], right + padding)
    bottom = min(img.shape[0], bottom + padding)
    left = max(0, left - padding)
    return img[top:bottom, left:right]

# Возвращает фронтенд файл index.html
@app.get("/")
async def get_frontend():
    return FileResponse(os.path.join("static", "index.html"))

# Эндпоинт для множественной загрузки лиц и их сохранения в базе данных
@app.post("/upload_multiple_faces/")
async def upload_multiple_faces(files: List[UploadFile] = File(...), tolerance: float = 0.4, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    saved_faces = []

    for file in files:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Обнаруживаем лица и сортируем их слева направо
        face_locations = face_recognition.face_locations(img)
        face_locations = sorted(face_locations, key=lambda x: x[3])
        file_faces = []

        if len(face_locations) == 0:
            saved_faces.append({"file": file.filename, "faces": [{"message": "Лица не обнаружены"}]})
            continue

        for (top, right, bottom, left) in face_locations:
            encoding = face_recognition.face_encodings(img, [(top, right, bottom, left)])[0]

            # Проверка на совпадения в базе данных
            all_faces = db.query(Face).all()
            match_found = False
            for face in all_faces:
                saved_encoding = np.frombuffer(face.encoding, dtype=np.float64)
                if face_recognition.compare_faces([saved_encoding], encoding, tolerance=tolerance)[0]:
                    file_faces.append({"message": "Лицо уже существует в базе данных", "id": face.id, "name": face.name})
                    match_found = True
                    break

            if not match_found:
                # Сохранение лица с временным именем
                face_image = crop_face_with_padding(img, top, right, bottom, left)
                _, face_image_encoded = cv2.imencode('.jpg', face_image)
                new_face = Face(name="temp", encoding=encoding.tobytes(), image=face_image_encoded.tobytes())
                db.add(new_face)
                db.commit()
                db.refresh(new_face)

                # Обновление имени на основе ID
                new_face.name = f"Person_{new_face.id}"
                db.commit()
                file_faces.append({"id": new_face.id, "name": new_face.name})

        saved_faces.append({"file": file.filename, "faces": file_faces})

    db.close()
    return {"message": "Обработка завершена", "results": saved_faces}

# Эндпоинт для множественного распознавания лиц
@app.post("/recognize_multiple_faces/")
async def recognize_multiple_faces(files: List[UploadFile] = File(...), tolerance: float = 0.4, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    all_faces = db.query(Face).all()
    results = []

    for file in files:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Обнаруживаем все лица и сортируем их слева направо
        face_locations = face_recognition.face_locations(img)
        face_locations = sorted(face_locations, key=lambda x: x[3])
        encodings = face_recognition.face_encodings(img, face_locations)

        file_faces = []
        for encoding in encodings:
            matches = []
            for face in all_faces:
                saved_encoding = np.frombuffer(face.encoding, dtype=np.float64)
                if face_recognition.compare_faces([saved_encoding], encoding, tolerance=tolerance)[0]:
                    matches.append({"id": face.id, "name": face.name})
            file_faces.append({"matches": matches if matches else "No match found"})

        results.append({"file": file.filename, "faces": file_faces})

    db.close()
    return {"message": "Распознавание завершено", "results": results}

# Эндпоинт для получения изображения лица по его ID
@app.get("/get_face_image/{face_id}")
def get_face_image(face_id: int, current_user: User = Depends(get_current_user)):
    db = SessionLocal()
    face = db.query(Face).filter(Face.id == face_id).first()

    if face is None:
        raise HTTPException(status_code=404, detail="С таким ID нет изображения лица")

    return StreamingResponse(BytesIO(face.image), media_type="image/jpeg")
