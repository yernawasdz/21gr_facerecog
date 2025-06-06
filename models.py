from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Подключение к базе данных SQLite
DATABASE_URL = "sqlite:///./face_recognition.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# Определение таблицы для хранения лиц
class Face(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    encoding = Column(LargeBinary)  # Вектор признаков лица
    image = Column(LargeBinary)     # Изображение лица в формате бинарных данных

# Создание таблицы
Base.metadata.create_all(bind=engine)
