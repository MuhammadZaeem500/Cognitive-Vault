import hashlib
import os
from sqlalchemy import Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DB_FILE = "cognitive_vault.db"
engine = create_engine(
    f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserDB(Base):
  __tablename__ = "users"
  id = Column(String, primary_key=True, index=True)
  name = Column(String, nullable=True)
  email = Column(String, unique=True, index=True, nullable=False)
  hashed_password = Column(String, nullable=False)


def init_db():
  Base.metadata.create_all(bind=engine)


def get_password_hash(password: str) -> str:
  salt = os.urandom(16)
  pwd_hash = hashlib.pbkdf2_hmac(
      "sha256", password.encode("utf-8"), salt, 100000
  )
  return salt.hex() + ":" + pwd_hash.hex()


def verify_password(plain_password: str, stored_hash: str) -> bool:
  try:
    salt_hex, hash_hex = stored_hash.split(":")
    salt = bytes.fromhex(salt_hex)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256", plain_password.encode("utf-8"), salt, 100000
    )
    return pwd_hash.hex() == hash_hex
  except Exception:
    return False