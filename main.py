"""
EquityWalls — FastAPI Backend
==============================
Handles: property listings, user auth, LiDAR data processing,
         Zillow price fetching, blockchain event sync, waitlist.

Install:
    pip install fastapi uvicorn[standard] sqlalchemy pydantic python-jose
                passlib httpx python-dotenv alembic

Run:
    uvicorn main:app --reload --port 8000

API docs auto at: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import os
import math
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional, List

import httpx
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    DateTime, JSON, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker, relationship
from jose import jwt, JWTError
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ─────────────────────────────────────────────────────
# Railway injects postgres:// — SQLAlchemy needs postgresql://
_db_url = os.getenv("DATABASE_URL", "sqlite:///./equitywalls.db")
DATABASE_URL = _db_url.replace("postgres://", "postgresql://", 1)
SECRET_KEY     = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING")
ALGORITHM      = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h

ZILLOW_API_KEY = os.getenv("ZILLOW_API_KEY", "")  # RapidAPI Zillow key
POLYGON_RPC    = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")
PLATFORM_FEE   = 0.03   # 3%
TRADING_FEE    = 0.01   # 1%

# ── DATABASE ────────────────────────────────────────────────────
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email         = Column(String, unique=True, index=True, nullable=False)
    hashed_pw     = Column(String, nullable=False)
    full_name     = Column(String)
    role          = Column(String, default="investor")  # investor | owner | admin
    kyc_status    = Column(String, default="pending")   # pending | approved | rejected
    wallet_address= Column(String)
    created_at    = Column(DateTime, default=datetime.utcnow)
    properties    = relationship("PropertyDB", back_populates="owner")
    holdings      = relationship("HoldingDB",  back_populates="investor")


class PropertyDB(Base):
    __tablename__ = "properties"
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id       = Column(String, ForeignKey("users.id"))
    street_address = Column(String, nullable=False)
    city           = Column(String, nullable=False)
    state          = Column(String, nullable=False)
    zip_code       = Column(String)
    total_sqft     = Column(Float)
    total_squares  = Column(Integer, default=100)
    price_per_square = Column(Float)   # in USD
    total_value    = Column(Float)
    max_equity_pct = Column(Float, default=0.49)
    sold_squares   = Column(Integer, default=0)
    contract_addr  = Column(String)    # deployed Polygon contract
    lidar_hash     = Column(String)    # SHA-256 of point cloud
    lidar_model_url= Column(String)    # S3/IPFS URL of 3D model
    zestimate      = Column(Float)
    status         = Column(String, default="pending")  # pending|active|sold
    llc_confirmed  = Column(Boolean, default=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    owner          = relationship("UserDB", back_populates="properties")
    holdings       = relationship("HoldingDB", back_populates="property")


class HoldingDB(Base):
    __tablename__ = "holdings"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    investor_id   = Column(String, ForeignKey("users.id"))
    property_id   = Column(String, ForeignKey("properties.id"))
    square_ids    = Column(JSON)   # list of square IDs owned
    purchase_price= Column(Float)
    current_value = Column(Float)
    token_ids     = Column(JSON)   # blockchain token IDs
    tx_hash       = Column(String)
    purchased_at  = Column(DateTime, default=datetime.utcnow)
    investor      = relationship("UserDB",     back_populates="holdings")
    property      = relationship("PropertyDB", back_populates="holdings")


class WaitlistDB(Base):
    __tablename__ = "waitlist"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email      = Column(String, unique=True)
    role       = Column(String)   # owner | investor | partner
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ── AUTH ────────────────────────────────────────────────────────
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2  = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> UserDB:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ── PYDANTIC SCHEMAS ────────────────────────────────────────────

class UserCreate(BaseModel):
    email:     EmailStr
    password:  str = Field(min_length=8)
    full_name: str
    role:      str = "investor"

class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    role:         str

class PropertyCreate(BaseModel):
    street_address: str
    city:           str
    state:          str
    zip_code:       str
    total_sqft:     float
    total_squares:  int = 100
    total_value:    float
    max_equity_pct: float = 0.49

class PropertyOut(BaseModel):
    id:              str
    street_address:  str
    city:            str
    state:           str
    total_sqft:      float
    total_squares:   int
    price_per_square:float
    total_value:     float
    sold_squares:    int
    status:          str
    zestimate:       Optional[float]
    llc_confirmed:   bool
    created_at:      datetime
    model_config = {"from_attributes": True}

class LidarUpload(BaseModel):
    property_id:    str
    point_cloud_b64: str   # base64 encoded .ply or .obj binary
    floor_area_sqft: float

class PurchaseRequest(BaseModel):
    property_id: str
    square_ids:  List[int]

class WaitlistEntry(BaseModel):
    email: EmailStr
    role:  str = "investor"


# ── APP ─────────────────────────────────────────────────────────
app = FastAPI(
    title="EquityWalls API",
    description="Fractional real estate equity platform",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── AUTH ROUTES ─────────────────────────────────────────────────

@app.post("/auth/register", response_model=Token, tags=["auth"])
def register(body: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    user = UserDB(
        email      = body.email,
        hashed_pw  = hash_password(body.password),
        full_name  = body.full_name,
        role       = body.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_token({"sub": user.email, "role": user.role})
    return Token(access_token=token, user_id=user.id, role=user.role)

@app.post("/auth/login", response_model=Token, tags=["auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(401, "Invalid credentials")
    token = create_token({"sub": user.email, "role": user.role})
    return Token(access_token=token, user_id=user.id, role=user.role)

@app.get("/auth/me", tags=["auth"])
def me(user: UserDB = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "role": user.role,
            "kyc_status": user.kyc_status, "full_name": user.full_name}


# ── PROPERTY ROUTES ─────────────────────────────────────────────

@app.post("/properties", response_model=PropertyOut, tags=["properties"])
def create_property(
    body: PropertyCreate,
    db:   Session = Depends(get_db),
    user: UserDB  = Depends(get_current_user)
):
    """Homeowner creates a property listing."""
    price_per_sq = (body.total_value * body.max_equity_pct) / body.total_squares
    prop = PropertyDB(
        owner_id       = user.id,
        street_address = body.street_address,
        city           = body.city,
        state          = body.state,
        zip_code       = body.zip_code,
        total_sqft     = body.total_sqft,
        total_squares  = body.total_squares,
        total_value    = body.total_value,
        price_per_square = round(price_per_sq, 2),
        max_equity_pct = body.max_equity_pct,
    )
    db.add(prop)
    db.commit()
    db.refresh(prop)
    return prop

@app.get("/properties", response_model=List[PropertyOut], tags=["properties"])
def list_properties(
    state:  Optional[str] = None,
    status: Optional[str] = "active",
    limit:  int = 20,
    offset: int = 0,
    db:     Session = Depends(get_db)
):
    """List active properties available for investment."""
    q = db.query(PropertyDB)
    if status:
        q = q.filter(PropertyDB.status == status)
    if state:
        q = q.filter(PropertyDB.state == state.upper())
    return q.offset(offset).limit(limit).all()

@app.get("/properties/{property_id}", response_model=PropertyOut, tags=["properties"])
def get_property(property_id: str, db: Session = Depends(get_db)):
    prop = db.query(PropertyDB).filter(PropertyDB.id == property_id).first()
    if not prop:
        raise HTTPException(404, "Property not found")
    return prop


# ── LIDAR PROCESSING ────────────────────────────────────────────

@app.post("/properties/{property_id}/lidar", tags=["lidar"])
async def upload_lidar(
    property_id: str,
    body:        LidarUpload,
    bg:          BackgroundTasks,
    db:          Session = Depends(get_db),
    user:        UserDB  = Depends(get_current_user)
):
    """
    Receive LiDAR point cloud data from iOS app.
    - Compute hash for on-chain anchoring
    - Store 3D model reference
    - Auto-calculate grid dimensions
    """
    prop = db.query(PropertyDB).filter(
        PropertyDB.id == property_id,
        PropertyDB.owner_id == user.id
    ).first()

    if not prop:
        raise HTTPException(404, "Property not found or unauthorized")

    # Compute SHA-256 of point cloud for blockchain anchoring
    import base64
    raw_bytes    = base64.b64decode(body.point_cloud_b64)
    lidar_hash   = hashlib.sha256(raw_bytes).hexdigest()

    prop.lidar_hash    = lidar_hash
    prop.total_sqft    = body.floor_area_sqft

    # Recalculate grid based on measured sqft
    # Each square = 12.5 sqft (adjustable)
    sq_per_sqft = 12.5
    prop.total_squares = max(10, min(500, int(body.floor_area_sqft / sq_per_sqft)))
    prop.price_per_square = round(
        (prop.total_value * prop.max_equity_pct) / prop.total_squares, 2
    )

    db.commit()

    # Background: store raw model (would upload to S3/IPFS in production)
    bg.add_task(mock_store_model, property_id, lidar_hash)

    return {
        "status":         "processed",
        "lidar_hash":     lidar_hash,
        "total_sqft":     body.floor_area_sqft,
        "total_squares":  prop.total_squares,
        "price_per_square": prop.price_per_square,
        "grid_dimensions": _calculate_grid(body.floor_area_sqft),
    }

def _calculate_grid(sqft: float) -> dict:
    """Returns grid cols/rows for the 3D visualization."""
    cols = max(5, min(20, int(math.sqrt(sqft / 10))))
    rows = max(5, min(20, int(sqft / (cols * 10))))
    return {"cols": cols, "rows": rows, "cell_sqft": round(sqft / (cols * rows), 1)}

async def mock_store_model(property_id: str, lidar_hash: str):
    """In production: upload to IPFS/S3, update DB with URL."""
    await asyncio.sleep(0)
    print(f"[BG] Model stored for {property_id} — hash {lidar_hash[:16]}...")


# ── MARKET DATA (ZILLOW) ─────────────────────────────────────────

@app.get("/market/zestimate", tags=["market"])
async def get_zestimate(address: str, zip_code: str):
    """
    Fetch Zestimate from Zillow via RapidAPI.
    Returns estimated value and price/sqft.
    """
    if not ZILLOW_API_KEY:
        # Mock data for development
        return {
            "address":        address,
            "zestimate":      850000,
            "price_per_sqft": 680,
            "yoy_change":     0.054,
            "source":         "mock_dev",
        }

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://zillow-com1.p.rapidapi.com/zestimate",
            params={"address": address, "zip": zip_code},
            headers={
                "X-RapidAPI-Key":  ZILLOW_API_KEY,
                "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com",
            },
            timeout=10,
        )

    if resp.status_code != 200:
        raise HTTPException(502, "Zillow API error")

    data = resp.json()
    return {
        "address":        address,
        "zestimate":      data.get("zestimate", {}).get("amount", {}).get("value"),
        "price_per_sqft": data.get("zestimate", {}).get("valuePerSqFt"),
        "yoy_change":     data.get("zestimate", {}).get("percentChange"),
        "source":         "zillow",
    }

@app.get("/market/comparable-sales", tags=["market"])
async def comparable_sales(city: str, state: str, sqft: float):
    """Returns comparable recent sales for pricing guidance."""
    # Mock implementation — replace with Redfin/Zillow API
    base_price = {"TX": 580, "FL": 890, "CA": 1240, "NY": 1050}.get(state.upper(), 680)
    variance   = 0.08

    return {
        "city":  city,
        "state": state,
        "comps": [
            {"address": f"Nearby Sale {i+1}", "price_sqft": round(base_price * (1 + (i - 1) * variance), 0)}
            for i in range(5)
        ],
        "median_price_sqft": base_price,
        "recommended_price_per_square": round(base_price * (sqft / 100), 2),
    }


# ── PURCHASE (SIMULATED) ─────────────────────────────────────────

@app.post("/invest/purchase", tags=["invest"])
def purchase_squares(
    body: PurchaseRequest,
    db:   Session = Depends(get_db),
    user: UserDB  = Depends(get_current_user)
):
    """
    Simulate a square purchase.
    In production: trigger the smart contract via web3.py.
    """
    if user.kyc_status != "approved":
        raise HTTPException(403, "KYC not approved. Complete identity verification first.")

    prop = db.query(PropertyDB).filter(
        PropertyDB.id == body.property_id,
        PropertyDB.status == "active"
    ).first()

    if not prop:
        raise HTTPException(404, "Property not found or not active")

    max_sellable = int(prop.total_squares * prop.max_equity_pct)
    available    = max_sellable - prop.sold_squares

    if len(body.square_ids) > available:
        raise HTTPException(400, f"Only {available} squares remaining")

    total_cost = len(body.square_ids) * prop.price_per_square

    # Create holding record
    holding = HoldingDB(
        investor_id    = user.id,
        property_id    = prop.id,
        square_ids     = body.square_ids,
        purchase_price = total_cost,
        current_value  = total_cost,
        token_ids      = body.square_ids,  # mirrors square IDs for MVP
        tx_hash        = "0x" + hashlib.sha256(
            f"{user.id}{prop.id}{time.time()}".encode()
        ).hexdigest()[:40],
    )

    prop.sold_squares += len(body.square_ids)

    db.add(holding)
    db.commit()
    db.refresh(holding)

    return {
        "status":      "confirmed",
        "holding_id":  holding.id,
        "squares":     body.square_ids,
        "total_paid":  total_cost,
        "tx_hash":     holding.tx_hash,
        "token_ids":   holding.token_ids,
        "message":     f"You now own {len(body.square_ids)} squares at {prop.street_address}",
    }

@app.get("/invest/portfolio", tags=["invest"])
def get_portfolio(
    db:   Session = Depends(get_db),
    user: UserDB  = Depends(get_current_user)
):
    holdings = db.query(HoldingDB).filter(HoldingDB.investor_id == user.id).all()
    total_invested = sum(h.purchase_price for h in holdings)
    total_current  = sum(h.current_value  for h in holdings)

    return {
        "total_invested": total_invested,
        "total_current":  total_current,
        "unrealized_pnl": total_current - total_invested,
        "total_squares":  sum(len(h.square_ids) for h in holdings),
        "holdings": [
            {
                "id":         h.id,
                "property":   h.property.street_address if h.property else None,
                "squares":    h.square_ids,
                "paid":       h.purchase_price,
                "value":      h.current_value,
                "tx_hash":    h.tx_hash,
                "purchased_at": h.purchased_at,
            }
            for h in holdings
        ]
    }


# ── KYC (STUB — integrate Persona/Onfido in production) ─────────

@app.post("/kyc/submit", tags=["kyc"])
def submit_kyc(
    db:   Session = Depends(get_db),
    user: UserDB  = Depends(get_current_user)
):
    """
    In production: redirect to Persona widget URL.
    Here we simulate KYC approval for dev.
    """
    user.kyc_status = "approved"
    db.commit()
    return {"status": "approved", "message": "KYC verified (dev mode)"}


# ── WAITLIST ─────────────────────────────────────────────────────

@app.post("/waitlist", tags=["waitlist"])
def join_waitlist(body: WaitlistEntry, db: Session = Depends(get_db)):
    if db.query(WaitlistDB).filter(WaitlistDB.email == body.email).first():
        return {"status": "already_registered", "count": db.query(WaitlistDB).count()}
    entry = WaitlistDB(email=body.email, role=body.role)
    db.add(entry)
    db.commit()
    count = db.query(WaitlistDB).count()
    return {"status": "registered", "position": count, "count": count}

@app.get("/waitlist/count", tags=["waitlist"])
def waitlist_count(db: Session = Depends(get_db)):
    return {"count": db.query(WaitlistDB).count()}


# ── HEALTH ───────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "version": "0.1.0", "timestamp": datetime.utcnow().isoformat()}


# ── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
