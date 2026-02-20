from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    id: int
    is_verified_seller: bool


@dataclass
class Item:
    id: int
    user_id: int
    name: str
    description: str
    category: int
    images_qty: int
    is_closed: bool = False


@dataclass
class ItemWithUser:
    item_id: int
    seller_id: int
    is_verified_seller: bool
    name: str
    description: str
    category: int
    images_qty: int


@dataclass
class ModerationResult:
    task_id: int
    item_id: int
    status: str
    is_violation: Optional[bool]
    probability: Optional[float]
    error_message: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]
