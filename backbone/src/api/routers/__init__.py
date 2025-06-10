from fastapi import APIRouter
from .geolocation import router as geolocation_router
from .discovery import router as discovery_router

api_router = APIRouter()

api_router.include_router(geolocation_router)
api_router.include_router(discovery_router) 