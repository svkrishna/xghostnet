from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from ..core.security import SecurityManager, security_manager


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """
    Minimal token endpoint. Accepts any username/password in dev and returns
    a static token configured via env (GHOSTNET_API_TOKEN).
    """
    # In production, validate user credentials here
    token = security_manager.expected_token
    if not token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token not configured",
        )
    return {"access_token": token, "token_type": "bearer"}


@router.get("/whoami")
async def whoami(token: str = Depends(security_manager.oauth2_scheme)):
    security_manager.verify(token)
    return {"authenticated": True}


