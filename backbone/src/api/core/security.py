import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer


class SecurityManager:
    """
    Minimal bearer-token security manager providing an OAuth2 scheme for FastAPI
    and a simple token verification utility suitable for internal APIs.
    """

    def __init__(self, secret_key: str, expected_token: str | None = None):
        self.secret_key = secret_key
        # The tokenUrl must be absolute/relative to match OpenAPI flows
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
        # Single static token for simplicity; prefer setting via env in production
        self.expected_token = expected_token or os.environ.get("GHOSTNET_API_TOKEN", "dev-token")

    def verify(self, token: str) -> str:
        if not token or token != self.expected_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token


# Shared instance for dependencies
security_manager = SecurityManager(secret_key=os.environ.get("GHOSTNET_SECRET", "change-me"))


async def verify_token(token: str = Depends(security_manager.oauth2_scheme)) -> str:
    """FastAPI dependency to validate bearer token."""
    return security_manager.verify(token)


