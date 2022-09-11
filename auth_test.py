from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import asyncio

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.security import OAuth2, OAuth2PasswordRequestForm
from pydantic import BaseModel
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
        "role": "admin",
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": False,
        "role": "user",
    },
}

app = FastAPI()


def fake_hash_password(password: str):
    return "fakehashed" + password


class OAuth2Custom(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: Optional[str] = None,
        scopes: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password={"tokenUrl": tokenUrl, "scopes": scopes})
        super().__init__(
            flows=flows,
            scheme_name=scheme_name,
            description=description,
            auto_error=auto_error,
        )

    async def __call__(self, request: Request) -> Optional[str]:
        try:
            user = await self.bearer_auth(request)
            return user
        except HTTPException:
            pass

        try:
            user = await self.other_auth(request)
            return user
        except HTTPException:
            pass

        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async def bearer_auth(self, request: Request):
        authorization_header: str = request.headers.get("Authorization")
        if authorization_header is None:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        scheme, _, token = authorization_header.partition(" ")

        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Auth token
        user = fake_decode_token(token)

        return user

    async def other_auth(self, request: Request):
        authorization_header: str = request.headers.get("Session Cookie thing")
        if authorization_header is None:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Session"},
            )

        # Auth token / cookie thing
        user = "alice"

        if user == "service_account":
            user = request.headers.get("user-name")

        return user


oauth2_scheme = OAuth2Custom(tokenUrl="token")


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None
    role: Union[str, None] = None


class UserInDB(User):
    hashed_password: str


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def fake_decode_token(token):
    # This doesn't provide any security at all
    # Check the next version
    user = get_user(fake_users_db, token)
    return user


@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
):  # this logic will get token from keycloak
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}


@app.get("/users/auth/me")
async def read_users_me_auth(current_user: User = Depends(oauth2_scheme)):
    return current_user


# ----------------------------------------------------------


class BasePermission(ABC):
    """
    Abstract permission that all other Permissions must be inherited from.
    Defines basic error message, status & error codes.
    """

    error_msg = [{"msg": "You don't have permission to access or modify this resource."}]
    status_code = status.HTTP_403_FORBIDDEN
    role = None

    @abstractmethod
    def has_required_permissions(self, request: Request) -> bool:
        ...

    def __init__(
        self,
        request: Request,
    ):

        self.role = "user"

        if not self.has_required_permissions(request):
            raise HTTPException(status_code=self.status_code, detail=self.error_msg)


class AdminPermission(BasePermission):
    def has_required_permissions(self, request: Request) -> bool:
        return self.role == "admin"


class UserPermission(BasePermission):
    def has_required_permissions(self, request: Request) -> bool:
        try:
            AdminPermission(request)
            return True
        except HTTPException:
            return self.role == "user"


@app.get(
    "/user/endpoint",
    dependencies=[Depends(UserPermission)],
)
async def user_endpoint():
    return "😊"


@app.get(
    "/admin",
    dependencies=[Depends(AdminPermission)],
)
async def admin():
    return "😊"
