"""
AeroVision AI 审核服务

FastAPI 应用入口
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.core import settings, logger
from app.api import api_router
from app.core.exceptions import (
    AerovisionException,
    ImageLoadError,
    InferenceError,
    ModelNotLoadedError,
    ValidationError,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(f"启动 {settings.app_name} v{settings.app_version}")
    logger.info(f"运行环境: {'调试模式' if settings.debug else '生产模式'}")
    logger.info(f"设备: {settings.device}")

    # 预热模型（可选）
    if not settings.debug:
        try:
            from app.services import get_review_service
            service = get_review_service()
            service._lazy_load_infer()
            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    yield

    logger.info("服务关闭")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="航空摄影社区图片 AI 审核服务",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.exception(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务错误",
            "detail": str(exc) if settings.debug else None,
        },
    )


@app.exception_handler(AerovisionException)
async def aerovision_exception_handler(request: Request, exc: AerovisionException):
    """统一处理 Aerovision 异常"""
    logger.warning(f"Aerovision 异常: {exc.code} - {exc.message}")

    status_code_map = {
        "IMAGE_LOAD_ERROR": 400,
        "VALIDATION_ERROR": 422,
        "MODEL_NOT_LOADED": 503,
        "INFERENCE_ERROR": 500,
    }

    status_code = status_code_map.get(exc.code, 500)

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
            },
            "detail": str(exc) if settings.debug else None,
        },
    )


# 注册路由
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }


def main():
    """命令行启动入口"""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers if not settings.debug else 1,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
