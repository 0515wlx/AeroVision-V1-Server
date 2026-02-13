"""
历史记录 API 路由
提供历史审核记录推送接口
"""

from typing import List
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
import numpy as np

from app.core import settings, get_logger
from app.schemas.request import HistoricalRecordRequest
from app.schemas.response import SimilarRecord
from app.services import get_review_service, ReviewService

router = APIRouter()
logger = get_logger("api.history")


@router.post("/push")
async def push_historical_records(
    records: List[HistoricalRecordRequest],
    review_service: ReviewService = Depends(get_review_service)
) -> dict:
    """
    推送历史审核记录

    将历史审核记录推送到向量数据库，用于后续的相似度检索和新类别发现。

    **请求参数**:
    - `records`: 历史记录列表

    **返回**:
    - 推送结果统计
    """
    logger.info(f"收到历史记录推送请求，记录数: {len(records)}")

    try:
        # 检查推理模块是否支持向量数据库
        if not hasattr(review_service, '_enhanced_predictor') or review_service._enhanced_predictor is None:
            raise HTTPException(
                status_code=503,
                detail="向量数据库功能未启用"
            )

        # 确保增强预测器已加载
        if not hasattr(review_service, '_predictor') or review_service._predictor is None:
            try:
                from aerovision_inference import ModelPredictor
                import torch

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model_dir = settings.model_dir if hasattr(settings, 'model_dir') else 'models'

                config = {
                    'aircraft': {
                        'path': str(Path(model_dir) / 'aircraft.pt'),
                        'device': device,
                        'image_size': 640
                    },
                    'airline': {
                        'path': str(Path(model_dir) / 'airline.pt'),
                        'device': device,
                        'image_size': 640
                    }
                }

                review_service._predictor = ModelPredictor(config)
                logger.info("模型预测器初始化完成")
            except Exception as e:
                logger.error(f"初始化模型预测器失败: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"无法初始化特征提取模型: {str(e)}"
                )

        # 转换为 VectorRecord 并添加到数据库
        from aerovision_inference import VectorRecord

        added_count = 0
        failed_count = 0

        for record in records:
            try:
                # 从图片路径加载图片并提取特征向量
                image_path = record.image_path

                if not Path(image_path).exists():
                    logger.warning(f"图片路径不存在: {image_path}，跳过此记录")
                    failed_count += 1
                    continue

                # 使用 YOLO 的 embed 方法提取特征向量
                try:
                    aircraft_emb_result = review_service._predictor.aircraft_model.embed(
                        image_path,
                        imgsz=640,
                        device=review_service._predictor.device,
                        verbose=False
                    )

                    airline_emb_result = review_service._predictor.airline_model.embed(
                        image_path,
                        imgsz=640,
                        device=review_service._predictor.device,
                        verbose=False
                    )

                    # 转换为 numpy 数组并扁平化
                    aircraft_emb = aircraft_emb_result[0]
                    airline_emb = airline_emb_result[0]

                    if hasattr(aircraft_emb, 'cpu'):
                        aircraft_emb = aircraft_emb.cpu().numpy()
                    if hasattr(airline_emb, 'cpu'):
                        airline_emb = airline_emb.cpu().numpy()

                    aircraft_emb = np.array(aircraft_emb).flatten()
                    airline_emb = np.array(airline_emb).flatten()

                except Exception as embed_error:
                    logger.error(f"特征提取失败 {record.id}: {embed_error}")
                    failed_count += 1
                    continue

                vector_record = VectorRecord(
                    id=record.id,
                    image_path=record.image_path,
                    aircraft_embedding=aircraft_emb,
                    airline_embedding=airline_emb,
                    aircraft_type=record.aircraft_type,
                    airline=record.airline,
                    aircraft_confidence=record.aircraft_confidence,
                    airline_confidence=record.airline_confidence,
                    timestamp=record.timestamp.isoformat() if record.timestamp else None,
                    metadata=record.metadata
                )

                if review_service._enhanced_predictor.vector_db.add_record(vector_record):
                    added_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"添加记录 {record.id} 失败: {e}")
                failed_count += 1

        logger.info(f"历史记录推送完成: 成功={added_count}, 失败={failed_count}")

        return {
            "success": True,
            "total": len(records),
            "added": added_count,
            "failed": failed_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"历史记录推送失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"历史记录推送失败: {str(e)}"
        )


@router.get("/stats")
async def get_history_stats(
    review_service: ReviewService = Depends(get_review_service)
) -> dict:
    """
    获取历史记录统计信息

    **返回**:
    - 向量数据库统计信息
    """
    try:
        if not hasattr(review_service, '_enhanced_predictor') or review_service._enhanced_predictor is None:
            return {
                "available": False,
                "message": "向量数据库功能未启用"
            }

        stats = review_service._enhanced_predictor.get_db_statistics()
        return stats

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )
