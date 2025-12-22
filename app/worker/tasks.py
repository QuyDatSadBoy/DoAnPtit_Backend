"""
Celery Tasks for X-ray to CT Inference
"""
import os
import sys
import json
import uuid
from datetime import datetime
from celery import shared_task
import redis

from app.worker.celery_app import celery_app
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.medical_record import MedicalRecord

# Redis client for Socket.IO notifications
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True
)


def send_socket_notification(user_id: str, notification: dict):
    """Send notification via Redis pub/sub for Socket.IO"""
    channel = f"notifications:{user_id}"
    redis_client.publish(channel, json.dumps(notification))
    
    # Also store in a list for persistence
    redis_client.lpush(f"notifications_list:{user_id}", json.dumps(notification))
    redis_client.ltrim(f"notifications_list:{user_id}", 0, 99)  # Keep last 100


def update_inference_status(medical_record_id: str, inference_id: str, status: str, ct_path: str = None, error: str = None):
    """Update inference status in database"""
    db = SessionLocal()
    try:
        record = db.query(MedicalRecord).filter(
            MedicalRecord.id == uuid.UUID(medical_record_id)
        ).first()
        
        if record and record.infer_history:
            new_history = []
            for item in record.infer_history:
                if item.get("id") == inference_id:
                    item["status"] = status
                    if ct_path:
                        item["ct_path"] = ct_path
                    if error:
                        item["error"] = error
                    if status == "completed":
                        item["completed_at"] = datetime.utcnow().isoformat()
                new_history.append(item)
            
            record.infer_history = new_history
            db.commit()
    finally:
        db.close()


@celery_app.task(bind=True, name="app.worker.tasks.process_inference")
def process_inference(
    self,
    inference_id: str,
    medical_record_id: str,
    patient_id: str,
    xray_path: str,
    guidance_scale: float = 1.0
):
    """
    Process X-ray to CT inference
    This task runs on Celery worker with GPU access
    """
    print(f"🚀 Starting inference: {inference_id}")
    
    # Update status to processing
    update_inference_status(medical_record_id, inference_id, "processing")
    
    try:
        # Import inference modules
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'DoAnPtit_Xray2CT'))
        
        import torch
        import numpy as np
        from PIL import Image
        import cv2
        
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Load inference module
        from DoAnPtit_Xray2CT.inference import XrayToCTPAInference
        
        # Model config
        model_config = {
            'denoising_fn': 'Unet3D',
            'diffusion_img_size': 32,
            'diffusion_depth_size': 128,
            'diffusion_num_channels': 4,
            'dim_mults': [1, 2, 4, 8],
            'cond_dim': 512,
            'timesteps': 1000,
            'loss_type': 'l1_lpips',
            'l1_weight': 1.0,
            'perceptual_weight': 0.01,
            'discriminator_weight': 0.0,
            'classification_weight': 0.0,
            'classifier_free_guidance': False,
            'medclip': True,
            'name_dataset': 'LIDC',
            'dataset_min_value': -12.911299,
            'dataset_max_value': 9.596558,
            'vae_ckpt': 'stabilityai/sd-vae-ft-mse-original',
            'vqgan_ckpt': None
        }
        
        # Load model
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'DoAnPtit_Xray2CT', 'checkpoints', 'model-81.pt'
        )
        
        inferencer = XrayToCTPAInference(
            model_checkpoint=checkpoint_path,
            model_config=model_config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Preprocess X-ray
        if xray_path.endswith('.npy'):
            xray = np.load(xray_path)
        else:
            # Load image and preprocess
            img = Image.open(xray_path).convert('L')
            img = img.resize((224, 224))
            xray = np.array(img).astype(np.float32) / 255.0
            # Convert to 3 channels
            xray = np.stack([xray, xray, xray], axis=0)
        
        # Save preprocessed input
        xray_npy_path = xray_path.rsplit('.', 1)[0] + '.npy'
        np.save(xray_npy_path, xray.astype(np.float32))
        
        # Get output directory
        output_dir = os.path.dirname(xray_path)
        output_filename = f"ct_{inference_id}"
        
        # Run inference
        ctpa_volume, _ = inferencer.inference_pipeline(
            xray_path=xray_npy_path,
            output_dir=output_dir,
            filename=output_filename,
            guidance_scale=guidance_scale,
            show_viewer=False
        )
        
        # CT output path
        ct_path = os.path.join(output_dir, f"{output_filename}_none_style.nii.gz")
        ct_gif_path = os.path.join(output_dir, f"{output_filename}_none_style.gif")
        
        # Update status to completed
        update_inference_status(medical_record_id, inference_id, "completed", ct_path)
        
        # Send notification
        notification = {
            "type": "inference_complete",
            "inference_id": inference_id,
            "medical_record_id": medical_record_id,
            "patient_id": patient_id,
            "status": "completed",
            "ct_path": ct_path,
            "ct_gif_path": ct_gif_path,
            "message": "X-ray to CT conversion completed successfully!",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to all connected clients
        redis_client.publish("inference_notifications", json.dumps(notification))
        
        print(f"✅ Inference completed: {inference_id}")
        
        return {
            "success": True,
            "inference_id": inference_id,
            "ct_path": ct_path
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Inference failed: {inference_id} - {error_msg}")
        
        # Update status to failed
        update_inference_status(medical_record_id, inference_id, "failed", error=error_msg)
        
        # Send failure notification
        notification = {
            "type": "inference_failed",
            "inference_id": inference_id,
            "medical_record_id": medical_record_id,
            "patient_id": patient_id,
            "status": "failed",
            "error": error_msg,
            "message": f"X-ray to CT conversion failed: {error_msg}",
            "timestamp": datetime.utcnow().isoformat()
        }
        redis_client.publish("inference_notifications", json.dumps(notification))
        
        raise


@celery_app.task(name="app.worker.tasks.send_notification")
def send_notification(user_id: str, notification: dict):
    """Send notification to user via Socket.IO"""
    send_socket_notification(user_id, notification)
    return {"success": True}
