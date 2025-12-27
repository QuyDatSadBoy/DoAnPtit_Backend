"""
Celery Tasks for X-ray to CT Inference
"""
import os
import sys
import json
import uuid
from celery import shared_task
import redis

from app.worker.celery_app import celery_app
from app.core.config import settings, get_base_dir
from app.core.timezone import now_vn
from app.db.session import SessionLocal
from app.models.medical_record import MedicalRecord

# Base directory of the backend project
BASE_DIR = get_base_dir()

# Redis client for Socket.IO notifications
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True
)


def update_inference_status(medical_record_id: str, inference_id: str, status: str, ct_path: str = None, error: str = None):
    """Update inference status in database"""
    from sqlalchemy.orm.attributes import flag_modified
    
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
                        item["completed_at"] = now_vn().isoformat()
                    if status == "failed":
                        item["failed_at"] = now_vn().isoformat()
                new_history.append(item)
            
            record.infer_history = new_history
            # Force SQLAlchemy to detect JSONB changes
            flag_modified(record, "infer_history")
            db.commit()
            print(f"📝 Updated inference {inference_id} status to {status}")
    except Exception as e:
        print(f"⚠️ Failed to update inference status: {e}")
        db.rollback()
    finally:
        db.close()


@celery_app.task(bind=True, name="app.worker.tasks.process_inference")
def process_inference(
    self,
    inference_id: str,
    medical_record_id: str,
    patient_id: str,
    user_id: str,  # User who initiated the inference (for Socket.IO notification)
    xray_path: str,
    guidance_scale: float = 1.0
):
    """
    Process X-ray to CT inference
    This task runs on Celery worker with GPU access
    """
    print(f"🚀 Starting inference: {inference_id}")
    
    # Update status to processing
    try:
        update_inference_status(medical_record_id, inference_id, "processing")
    except Exception as e:
        print(f"⚠️ Failed to update processing status: {e}")
    
    # Setup paths
    backend_dir = str(BASE_DIR)
    xray2ct_dir = os.path.join(backend_dir, 'DoAnPtit_Xray2CT')
    original_cwd = os.getcwd()
    
    try:
        # Verify xray file exists
        if not os.path.exists(xray_path):
            raise FileNotFoundError(f"X-ray file not found: {xray_path}")
        
        # Add paths for imports
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        if xray2ct_dir not in sys.path:
            sys.path.insert(0, xray2ct_dir)
        
        # CRITICAL: Change working directory to DoAnPtit_Xray2CT
        # This fixes relative path issues in inference.py (e.g., ./pretrained_models/)
        os.chdir(xray2ct_dir)
        print(f"📂 Changed working directory to: {xray2ct_dir}")
        
        import torch
        import numpy as np
        from PIL import Image
        
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
        checkpoint_path = os.path.join(xray2ct_dir, 'checkpoints', 'model-81.pt')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
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
        
        # Build relative path for database storage
        path_parts = output_dir.split(os.sep)
        if 'patient_files' in path_parts:
            pf_idx = path_parts.index('patient_files')
            rel_patient_id = path_parts[pf_idx + 1]
            rel_record_id = path_parts[pf_idx + 2]
            ct_relative_path = f"patient_files/{rel_patient_id}/{rel_record_id}/{output_filename}_none_style.nii.gz"
            ct_gif_relative_path = f"patient_files/{rel_patient_id}/{rel_record_id}/{output_filename}_none_style.gif"
        else:
            ct_relative_path = os.path.join(output_dir, f"{output_filename}_none_style.nii.gz")
            ct_gif_relative_path = os.path.join(output_dir, f"{output_filename}_none_style.gif")
        
        # Update status to completed with RELATIVE path
        update_inference_status(medical_record_id, inference_id, "completed", ct_relative_path)
        
        # Send success notification
        try:
            notification = {
                "type": "inference_complete",
                "inference_id": inference_id,
                "medical_record_id": medical_record_id,
                "patient_id": patient_id,
                "user_id": user_id,
                "status": "completed",
                "ct_path": ct_relative_path,
                "ct_gif_path": ct_gif_relative_path,
                "message": "Tái tạo CT hoàn thành!",
                "timestamp": now_vn().isoformat()
            }
            redis_client.publish("inference_notifications", json.dumps(notification))
            print(f"📤 Sent success notification for: {inference_id}")
        except Exception as redis_err:
            print(f"⚠️ Failed to send notification: {redis_err}")
        
        print(f"✅ Inference completed: {inference_id}")
        
        return {
            "success": True,
            "inference_id": inference_id,
            "ct_path": ct_relative_path
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Inference failed: {inference_id} - {error_msg}")
        
        # Update status to failed
        try:
            update_inference_status(medical_record_id, inference_id, "failed", error=error_msg)
        except Exception as db_err:
            print(f"⚠️ Failed to update DB status: {db_err}")
        
        # Send failure notification
        try:
            notification = {
                "type": "inference_failed",
                "inference_id": inference_id,
                "medical_record_id": medical_record_id,
                "patient_id": patient_id,
                "user_id": user_id,
                "status": "failed",
                "error": error_msg,
                "message": f"Tái tạo CT thất bại: {error_msg}",
                "timestamp": now_vn().isoformat()
            }
            redis_client.publish("inference_notifications", json.dumps(notification))
            print(f"📤 Sent failure notification for: {inference_id}")
        except Exception as redis_err:
            print(f"⚠️ Failed to send notification: {redis_err}")
        
        return {
            "success": False,
            "inference_id": inference_id,
            "error": error_msg
        }
    
    finally:
        # ALWAYS restore original working directory
        os.chdir(original_cwd)
        print(f"📂 Restored working directory to: {original_cwd}")
