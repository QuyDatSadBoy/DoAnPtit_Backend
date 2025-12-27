#!/usr/bin/env python3
"""
Test Socket.IO notification flow
Run: python scripts/test_socket_notification.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.timezone import now_vn

import redis
from app.core.config import settings


def test_notification(user_id: str, notification_type: str = "inference_failed"):
    """
    Send a test notification through Redis pub/sub
    
    Args:
        user_id: Target user ID
        notification_type: 'inference_complete' or 'inference_failed'
    """
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True
    )
    
    notification = {
        "type": notification_type,
        "inference_id": "test-inference-123",
        "medical_record_id": "test-record-456",
        "patient_id": "test-patient-789",
        "user_id": user_id,
        "status": "completed" if notification_type == "inference_complete" else "failed",
        "message": f"Test {notification_type} notification",
        "timestamp": now_vn().isoformat()
    }
    
    if notification_type == "inference_failed":
        notification["error"] = "This is a test error message"
    else:
        notification["ct_path"] = "patient_files/test/ct_output.nii.gz"
        notification["ct_gif_path"] = "patient_files/test/ct_output.gif"
    
    # Publish to Redis
    redis_client.publish("inference_notifications", json.dumps(notification))
    
    print(f"✅ Published test notification:")
    print(f"   Type: {notification_type}")
    print(f"   User: {user_id}")
    print(f"   Channel: inference_notifications")
    print(f"\n📋 Notification data:")
    print(json.dumps(notification, indent=2))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Socket.IO notifications")
    parser.add_argument("--user", "-u", required=True, help="Target user ID (UUID)")
    parser.add_argument("--type", "-t", choices=["complete", "failed"], default="failed",
                       help="Notification type: 'complete' or 'failed'")
    
    args = parser.parse_args()
    
    noti_type = "inference_complete" if args.type == "complete" else "inference_failed"
    test_notification(args.user, noti_type)
