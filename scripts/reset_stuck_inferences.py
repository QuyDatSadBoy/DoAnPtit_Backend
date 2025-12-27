#!/usr/bin/env python3
"""
Script để reset các inference bị treo ở trạng thái 'processing' hoặc 'pending'
Chạy: python scripts/reset_stuck_inferences.py
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from app.db.session import SessionLocal
from app.models.medical_record import MedicalRecord
from app.core.timezone import now_vn


def reset_stuck_inferences(older_than_minutes: int = 10):
    """
    Reset các inference bị treo (processing/pending) lâu hơn X phút
    
    Args:
        older_than_minutes: Inference cũ hơn X phút sẽ được reset thành 'failed'
    """
    db = SessionLocal()
    
    try:
        # Lấy tất cả medical records có infer_history
        records = db.query(MedicalRecord).filter(
            MedicalRecord.infer_history.isnot(None)
        ).all()
        
        cutoff_time = now_vn() - timedelta(minutes=older_than_minutes)
        updated_count = 0
        
        for record in records:
            if not record.infer_history:
                continue
            
            new_history = []
            record_updated = False
            
            for item in record.infer_history:
                status = item.get("status", "")
                created_at_str = item.get("created_at", "")
                
                # Chỉ xử lý các item đang processing hoặc pending
                if status in ["processing", "pending"]:
                    try:
                        # Parse created_at
                        if created_at_str:
                            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00").replace("+00:00", ""))
                        else:
                            created_at = datetime.min
                        
                        # Nếu cũ hơn cutoff, đánh dấu failed
                        if created_at < cutoff_time:
                            item["status"] = "failed"
                            item["error"] = "Quá trình tái tạo CT bị gián đoạn. Vui lòng thử lại."
                            item["failed_at"] = now_vn().isoformat()
                            record_updated = True
                            print(f"  ⚠️ Reset inference {item.get('id')} (status: {status}, created: {created_at_str})")
                    except Exception as e:
                        print(f"  ❌ Error parsing date for inference {item.get('id')}: {e}")
                
                new_history.append(item)
            
            if record_updated:
                # Force SQLAlchemy to detect JSONB change
                from sqlalchemy.orm.attributes import flag_modified
                record.infer_history = new_history
                flag_modified(record, "infer_history")
                updated_count += 1
                print(f"✅ Updated record: {record.id}")
        
        if updated_count > 0:
            db.commit()
            print(f"\n🎉 Đã reset {updated_count} record(s) với inference bị treo")
        else:
            print(f"\n✨ Không có inference nào bị treo (cũ hơn {older_than_minutes} phút)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()


def list_stuck_inferences():
    """Liệt kê tất cả inference đang processing/pending"""
    db = SessionLocal()
    
    try:
        records = db.query(MedicalRecord).filter(
            MedicalRecord.infer_history.isnot(None)
        ).all()
        
        stuck = []
        for record in records:
            if not record.infer_history:
                continue
            
            for item in record.infer_history:
                status = item.get("status", "")
                if status in ["processing", "pending"]:
                    stuck.append({
                        "record_id": str(record.id),
                        "inference_id": item.get("id"),
                        "status": status,
                        "created_at": item.get("created_at"),
                        "xray_path": item.get("xray_path")
                    })
        
        if stuck:
            print(f"\n📋 Có {len(stuck)} inference đang chờ/xử lý:\n")
            for s in stuck:
                print(f"  Record: {s['record_id']}")
                print(f"  Inference: {s['inference_id']}")
                print(f"  Status: {s['status']}")
                print(f"  Created: {s['created_at']}")
                print(f"  X-ray: {s['xray_path']}")
                print("-" * 50)
        else:
            print("\n✨ Không có inference nào đang chờ/xử lý")
        
        return stuck
        
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reset stuck inferences")
    parser.add_argument("--list", "-l", action="store_true", help="Liệt kê inference bị treo")
    parser.add_argument("--reset", "-r", action="store_true", help="Reset inference bị treo")
    parser.add_argument("--minutes", "-m", type=int, default=10, help="Inference cũ hơn X phút (default: 10)")
    
    args = parser.parse_args()
    
    if args.list:
        list_stuck_inferences()
    elif args.reset:
        print(f"🔄 Đang reset inference bị treo (cũ hơn {args.minutes} phút)...\n")
        reset_stuck_inferences(args.minutes)
    else:
        # Mặc định: list trước
        stuck = list_stuck_inferences()
        if stuck:
            print("\n💡 Để reset, chạy: python scripts/reset_stuck_inferences.py --reset")
