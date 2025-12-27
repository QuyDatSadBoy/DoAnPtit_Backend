"""
Timezone Utilities - Vietnam Timezone (UTC+7)
"""
from datetime import datetime, timezone, timedelta

# Vietnam timezone (UTC+7)
VN_TIMEZONE = timezone(timedelta(hours=7))


def now_vn() -> datetime:
    """Get current datetime in Vietnam timezone (UTC+7)"""
    return datetime.now(VN_TIMEZONE)


def utc_to_vn(dt: datetime) -> datetime:
    """Convert UTC datetime to Vietnam timezone"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(VN_TIMEZONE)


def vn_to_utc(dt: datetime) -> datetime:
    """Convert Vietnam timezone to UTC"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume Vietnam timezone if no timezone
        dt = dt.replace(tzinfo=VN_TIMEZONE)
    return dt.astimezone(timezone.utc)


def format_vn_datetime(dt: datetime, format_str: str = "%H:%M:%S %d/%m/%Y") -> str:
    """Format datetime in Vietnam format"""
    if dt is None:
        return ""
    vn_dt = utc_to_vn(dt) if dt.tzinfo != VN_TIMEZONE else dt
    return vn_dt.strftime(format_str)


# Alias for compatibility
def get_vietnam_now() -> datetime:
    """Alias for now_vn()"""
    return now_vn()
