"""
Socket.IO Manager for Real-time Notifications
"""
import json
import asyncio
from typing import Dict, Set
import socketio
import redis.asyncio as redis

from app.core.config import settings
from app.core.security import decode_token

# Get CORS origins at import time
CORS_ORIGINS = settings.cors_origins

# Create Socket.IO server with CORS
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=CORS_ORIGINS,
    logger=True,
    engineio_logger=True
)

# Connected users: {user_id: set of sid}
connected_users: Dict[str, Set[str]] = {}


async def get_redis():
    """Get async Redis connection"""
    # Sử dụng redis_url property (có fallback) thay vì REDIS_URL (có thể None)
    return await redis.from_url(settings.redis_url)


@sio.event
async def connect(sid, environ, auth):
    """Handle client connection"""
    print(f"🔌 Client connecting: {sid}")
    
    # Authenticate user from token
    token = None
    if auth and 'token' in auth:
        token = auth['token']
    elif 'HTTP_AUTHORIZATION' in environ:
        auth_header = environ['HTTP_AUTHORIZATION']
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
    
    if not token:
        print(f"❌ No token provided for {sid}")
        return False
    
    try:
        payload = decode_token(token)
        user_id = payload.get('sub')
        
        # Store connection
        if user_id not in connected_users:
            connected_users[user_id] = set()
        connected_users[user_id].add(sid)
        
        # Store user_id in session
        await sio.save_session(sid, {'user_id': user_id})
        
        print(f"✅ User {user_id} connected with sid {sid}")
        
        # Join user-specific room
        await sio.enter_room(sid, f"user:{user_id}")
        
        # Send pending notifications
        redis_conn = await get_redis()
        pending = await redis_conn.lrange(f"notifications_list:{user_id}", 0, -1)
        for notification in pending:
            await sio.emit('notification', json.loads(notification), to=sid)
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed for {sid}: {e}")
        return False


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    session = await sio.get_session(sid)
    user_id = session.get('user_id') if session else None
    
    if user_id and user_id in connected_users:
        connected_users[user_id].discard(sid)
        if not connected_users[user_id]:
            del connected_users[user_id]
    
    print(f"🔌 Client disconnected: {sid}")


@sio.event
async def subscribe_inference(sid, data):
    """Subscribe to inference updates"""
    inference_id = data.get('inference_id')
    if inference_id:
        await sio.enter_room(sid, f"inference:{inference_id}")
        print(f"📡 {sid} subscribed to inference:{inference_id}")


@sio.event
async def unsubscribe_inference(sid, data):
    """Unsubscribe from inference updates"""
    inference_id = data.get('inference_id')
    if inference_id:
        await sio.leave_room(sid, f"inference:{inference_id}")


async def broadcast_inference_update(notification: dict):
    """Broadcast inference update to all connected clients"""
    await sio.emit('inference_update', notification)


async def send_to_user(user_id: str, event: str, data: dict):
    """Send event to specific user"""
    room = f"user:{user_id}"
    await sio.emit(event, data, room=room)


# Global task reference to prevent garbage collection
_redis_listener_task = None
_shutdown_event = asyncio.Event()


async def redis_listener():
    """Listen for Redis pub/sub messages and broadcast to Socket.IO"""
    redis_conn = None
    pubsub = None
    
    try:
        redis_conn = await get_redis()
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe("inference_notifications")
        
        print("📡 Redis listener started and waiting for messages...")
        
        while not _shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                    timeout=2.0
                )
                
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        
                        # Lấy user_id từ notification để gửi đúng người
                        user_id = data.get('user_id')
                        inference_id = data.get('inference_id')
                        notification_type = data.get('type', 'inference_update')
                        
                        print(f"📨 Received notification: type={notification_type}, user={user_id}, inference={inference_id}")
                        
                        # Map notification type to event name
                        event_map = {
                            'inference_complete': 'inference_completed',
                            'inference_completed': 'inference_completed',
                            'inference_failed': 'inference_failed',
                            'inference_status': 'inference_status',
                        }
                        event_name = event_map.get(notification_type, 'inference_update')
                        
                        if user_id:
                            room = f"user:{user_id}"
                            await sio.emit(event_name, data, room=room)
                            print(f"📢 Sent '{event_name}' to room {room}")
                        elif inference_id:
                            room = f"inference:{inference_id}"
                            await sio.emit(event_name, data, room=room)
                            print(f"📢 Sent '{event_name}' to room {room}")
                        else:
                            print(f"⚠️ No target in notification, skipping")
                            
                    except Exception as e:
                        print(f"Error processing message: {e}")
                        
            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                continue
                    
    except asyncio.CancelledError:
        print("📡 Redis listener cancelled")
        raise
    except Exception as e:
        print(f"❌ Redis listener error: {e}")
    finally:
        # Cleanup
        if pubsub:
            try:
                await pubsub.unsubscribe("inference_notifications")
                await pubsub.close()
            except:
                pass
        if redis_conn:
            try:
                await redis_conn.close()
            except:
                pass


def start_redis_listener():
    """Start Redis listener in background"""
    global _redis_listener_task, _shutdown_event
    
    # Reset shutdown event
    _shutdown_event.clear()
    
    # Cancel existing task if any
    if _redis_listener_task and not _redis_listener_task.done():
        _redis_listener_task.cancel()
    
    _redis_listener_task = asyncio.create_task(redis_listener())
    
    # Add callback to handle task completion
    def on_task_done(task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"❌ Redis listener task error: {e}")
    
    _redis_listener_task.add_done_callback(on_task_done)


async def stop_redis_listener():
    """Stop Redis listener gracefully"""
    global _redis_listener_task, _shutdown_event
    
    _shutdown_event.set()
    
    if _redis_listener_task and not _redis_listener_task.done():
        _redis_listener_task.cancel()
        try:
            await _redis_listener_task
        except asyncio.CancelledError:
            pass
    
    print("📡 Redis listener stopped")
