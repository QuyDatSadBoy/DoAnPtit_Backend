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

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=settings.CORS_ORIGINS,
    logger=True,
    engineio_logger=True
)

# Connected users: {user_id: set of sid}
connected_users: Dict[str, Set[str]] = {}


async def get_redis():
    """Get async Redis connection"""
    return await redis.from_url(settings.REDIS_URL)


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
        sio.enter_room(sid, f"user:{user_id}")
        
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
    user_id = session.get('user_id')
    
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
        sio.enter_room(sid, f"inference:{inference_id}")
        print(f"📡 {sid} subscribed to inference:{inference_id}")


@sio.event
async def unsubscribe_inference(sid, data):
    """Unsubscribe from inference updates"""
    inference_id = data.get('inference_id')
    if inference_id:
        sio.leave_room(sid, f"inference:{inference_id}")


async def broadcast_inference_update(notification: dict):
    """Broadcast inference update to all connected clients"""
    await sio.emit('inference_update', notification)


async def send_to_user(user_id: str, event: str, data: dict):
    """Send event to specific user"""
    room = f"user:{user_id}"
    await sio.emit(event, data, room=room)


async def redis_listener():
    """Listen for Redis pub/sub messages and broadcast to Socket.IO"""
    redis_conn = await get_redis()
    pubsub = redis_conn.pubsub()
    await pubsub.subscribe("inference_notifications")
    
    print("📡 Redis listener started")
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                data = json.loads(message['data'])
                await sio.emit('inference_update', data)
                print(f"📢 Broadcast notification: {data.get('inference_id')}")
            except Exception as e:
                print(f"Error broadcasting: {e}")


def start_redis_listener():
    """Start Redis listener in background"""
    asyncio.create_task(redis_listener())
