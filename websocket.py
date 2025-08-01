import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from datetime import datetime
import os
import traceback

class FrameProcessor:
    def __init__(self):
        self.frame_count = 0
        self.save_frames = True  # Set to True to save frames to disk
        self.output_dir = "captured_frames"
        
        # Create output directory if it doesn't exist
        if self.save_frames and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def process_frame(self, base64_data):
        """Process the received frame data"""
        try:
            # Validate input
            if not base64_data:
                print("No base64 data provided")
                return None
            
            # Decode base64 to image
            try:
                image_data = base64.b64decode(base64_data)
            except Exception as e:
                print(f"Base64 decode error: {e}")
                return None
            
            # Convert to numpy array
            np_array = np.frombuffer(image_data, np.uint8)
            
            if len(np_array) == 0:
                print("Empty numpy array from image data")
                return None
            
            # Decode image
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Failed to decode frame with OpenCV")
                return None
            
            # Process the frame (example: add timestamp)
            processed_frame = self.add_timestamp(frame.copy())
            
            # Save frame if enabled
            if self.save_frames:
                try:
                    self.save_frame(processed_frame)
                except Exception as e:
                    print(f"Error saving frame: {e}")
            
            # Extract frame information
            frame_info = {
                'width': frame.shape[1],
                'height': frame.shape[0],
                'channels': frame.shape[2],
                'frame_number': self.frame_count,
                'timestamp': datetime.now().isoformat()
            }
            
            self.frame_count += 1
            
            return frame_info
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
            return None
    
    def add_timestamp(self, frame):
        """Add timestamp to frame"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green
        thickness = 2
        
        # Add timestamp to top-left corner
        cv2.putText(frame, timestamp, (10, 30), font, font_scale, color, thickness)
        
        return frame
    
    def save_frame(self, frame):
        """Save frame to disk"""
        try:
            filename = f"frame_{self.frame_count:06d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            success = cv2.imwrite(filepath, frame)
            if success:
                print(f"Saved frame: {filename}")
            else:
                print(f"Failed to save frame: {filename}")
                
        except Exception as e:
            print(f"Error saving frame: {e}")

# Global frame processor instance
frame_processor = FrameProcessor()

async def handle_client(websocket):
    """Handle WebSocket client connections"""
    client_address = None
    
    try:
        client_address = websocket.remote_address
        print(f"New client connected: {client_address}")
        
        # Send welcome message
        welcome_msg = {
            'status': 'connected',
            'message': 'WebSocket connection established'
        }
        await websocket.send(json.dumps(welcome_msg))
        
        async for message in websocket:
            try:
                # Parse JSON message
                data = json.loads(message)
                
                if data.get('type') == 'frame':
                    # Process the frame
                    frame_data = data.get('data')
                    if not frame_data:
                        await websocket.send(json.dumps({
                            'status': 'error',
                            'message': 'No frame data received'
                        }))
                        continue
                    
                    frame_info = frame_processor.process_frame(frame_data)
                    
                    if frame_info:
                        # Send response back to client
                        response = {
                            'status': 'success',
                            'frame_info': frame_info,
                            'message': f"Frame {frame_info['frame_number']} processed"
                        }
                        
                        await websocket.send(json.dumps(response))
                        print(f"Processed frame {frame_info['frame_number']} - {frame_info['width']}x{frame_info['height']}")
                    else:
                        # Send error response
                        error_response = {
                            'status': 'error',
                            'message': 'Failed to process frame'
                        }
                        await websocket.send(json.dumps(error_response))
                
                elif data.get('type') == 'ping':
                    # Handle ping messages
                    await websocket.send(json.dumps({'type': 'pong', 'status': 'ok'}))
                
                else:
                    # Handle other message types
                    response = {
                        'status': 'error',
                        'message': f"Unknown message type: {data.get('type', 'unknown')}"
                    }
                    await websocket.send(json.dumps(response))
                    
            except json.JSONDecodeError as e:
                print(f"Invalid JSON received: {e}")
                try:
                    error_response = {
                        'status': 'error',
                        'message': 'Invalid JSON format'
                    }
                    await websocket.send(json.dumps(error_response))
                except:
                    pass
                
            except Exception as e:
                print(f"Error handling message: {e}")
                traceback.print_exc()
                try:
                    error_response = {
                        'status': 'error',
                        'message': f'Server error: {str(e)}'
                    }
                    await websocket.send(json.dumps(error_response))
                except:
                    pass
                
    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_address} disconnected normally")
    except websockets.exceptions.ConnectionClosedError:
        print(f"Client {client_address} connection closed with error")
    except Exception as e:
        print(f"Unexpected error with client {client_address}: {e}")
        traceback.print_exc()

async def main():
    """Main server function"""
    host = "localhost"
    port = 8765
    
    print(f"Starting WebSocket server on {host}:{port}")
    print("Waiting for connections...")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start the server
        async with websockets.serve(handle_client, host, port):
            print(f"Server started successfully on ws://{host}:{port}")
            # Keep the server running
            await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except OSError as e:
        print(f"Failed to start server: {e}")
        print("Make sure the port is not already in use")
    except Exception as e:
        print(f"Server startup error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Failed to start server: {e}")
        traceback.print_exc()