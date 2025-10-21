from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np

app = FastAPI()

# allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected!")
    try:
        while True:
            data = await websocket.receive_text()
            # remove base64 prefix
            img_data = data.split(",")[1]
            nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Example: draw text
            cv2.putText(frame, "Frame Received!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            cv2.imshow("Backend Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print("WebSocket disconnected:", e)
    finally:
        cv2.destroyAllWindows()
