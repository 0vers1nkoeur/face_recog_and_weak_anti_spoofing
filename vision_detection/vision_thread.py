import threading
import cv2
from collections import deque

from vision_detection.video_capture import init_camera, get_frame, release_camera
from vision_detection.face_detection import FaceMeshDetector
from vision_detection.face_alignment import align_and_crop, compute_eye_centers
from vision_detection.utils import FPSMeter, EyeSmoother

#We are creating class that will run all detection, alignement and cropping in the backgorund
#It needs to give us output that is buffer of frames
#Also it needs to work in backgorund so that we don't need to exit camera to get buffer out
class VisionThread(threading.Thread):   #By adding threading.Thread that means this class will work in backgorund thread

    def __init__(self, buffer_seconds=15, fps_estimate=30, camera_index=0, debug=False):
        super().__init__()  #We are calling constructor of parent class (threading)

        #We need to determin buffer size
        #We need to set which camer we are using
        self.buffer_size = buffer_seconds * fps_estimate
        self.camera_index = camera_index
        self.debug = debug

        #Main things other modules will read in real-time
        self.frame_buffer = deque(maxlen=self.buffer_size)   #Big video frame buffer (450 frames)
        self.last_aligned_face = None                        
        self.last_coords = None                               
        self.last_frame = None                                
        self.last_bbox = None                                 

        self.running = False    #Used to start and stop thread
        self.detector = FaceMeshDetector(max_num_faces=1)   #Represents detector that uses mediapipe to detect face
        self.fps_meter = FPSMeter()     #Counts frames in secons
        self.eye_smoother = EyeSmoother(window=5)   #Smothers the eyes


    def stop(self):
        #Stop thread from outside
        self.running = False


    def run(self):
        cap = init_camera(self.camera_index)    #Creating camera (with default index)
        self.running = True     #Turning on thread

        print("Camera thread started")

        while self.running:

            frame = get_frame(cap)      #Captureing frame from camera 
            if frame is None:
                continue

            #Storeing last frame
            self.last_frame = frame.copy()

            #Appending frame into big video buffer ---------------------------------------FOR LORENZO
            self.frame_buffer.append(frame.copy())

            #Just for aesthetics we will show on screen when buffer is ready
            if len(self.frame_buffer) == self.buffer_size:
                cv2.putText(frame, "BUFFER READY", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

            #We wil try to apply mesh (mediapipe mesh) on every possible frame and extract coords of a face
            coords = self.detector.process(frame)
            self.last_coords = coords

            #If we find some face we want to crop it out
            #We want to get eye centers for alignment
            if coords is not None:

                #Find bounding box
                min_x = min(p[0] for p in coords)
                min_y = min(p[1] for p in coords)
                max_x = max(p[0] for p in coords)
                max_y = max(p[1] for p in coords)
                self.last_bbox = (min_x, min_y, max_x, max_y)

                #Just to help the user see the captured face we are adding box around it
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y),
                              (0,255,0), 2)

                #Get eye centers
                left_xy, right_xy = compute_eye_centers(coords)

                #Smooth eyes to avoid shaking
                self.eye_smoother.update(left_xy, right_xy)

                #We need to cast coords into numbers so we can compare them 
                # lx, ly = map(int, left_xy)
                # rx, ry = map(int, right_xy)

                #Small yellow dots for eyes (again just for aesthetics)
                # cv2.circle(frame, (lx, ly), 5, (0,255,255), 1)
                # cv2.circle(frame, (rx, ry), 5, (0,255,255), 1)

                #Try to align and crop the face
                aligned = align_and_crop(frame, coords)
                if aligned is not None:
                    self.last_aligned_face = aligned

            #FPS counter (just for aesthetics)
            fps_value = int(self.fps_meter.tick())
            cv2.putText(frame, f"FPS: {fps_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

            #GUI show & keyboard handling (guarded for headless environments)
            if self.debug:
                try:
                    cv2.imshow("Vision & Detection", frame)
                    if cv2.waitKey(1) == 27:
                        self.stop()
                except cv2.error as exc:
                    # When running headless OpenCV may lack GUI support; disable debug windows gracefully.
                    print(f"Warning: OpenCV GUI unavailable ({exc}), disabling debug view.")
                    self.debug = False
                    cv2.destroyAllWindows()

        #Clean up
        print("Stopping camera GUI")
        release_camera(cap)
        cv2.destroyAllWindows()

        print("Camera thread stopped")
