from vision_detection.vision_thread import VisionThread
import time
import cv2


def main():

    vt = VisionThread()
    vt.start()

    print("VisionThread started.")

    while True:

        print("Buffer len:", len(vt.frame_buffer))
        time.sleep(1)

        #---------------------- LORENZO (ANTI-SPOOF) ----------------------
        raw_frames = list(vt.frame_buffer)


        #------------------------------------------------------------------


        #---------------------- KONSTANTINOS (VERIFICATION) ---------------
        aligned = vt.last_aligned_face
        

        #------------------------------------------------------------------

        # ESC stops EVERYTHING immediately
        if cv2.waitKey(1) == 27:
            vt.stop()       # stop Thread
            break           # EXIT MAIN LOOP IMMEDIATELY

        # If thread died for any reason
        if not vt.running:
            break

    vt.join()
    print("VisionThread stopped")

    cv2.destroyAllWindows()  #Close all windows immediately


if __name__ == "__main__":
    main()
