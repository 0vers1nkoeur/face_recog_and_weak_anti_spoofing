import os
import shutil       #We use it to copy image form one to another

#This function extracts the newest aligned face from the preview folder
#Saves it into a dedicated folder for verification
def save_latest_aligned_face(
    preview_dir: str = "data/aligned_preview",
    save_dir: str = "data/verification"
):
    #Ensure output folder exists
    os.makedirs(save_dir, exist_ok=True)

    #List all images in the preview folder
    files = sorted(
        [f for f in os.listdir(preview_dir) if f.lower().endswith(".jpg")]
    )

    #If there are no images, we cannot export anything
    if not files:
        print("No preview images available to save.")
        return None

    #Take the last (newest) one
    latest_file = files[-1]

    src_path = os.path.join(preview_dir, latest_file)
    dst_path = os.path.join(save_dir, latest_file)

    #Copy the file to the verification directory
    shutil.copy(src_path, dst_path)

    print(f"Saved latest aligned face to: {dst_path}")      #Terminal message for user to know
    return dst_path
