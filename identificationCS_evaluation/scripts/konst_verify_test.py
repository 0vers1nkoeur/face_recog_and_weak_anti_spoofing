import os
import sys
import numpy as np
import cv2

# Make Python see the vision_detection package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from identificationCS_evaluation.verification import get_embedding_from_aligned_face, compare_embeddings

def main():
    # 1. Load reference embedding for user1 (Aleksa)
    user_id = "user1"
    ref_path = os.path.join("data", "embeddings", f"{user_id}.npy")
    if not os.path.exists(ref_path):
        print(f"❌ Reference embedding not found: {ref_path}")
        return

    emb_ref = np.load(ref_path)
    print(f"📂 Loaded reference embedding for {user_id} from {ref_path}")

    # New HOG-based embedding has shape (1764,). Bail out early if the stored
    # reference comes from the older pipeline.
    if emb_ref.shape[0] != 1764:
        print("⚠️ Stored embedding uses an old format. Recreate it with the current pipeline.")
        return

    # 2. Load OTHER person image
    img_path = os.path.join("data", "verification", "other_01.jpg")
    if not os.path.exists(img_path):
        print(f"❌ Other person image not found: {img_path}")
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"❌ Could not read image file: {img_path}")
        return

    # 3. Compute embedding for other person
    emb_live = get_embedding_from_aligned_face(img_bgr)
    if emb_live is None:
        print("❌ Could not compute embedding (no face detected?).")
        return

    # 4. Compare embeddings with a stricter threshold for the HOG embedding
    distance, is_match = compare_embeddings(emb_live, emb_ref, threshold=0.16)

    print("\n🔍 Verification result (OTHER PERSON TEST)")
    print("-----------------------------------------")
    print(f"Distance: {distance:.4f}")
    print(f"Match:    {is_match}")

    if is_match:
        print(f"✅ ACCEPT (this would be a FALSE ACCEPT if this is not {user_id})")
    else:
        print(f"❌ REJECT: face does NOT match {user_id}")

if __name__ == "__main__":
    main()
