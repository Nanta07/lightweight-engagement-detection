print("=== SCRIPT test_extract.py STARTED ===")

from extract_landmarks import extract_landmark_vector

img_path = "C:\\Users\\Ananta\\Documents\\1. Collage\\PKL\\Student Employee\\DATASET_SEKUNDER\\Dataset\\Dataset_Sekunder\\Dataset_Daisee_Kurasi\\Sampled_Dataset_Test\\engagement_0\\9877360133_frame_0040.jpg"

vec = extract_landmark_vector(img_path)

if vec is None:
    print("❌ Landmark gagal diekstrak")
else:
    print("✅ Landmark berhasil:", vec.shape)

print("=== SCRIPT FINISHED ===")