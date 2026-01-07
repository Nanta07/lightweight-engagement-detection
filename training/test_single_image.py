#test_single_image.py

from extract_landmarks import extract_landmark_vector

img_path = "C:\\Users\\Ananta\\Documents\\1. Collage\\PKL\\Student Employee\\DATASET_SEKUNDER\\Dataset\\Dataset_Sekunder\\Dataset_Daisee_Kurasi\\Sampled_Dataset_Test\\\engagement_0\9877360133_frame_0038.jpg"

vec = extract_landmark_vector(img_path)
print(vec.shape if vec is not None else "Landmark gagal diekstrak")
