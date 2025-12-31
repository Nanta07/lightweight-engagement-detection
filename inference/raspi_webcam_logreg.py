#Using Logistic Regression or MLP to classify engagement level from webcam feed on Raspberry Pi
import os
import time
import cv2
import mediapipe as mp
import joblib
import csv
import tkinter as tk
from tkinter import messagebox
from tkcalendar import Calendar
from PIL import Image, ImageTk

# Konfigurasi Global
BASE_FOLDER = "/home/elvindo/Documents/pi/Day2"
csv_file_path = None
video_writer = None
stop_recording = False
MODEL_TYPE = "mlp"

# Inisialisasi MediaPipe & Model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5)

# Load model sesuai pilihan
if MODEL_TYPE == "logreg":
    model_path = "fix_logreg_engagement.pkl"
elif MODEL_TYPE == "mlp":
    model_path = "fix_mlp_engagement.pkl"
else:
    raise ValueError("MODEL_TYPE harus 'logreg' atau 'mlp'")

model = joblib.load(model_path)

# Utilitas
def format_timestamp(timestamp):
    """Konversi UNIX timestamp ke format HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.localtime(timestamp))

# Fungsi Memulai Sesi Baru
def start_session(selected_date):
    """Membuat folder sesi, CSV, dan memulai webcam."""
    global csv_file_path, video_writer, stop_recording
    stop_recording = False

    # Buat folder sesi berdasarkan tanggal
    session_folder = os.path.join(BASE_FOLDER, f"session_{selected_date}")
    os.makedirs(session_folder, exist_ok=True)

    # Buat folder engagement 0–3
    engagement_folder = os.path.join(session_folder, "engagement")
    for level in ["0", "1", "2", "3"]:
        os.makedirs(os.path.join(engagement_folder, level), exist_ok=True)

    # Siapkan CSV log
    csv_file_path = os.path.join(session_folder, "engagement_results.csv")
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp (Unix)", "Time (HH:MM:SS)", "Frame Name",
            "Engagement Level", "Confidence", "Response Time", "FPS"
        ])

    video_path = os.path.join(session_folder, "session_video.mp4")
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480)
    )

    capture_frames_from_webcam(session_folder)

# Webcam Capture
def capture_frames_from_webcam(session_folder):
    """Mengambil frame webcam + klasifikasi + simpan video."""
    global video_writer, stop_recording

    cap = cv2.VideoCapture(0)

    # Buat window Tkinter untuk preview
    camera_window = tk.Toplevel()
    camera_window.title("Camera Preview")
    camera_window.geometry("700x600")

    video_label = tk.Label(camera_window)
    video_label.pack()

    # Tombol Stop
    tk.Button(
        camera_window,
        text="Stop Recording",
        command=lambda: stop_camera(camera_window)
    ).pack(pady=10)

    # Loop menampilkan frame setiap 10ms
    def show_frame():
        nonlocal cap

        ret, frame = cap.read()
        if not ret or stop_recording:
            return

        start_time = time.time()
        timestamp = int(time.time() * 1000)
        formatted_time = format_timestamp(timestamp // 1000)
        frame_name = f"frame_{timestamp}.jpg"

        # Klasifikasi frame
        level, confidence = process_and_classify_frame(
            frame, frame_name, session_folder)

        # Hitung metrik waktu
        response_time = time.time() - start_time
        fps = 1 / response_time if response_time > 0 else 0

        # Simpan ke CSV
        with open(csv_file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, formatted_time, frame_name,
                level, confidence, response_time, fps
            ])

        # Simpan frame ke video
        video_writer.write(frame)

        # Tampilkan ke Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        video_label.configure(image=img)
        video_label.img_tk = img

        camera_window.after(10, show_frame)

    show_frame()
    camera_window.mainloop()

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    display_classification_report()

# Stop Camera
def stop_camera(window):
    """Stop kamera dan tampilkan laporan klasifikasi."""
    global stop_recording
    stop_recording = True
    window.destroy()
    display_classification_report()

# Proses Klasifikasi Frame
def process_and_classify_frame(frame, frame_name, session_folder):
    """Deteksi wajah, ambil landmark, prediksi, simpan frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    engagement_level = -1
    confidence = 0.0

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
            flattened = [c for p in landmarks for c in p]

            # 468 landmark × 2 = 936 fitur
            if len(flattened) == 936:
                # Prediksi menggunakan model yang dipilih
                if MODEL_TYPE == "logreg" or MODEL_TYPE == "mlp":
                    probabilities = model.predict_proba([flattened])
                    engagement_level = model.predict([flattened])[0]
                    confidence = max(probabilities[0])

                    # Simpan frame sesuai folder engagement
                    folder = os.path.join(
                        session_folder, "engagement", str(engagement_level)
                    )
                    cv2.imwrite(os.path.join(folder, frame_name), frame)

    return engagement_level, confidence

# Tampilkan Laporan Akhir
def display_classification_report():
    """Menghitung jumlah frame setiap level dan tampilkan hasil."""
    if not csv_file_path:
        return

    engagement_counts = {"0": 0, "1": 0, "2": 0, "3": 0}

    with open(csv_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            level = row["Engagement Level"]
            if level in engagement_counts:
                engagement_counts[level] += 1

    low = engagement_counts["0"] + engagement_counts["1"]
    high = engagement_counts["2"] + engagement_counts["3"]
    result = "Low Engagement" if low > high else "High Engagement"

    messagebox.showinfo(
        "Classification Report",
        f"""
Engagement 0 = {engagement_counts['0']}
Engagement 1 = {engagement_counts['1']}
Engagement 2 = {engagement_counts['2']}
Engagement 3 = {engagement_counts['3']}

Engagement Result = {result}
"""
    )


# Tkinter UI
def start_session_ui():
    date = calendar.get_date()
    if not date:
        messagebox.showwarning("Input Error", "Tanggal harus dipilih!")
        return
    start_session(date)


root = tk.Tk()
root.title("Start Webcam Session")
root.geometry("400x300")

tk.Label(root, text="Pilih Tanggal:").pack(pady=5)

calendar = Calendar(root, selectmode="day", date_pattern="yyyy-mm-dd")
calendar.pack(pady=10)

tk.Button(root, text="Start Session", command=start_session_ui).pack(pady=20)

root.mainloop()