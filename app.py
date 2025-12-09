# import streamlit as st
# import torch
# import numpy as np
# from PIL import Image
# import io
# from ultralytics import YOLO # Import YOLO from ultralytics

# # import gspread
# # from google.oauth2.service_account import Credentials
# # import datetime

# # scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# # creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
# # client = gspread.authorize(creds)

# # Set the title of the app
# st.title("Redbanded Stink Bug Detection")

# # Add user guidance
# st.markdown("Goal: Get the total number of redbanded stink bugs in the uploaded image")

# # st.markdown("##### How to Use:")
# # st.markdown("1. Upload a JPG image file using the file uploader below.")
# # st.markdown("2. The app processes the image and displays the count of detected redbanded stinkbugs.")

# # Implement file upload
# uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

# # Load the .pt model
# # model_path = './saved_models_of_1008_images/yolov8s_best.pt' # Use the correct path provided in the context.
# model_path = './saved_models/saved_models_thousand_eight_images/yolov8m_cbam_asff_finetuned.pt'
# model = None # Initialize model to None


# try:
#     # Load the model using ultralytics
#     model = YOLO(model_path)
#     st.success("Model loaded successfully!") # Optional: uncomment to show success message
#     # No need to call model.eval() explicitly for YOLO models loaded this way
# except FileNotFoundError:
#     st.error(f"Error: Model file not found at {model_path}")
# except Exception as e:
#     st.error(f"An error occurred while loading the model: {e}")

# # Process the uploaded image
# def process_image(uploaded_file, model):
#     """
#     Processes an uploaded image using a YOLOv8 model to detect objects
#     and draws bounding boxes or masks on the image.

#     Args:
#         uploaded_file: The file object uploaded by the user.
#         model: The loaded YOLOv8 model.

#     Returns:
#         A tuple containing:
#         - A Pillow Image object with detected objects outlined, or None if an error occurs.
#         - The number of detected objects, or 0 if none or an error occurs.
#     """
#     if uploaded_file is not None and model is not None:
#         try:
#             # Read the image data
#             image_data = uploaded_file.read()
#             # Open the image using Pillow
#             img = Image.open(io.BytesIO(image_data))

#             # Convert Pillow image to NumPy array (HWC format)
#             img_np = np.array(img)

#             # Perform inference
#             # Assuming the model is a YOLOv8 model and its predict method handles NumPy arrays
#             # or can be adapted. If the model expects a different format (e.g., PyTorch tensor),
#             # conversion will be needed here. YOLOv8 models from Ultralytics usually work with
#             # PIL Images or NumPy arrays directly.
#             results = model(img_np, conf =0.7)

#             # Get the number of detected objects
#             # Assuming results[0] contains the detection results
#             num_detections = len(results[0].boxes) if hasattr(results[0], 'boxes') else 0
#             # You might need to adjust this based on whether your model outputs masks or boxes
#             # For segmentation models, you might check len(results[0].masks)

#             # Process results and draw on the imafge
#             # The exact method for drawing depends on the results format.
#             # Assuming results is an object with a .plot() method (common in Ultralytics YOLO)
#             # which returns an image with detections drawn.
#             results_image_np = results[0].plot(labels=False)

#             # Convert the result NumPy array back to a Pillow Image
#             results_image_pil = Image.fromarray(results_image_np)

#             return results_image_pil, num_detections

#         except Exception as e:
#             st.error(f"Error processing image: {e}")
#             return None, 0
#     return None, 0

# # Display the output
# if uploaded_file is not None and model is not None:
#     # Process the uploaded image
#     results_image_pil, num_detections = process_image(uploaded_file, model)

#     # Display the results
#     if results_image_pil is not None:
#         st.image(results_image_pil, caption="Redbanded Stinkbug Detection", use_container_width=True)
#         st.subheader(f"Total number of redbanded stink bugs detected: {num_detections}")

# # Add instructions on how to run locally (optional for cloud deployment but good for development)
# st.markdown("---")


# # # Open your sheet
# # sheet = client.open("Stinkbug-Counting-Logs").sheet1

# # # Example data to log
# # timestamp = str(datetime.datetime.now())
# # user = st.text_input("Enter your name:")
# # detection_count = num_detections  # or whatever logic you're using

# # # Button to submit
# # if st.button("Log Detection"):
# #     sheet.append_row([timestamp, user, detection_count])
# #     st.success("Logged successfully!")strs 


# from here

import io
import os
import sqlite3
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
import smtplib
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

DB_PATH = BASE_DIR / "stinkbug.db"
MODEL_PATH = BASE_DIR / "yolov8m_cbam_asff_finetuned.pt"
THRESHOLD_PER_100 = 16
MIN_IMAGES_FOR_ALERT = 100
MAX_GALLERY_ITEMS = 20


# # Accessing secrets
# host = st.secrets["SMTP_HOST"]
# port = st.secrets["SMTP_PORT"]
# user = st.secrets["SMTP_USER"]
# password = st.secrets["SMTP_PASSWORD"]

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                report_date TEXT NOT NULL,
                location TEXT NOT NULL,
                image_name TEXT NOT NULL,
                insect_count INTEGER NOT NULL,
                priority TEXT DEFAULT 'Normal',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS gallery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                location TEXT NOT NULL,
                image_bytes BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    import hashlib

    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username: str, email: str, password: str) -> tuple[bool, str]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, hash_password(password)),
            )
            conn.commit()
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please pick a different one."


def authenticate_user(username: str, password: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
    if not row:
        return None
    user_id, name, email, stored_hash = row
    if hash_password(password) == stored_hash:
        return {"id": user_id, "username": name, "email": email}
    return None


def insert_reports(user, location: str, detection_rows: list[dict]) -> None:
    report_date = datetime.utcnow().strftime("%m::%d::%y")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO reports (user_id, username, report_date, location, image_name, insect_count, priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    user["id"],
                    user["username"],
                    report_date,
                    location,
                    row["image_name"],
                    row["count"],
                    row.get("priority", "Normal"),
                )
                for row in detection_rows
            ],
        )
        conn.commit()


def load_reports() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT username, report_date, location, image_name, insect_count, priority FROM reports ORDER BY id DESC",
            conn,
        )
    return df


def location_stats(location: str) -> tuple[int, int]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*), COALESCE(SUM(insect_count), 0) FROM reports WHERE location = ?",
            (location,),
        )
        images, insects = cursor.fetchone()
    return images or 0, insects or 0


def set_location_priority(location: str, priority: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE reports SET priority = ? WHERE location = ?",
            (priority, location),
        )
        conn.commit()


def send_email_alert(to_email: str, location: str, ratio: float, total_images: int, total_insects: int) -> bool:
    smtp_host = os.getenv("SMTP_HOST")
    print("smtp host", smtp_host)
    smtp_port = os.getenv("SMTP_PORT")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    # Accessing secrets
    # smtp_host = st.secrets["SMTP_HOST"]
    # smtp_port = st.secrets["SMTP_PORT"]
    # smtp_user = st.secrets["SMTP_USER"]
    # smtp_password = st.secrets["SMTP_PASSWORD"]

    if not all([smtp_host, smtp_port, smtp_user, smtp_password]):
        st.info(
            "Email alert skipped. Please set SMTP_HOST, SMTP_PORT, SMTP_USER, and SMTP_PASSWORD environment variables."
        )
        return False

    msg = EmailMessage()
    msg["Subject"] = f"Stink Bug Alert for {location}"
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content(
        f"""Hello,

Insect monitoring for {location} exceeded the safety threshold.

- Images analyzed: {total_images}
- Insects counted: {total_insects}
- Density: {int(ratio)} insects per 100 images (approximate calculation)

Please initiate precautionary applications immediately.

Thanks,
LSU Agcenter and IGLab 
"""
    )
    try:
        with smtplib.SMTP_SSL(smtp_host, int(smtp_port)) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True
    except Exception as exc:
        st.warning(f"Email alert could not be sent: {exc}")
        return False


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        st.error("Model file yolov8m_cbam_asff_finetuned.pt is missing.")
        st.stop()
    return YOLO(str(MODEL_PATH))


def count_insects(image_bytes: bytes) -> tuple[int, bytes]:
    model = load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image, verbose=False)
    if not results:
        return 0, image_bytes
    result = results[0]
    boxes = result.boxes
    count = int(boxes.shape[0]) if boxes is not None else 0
    annotated_array = result.plot()  # BGR numpy array
    annotated_image = Image.fromarray(annotated_array[..., ::-1])
    buffer = io.BytesIO()
    annotated_image.save(buffer, format="PNG")
    return count, buffer.getvalue()


def ensure_session_defaults():
    defaults = {
        "auth_user": None,
        "menu_choice": "Homepage",
        "latest_detections": [],
        "notifications": [],
        "processed_upload_signature": None,  # legacy, kept for compatibility
        "detection_cache": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def logout():
    st.session_state["auth_user"] = None
    st.session_state["menu_choice"] = "Homepage"
    st.session_state["latest_detections"] = []
    st.rerun()


def threshold_check(location: str, user_email: str) -> None:
    total_images, total_insects = location_stats(location)
    if total_images == 0:
        return
    ratio = (total_insects / total_images) * 100
    # User rule:
    # 1) When images are 100 and count is at least 16  -> 16 insects per 100 images
    # 2) When count is already 16 even if images < 100 -> early warning
    meets_threshold = (total_images >= 100 and ratio >= THRESHOLD_PER_100) or (
        total_images < 100 and total_insects >= THRESHOLD_PER_100
    )

    # Always log a notification so the user sees that the location was updated
    base_message = (
        f"Location '{location}' updated: {total_insects} insects over {total_images} images "
        f"({int(ratio)} insects / 100 images)."
    )

    if meets_threshold:
        set_location_priority(location, "Priority")
        message = "Priority alert: " + base_message
        st.toast(message, icon="⚠️")
        send_email_alert(user_email, location, ratio, total_images, total_insects)
    else:
        set_location_priority(location, "Normal")
        message = "Below threshold: " + base_message

    st.session_state["notifications"].insert(
        0,
        {
            "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
            "message": message,
            "location": location,
            "ratio": ratio,
        },
    )


def update_gallery(location: str, detections: list[dict]) -> None:
    timestamp = datetime.utcnow().strftime("%m/%d %H:%M")

    # Update in-memory gallery for the current session
    session_entries = [
        {
            "image_name": det["image_name"],
            "location": location,
            "annotated": det["annotated"],
            "timestamp": timestamp,
        }
        for det in detections
    ]
    existing = st.session_state.get("gallery", [])
    st.session_state["gallery"] = (session_entries + existing)[:MAX_GALLERY_ITEMS]

    # Persist annotated images so they are available after logout/login
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO gallery (image_name, location, image_bytes, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    det["image_name"],
                    location,
                    det["annotated"],
                    datetime.utcnow().isoformat(timespec="seconds"),
                )
                for det in detections
            ],
        )
        conn.commit()


def load_gallery(limit: int = MAX_GALLERY_ITEMS) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT image_name, location, image_bytes, created_at
            FROM gallery
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    entries: list[dict] = []
    for image_name, location, image_bytes, created_at in rows:
        entries.append(
            {
                "image_name": image_name,
                "location": location,
                "annotated": image_bytes,
                "timestamp": datetime.fromisoformat(created_at).strftime("%m/%d %H:%M"),
            }
        )
    return entries


def render_slideshow(title: str, entries: list[dict], slider_key: str):
    if not entries:
        st.caption("No images to display yet.")
        return
    st.subheader(title)
    counter = st_autorefresh(interval=10_000, key=f"{slider_key}_refresh")
    index = counter % len(entries)
    entry = entries[index]
    st.image(
        entry["annotated"],
        caption=f"{entry['image_name']} — {entry.get('location', 'Unknown location')}",
        use_container_width=True,
    )
    st.caption(f"Captured at: {entry.get('timestamp', '—')}")


def render_home():
    st.header("Welcome to Redbanded Stink bug (RBSB) Adults")
    st.write(
        "Upload RBSB images, count RBSBs  by location, and receive automated alerts when intervention thresholds are exceeded, recommending insecticide applications."
    )
    render_slideshow("Recent detections", load_gallery(), slider_key="home_gallery")
    notifications = st.session_state.get("notifications", [])
    if notifications:
        st.subheader("Recent alerts")
        for note in notifications[:5]:
            st.info(f"[{note['timestamp']}] {note['message']}")
    else:
        st.caption("No alerts yet. Alerts will appear here when thresholds are exceeded.")
    st.markdown(
        """
        **Workflow**
        1. Use the **Capture Image** panel to upload images from a field/location.
        2. Provide the location details and save the report.
        3. Review and export historic counts from **Reports**.
        """
    )


def render_capture():
    st.subheader("Capture & Analyze RBSB Images")
    uploaded_files = st.file_uploader(
        "Upload RBSB images (multiple files allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        cache = st.session_state.get("detection_cache", {})
        new_files = [f for f in uploaded_files if f.name not in cache]

        if new_files:
            with st.spinner("Analyzing RBSB images. This may take a while depending upon the quality and number of images."):
                for file in new_files:
                    bytes_data = file.getvalue()
                    insect_count, annotated = count_insects(bytes_data)
                    cache[file.name] = {
                        "image_name": file.name,
                        "count": insect_count,
                        "preview": bytes_data,
                        "annotated": annotated,
                    }
            st.session_state["detection_cache"] = cache
            st.success("Detection complete. Review results below.")

        # Build detections list from cache for all currently uploaded files
        detections = [cache[f.name] for f in uploaded_files if f.name in cache]
        st.session_state["latest_detections"] = detections

    detections = st.session_state.get("latest_detections", [])
    if detections:
        render_slideshow("Captured batch", detections, slider_key="capture_gallery")

        for det in detections:
            with st.expander(f"{det['image_name']} — {det['count']} insects"):
                st.image(det["annotated"], caption=f"Detected {det['count']} insects")
                st.image(det["preview"], caption="Original image")

        with st.form("location_form"):
            location = st.text_input("Location (city, block, GPS, etc.)", max_chars=120)
            submitted = st.form_submit_button("Save Report")

        if submitted:
            if not location.strip():
                st.error("Location is required to save the report.")
                return
            user = st.session_state["auth_user"]
            insert_reports(user, location.strip(), detections)
            st.session_state["latest_detections"] = []
            # keep cache so images can still be shown in slideshows; clear only the current batch
            threshold_check(location.strip(), user["email"])
            update_gallery(location.strip(), detections)
            st.success("Report saved and dashboard updated.")


def render_reports():
    st.subheader("Reports")
    df = load_reports()
    if df.empty:
        st.info("No reports yet. Capture and submit images to populate this table.")
        return

    render_slideshow("Recent detections", load_gallery(), slider_key="reports_gallery")

    summary = (
        df.groupby("location")
        .agg(
            total_images=("image_name", "count"),
            total_insects=("insect_count", "sum"),
            latest_date=("report_date", "max"),
            priority=("priority", lambda x: "Priority" if "Priority" in set(x) else "Normal"),
        )
        .reset_index()
    )
    # summary["density_per_100_images"] = (
    #     summary.apply(
    #         lambda row: (row["total_insects"] / row["total_images"]) * 100 if row["total_images"] else 0, axis=1
    #     )
    #     .round(1)
    # )
    st.markdown("**Location overview (aggregated counts)**")
    st.dataframe(summary, use_container_width=True)
    st.caption("Applications recommended when density reaches 16 insects per 100 images in the same location.")

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="stinkbug_reports.csv",
        mime="text/csv",
    )


def render_login_register():
    login_col, register_col = st.columns(2)

    with login_col:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in"):
            user = authenticate_user(username.strip(), password)
            if user:
                st.session_state["auth_user"] = user
                st.session_state["menu_choice"] = "Homepage"
                st.success(f"Welcome back, {user['username']}! Redirecting to Homepage...")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_col:
        st.header("Register")
        new_username = st.text_input("New username", key="register_username")
        email = st.text_input("Email", key="register_email")
        new_password = st.text_input("New password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm password", type="password", key="register_confirm")
        if st.button("Create account"):
            if not all([new_username.strip(), email.strip(), new_password, confirm_password]):
                st.error("All fields are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(new_username.strip(), email.strip(), new_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


def main():
    st.set_page_config(page_title="Redbanded Stink bug Adults", layout="wide")
    ensure_session_defaults()
    init_db()

    if not st.session_state["auth_user"]:
        render_login_register()
        return

    with st.sidebar:
        st.title("Redbanded Stin kbug Adults")
        st.image("https://www.pioneer.com/content/dam/dpagco/pioneer/na/us/en/agronomy/crop_focus/soybeans/pests/redbanded-stink-bug-adult-top-view.jpg", width=5000)
        st.title("Menu")
        st.session_state["menu_choice"] = st.radio(
            "",
            key="nav_menu",
            options=["Homepage", "Capture Image", "Reports"],
            index=["Homepage", "Capture Image", "Reports"].index(st.session_state["menu_choice"]),
        )
        st.divider()
        st.button("Logout", on_click=logout)

    choice = st.session_state["menu_choice"]
    if choice == "Homepage":
        render_home()
    elif choice == "Capture Image":
        render_capture()
    elif choice == "Reports":
        render_reports()


if __name__ == "__main__":
    main()

