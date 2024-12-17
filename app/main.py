from flask import Flask, request, jsonify
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from datetime import datetime, timedelta
from pymongo import MongoClient
import gridfs
import tempfile
import os
import faiss
import numpy as np
from numpy.linalg import norm
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from clarifai.client.model import Model
from flask_cors import CORS
from PIL import Image

import pymysql
import requests
PICKLE_FEATURES = 'Images_features.pkl'
PICKLE_FILENAMES = 'filenames.pkl'
IMG_SIZE = (224, 224)
TOP_K = 5
FAISS_THRESHOLD = 0.5

model_urlTEXT = "https://clarifai.com/clarifai/main/models/moderation-multilingual-text-classification"
MODEL_URL = "https://clarifai.com/clarifai/main/models/general-image-detection"
pat = "c9bcfa03a89c476b91936cab785c8b0d"


thresholds = {
    "toxic": 0.8,        # Điều chỉnh thấp hơn để kiểm tra sớm các từ ngữ tiêu cực
    "obscene": 0.3,       # Dành cho từ ngữ thô tục
    "insult": 0.3,        # Xúc phạm
    "identity_hate": 0.3,  # Ghét bỏ nhóm đối tượng
    "threat": 0.3,        # Đe dọa
    "severe_toxic": 0.3,  # Mức độ cực kỳ tiêu cực
}

app = Flask(__name__)
CORS(app)

# Hàm để chạy công việc tự động


def scheduled_task():
    """API endpoint to trigger the synchronization task."""
    # Fetch data from MySQL

    mysql_data = get_mysql_data()

    if mysql_data:
        print("Bắt đầu đồng bộ dữ liệu...")
        sync_to_mongodb(mysql_data)
        sync_data()

        print("Data sync completed successfully!")
    else:
        print("No data from MySQL to sync.")



# Khởi tạo BackgroundScheduler
scheduler = BackgroundScheduler()

# Sử dụng timezone từ pytz
tz = pytz.timezone('Asia/Ho_Chi_Minh')

# Đặt công việc chạy mỗi 24 giờ
scheduler.add_job(scheduled_task, 'interval', minutes=60, timezone=tz)

# Bắt đầu scheduler
scheduler.start()


def get_mysql_data():
    """Truy xuất dữ liệu từ MySQL."""
    connection = pymysql.connect(
        host='103.72.99.71',
        user='root',
        password='12345678',
        database='ebookLibrary'
    )
    try:
        with connection.cursor() as cursor:
            sql = "SELECT id, product_id, name FROM imageproducts"
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    except pymysql.MySQLError as e:
        print(f"Lỗi MySQL: {e}")
        return []
    finally:
        connection.close()


def download_image(url):
    """Tải ảnh từ URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        return response.content  # Trả về nội dung ảnh
    except requests.RequestException as e:
        print(f"Lỗi khi tải ảnh từ {url}: {e}")
        return None


def sync_to_mongodb(data):
    """Đồng bộ dữ liệu vào MongoDB với GridFS."""
    client = MongoClient(
        "mongodb://root:12345678@103.72.99.71:27017/?authSource=admin")
    db = client["MyDatabase"]
    fs = gridfs.GridFS(db)
    db["fs.files"].delete_many({})  # Xóa tất cả file metadata
    db["fs.chunks"].delete_many({})  # Xóa tất cả chunk dữ liệu
    for row in data:
        image_url = row[2]  # Trường 'name' chứa URL ảnh

        # Tải ảnh từ URL
        image_data = download_image(image_url)

        if image_data:
            # Lưu ảnh vào MongoDB GridFS
            file_id = fs.put(
                image_data,
                filename=image_url.split("/")[-1],
                metadata={
                    "product_id": row[1],  # Lưu product_id từ MySQL
                }
            )
            print(f"Đã lưu file với ID: {file_id} từ URL: {image_url}")

            # Tạo document MongoDB từ dữ liệu MySQL
            document = {
                "id": row[0],
                "file_id": file_id  # Lưu ID file trong GridFS
            }

            # Kiểm tra nếu document đã tồn tại trong MongoDB
            existing_doc = db["MyCollection"].find_one({"id": row[0]})
            if existing_doc:
                # Cập nhật nếu đã tồn tại
                db["MyCollection"].update_one(
                    {"id": row[0]},
                    {"$set": document}
                )
                print(f"Đã cập nhật document với id: {row[0]}")
            else:
                # Thêm mới nếu không tồn tại
                db["MyCollection"].insert_one(document)
                print(f"Đã thêm document mới với id: {row[0]}")


def load_model():
    """Tải mô hình ResNet50 đã được huấn luyện sẵn."""
    base_model = ResNet50(weights='imagenet',
                          include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False
    model = tf.keras.models.Sequential([base_model, GlobalMaxPool2D()])
    return model


def get_images_from_mongodb():
    """Lấy ảnh từ MongoDB GridFS và lưu vào thư mục tạm thời."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["MyDatabase"]
    fs = gridfs.GridFS(db)

    image_paths = []
    product_ids = []

    for file in fs.find():
        file_data = fs.get(file._id).read()  # Lấy dữ liệu ảnh từ GridFS
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(file_data)
            tmp_file.close()
            image_paths.append(tmp_file.name)  # Lưu đường dẫn tạm thời
            metadata = file.metadata if file.metadata else {}
            # Lưu product_id hoặc None
            product_ids.append(metadata.get('product_id', None))

    return image_paths, product_ids


def save_features_and_filenames(features, filenames, product_ids):
    """Lưu đặc trưng và tên file vào các file pickle."""
    with open(PICKLE_FEATURES, 'wb') as f:
        pkl.dump(features, f)
    with open(PICKLE_FILENAMES, 'wb') as f:
        pkl.dump({"filenames": filenames, "product_ids": product_ids}, f)


def load_features_and_filenames():
    """Tải đặc trưng và tên file từ các file pickle."""
    with open(PICKLE_FEATURES, 'rb') as f:
        features = pkl.load(f)
    with open(PICKLE_FILENAMES, 'rb') as f:
        data = pkl.load(f)
    return features, data["filenames"], data["product_ids"]


def extract_features_batch(model, image_paths, batch_size=32):
    """Trích xuất đặc trưng cho một batch ảnh từ danh sách các đường dẫn ảnh."""
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for path in batch_paths:
            img = image.load_img(path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            batch_images.append(img_array)
        batch_images = np.array(batch_images)
        batch_images = preprocess_input(batch_images)
        batch_features = model.predict(batch_images)
        features.append(batch_features)
    return np.vstack(features)


def build_faiss_index(features):
    """Tạo index FAISS từ các đặc trưng ảnh."""
    features_array = np.array(features, dtype='float32')
    features_array = features_array / \
        np.linalg.norm(features_array, axis=1, keepdims=True)
    index = faiss.IndexFlatL2(features_array.shape[1])  # L2 metric
    index.add(features_array)
    return index


def extract_features(image_path, model):
    """Trích xuất đặc trưng từ ảnh đầu vào."""
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))
    features = model.predict(img_preprocessed).flatten()
    return features / norm(features)


def find_similar_images(input_image, model, index, filenames, product_ids, k=TOP_K, threshold=FAISS_THRESHOLD):
    """Tìm kiếm ảnh tương tự từ cơ sở dữ liệu và trả về danh sách product_id."""
    input_image = input_image.convert("RGB")
    input_image.save("temp.jpg")  # Lưu tạm ảnh nhận từ request vào file
    input_features = extract_features("temp.jpg", model).astype('float32')

    distances, indices = index.search(np.array([input_features]), k)

    similar_product_ids = []  # Lưu danh sách product_id tương tự
    for idx, dist in zip(indices[0], distances[0]):
        if dist < threshold:
            # Lưu product_id tương tự
            similar_product_ids.append(product_ids[idx])

    return similar_product_ids


def sync_data():
    """Đồng bộ dữ liệu từ MongoDB vào pickle file."""
    model = load_model()
    image_paths, product_ids = get_images_from_mongodb()

    # Trích xuất đặc trưng từ ảnh
    features = extract_features_batch(model, image_paths)

    # Lưu đặc trưng và tên file vào pickle
    save_features_and_filenames(features, image_paths, product_ids)

    # Xóa các file tạm sau khi xử lý
    for path in image_paths:
        os.remove(path)

    print("Đồng bộ dữ liệu thành công!")


@app.route('/search-image', methods=['POST'])
def search_image():
    """Tìm kiếm sản phẩm tương tự với ảnh đầu vào."""
    # Nhận ảnh từ người dùng
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Load model và dữ liệu
    model = load_model()
    features, filenames, product_ids = load_features_and_filenames()
    index = build_faiss_index(features)

    # Tìm các sản phẩm tương tự
    input_image = Image.open(file.stream)
    similar_product_ids = find_similar_images(
        input_image, model, index, filenames, product_ids)

    # Trả kết quả
    return jsonify({'similar_product_ids': similar_product_ids})


@app.route('/check-text', methods=['POST'])
def check_text():
    try:
        # Lấy dữ liệu từ yêu cầu JSON
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        # Kiểm tra cấu trúc JSON
        if "data" not in data or "text" not in data["data"]:
            return jsonify({"error": "Missing 'data.text' field in request"}), 400

        text = data["data"]["text"]

        # Ghi văn bản vào tệp tạm thời
        temp_file_path = "temp_text.txt"
        with open(temp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(text)

        model = Model(url=model_urlTEXT, pat=pat)
        model_prediction = model.predict_by_filepath(temp_file_path, input_type="text")

        if not model_prediction.outputs:
            return jsonify({"error": "No outputs returned from model"}), 500

        concepts = model_prediction.outputs[-1].data.concepts
        predictions = {concept.name: concept.value for concept in concepts}

        violated_categories = []

        for category, threshold in thresholds.items():
            value = predictions.get(category, 0)
    
            print(f"Kiểm tra {category}: giá trị = {value}, ngưỡng = {threshold}")
    
            if value >= threshold:
                violated_categories.append({
                    "category": category,
                    "value": value,
                    "threshold": threshold
                })

        if violated_categories:
            return jsonify({
                "result": False,
            })

        return jsonify({
            "result": True,
        })
    except Exception as e:
        
        return jsonify({"error": str(e)}), 500


@app.route('/check-images', methods=['POST'])
def check_image():
    # Kiểm tra nếu file có trong yêu cầu
    detector_model = Model(url=MODEL_URL, pat=pat)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()

        prediction_response = detector_model.predict_by_bytes(image_bytes)

        regions = prediction_response.outputs[0].data.regions

        valid = False

        for region in regions:

            for concept in region.data.concepts:

                if concept.name in ["Poster", "Book"]:
                    valid = True

        if valid:
            return jsonify({"result": True})
        else:
            return jsonify({"result": False})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    except (KeyboardInterrupt, SystemExit):
        # Dừng scheduler khi thoát ứng dụng
        scheduler.shutdown()
