import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('best.pt')  # Ensure 'best.pt' is in the same directory

# Professional message
st.title("Book Detection with YOLOv8")
st.markdown("""
**Created by Siddharth Basale**

Upload an image containing books, and this app will detect the number of books and generate an image with the bounding boxes.
""")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform detection
    results = model(image)

    # Get the class ID for 'book'
    book_class_id = None
    for class_id, class_name in model.names.items():
        if class_name == 'book':
            book_class_id = class_id
            break

    # Count the number of books detected
    if book_class_id is not None:
        books_detected = len([r for r in results[0].boxes.data if int(r[-1]) == book_class_id])
    else:
        books_detected = 0  # No 'book' class detected in the model
    
    st.write(f"Number of books detected: {books_detected}")

    # Display the image with bounding boxes
    annotated_image = results[0].plot()  # Get image with bounding boxes
    st.image(annotated_image, caption="Processed Image with Book Detection", use_column_width=True)
