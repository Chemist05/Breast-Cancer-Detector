from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import io
import base64
import random
import cv2
import numpy as np
from ultralytics import YOLO

app = Dash(__name__)  # Initialize Dash app

app.layout = html.Div(children=[
    #header
    html.Div(className="header", children=[
        html.H1("Breast Cancer Detector")
    ]),

    #body
    html.Div(className="body", children=[
        html.Div(className="loaddiv", children=[
            dcc.Upload(id="upload-data", children=html.Div([
                "Drag and Drop or ", html.A("Select File", style={"color": "blue"})
            ]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            }, multiple=False)
        ]),

        #div contains images
        html.Div(className="images", children=[
            html.Div(className="image", children=[
                dcc.Graph(id="original", figure={}, style={"height": "700px", "width": "700px"})
            ]),

            html.Div(className="image", children=[
                dcc.Graph(id="predicted", figure={}, style={"height": "700px", "width": "700px"})
            ])
        ]),
        
        # div contains motivation text
        html.Div(className="motivation", children=[
            html.H1("important message", style={"textAlign" : "center", "padding" : "10px"}),
            html.H3("Breast cancer affects millions of women worldwide and is one of the biggest health challenges of our time. But despite this fact, we have the opportunity to reduce the risk and significantly increase the chances of a cure through early detection and regular screening. It is crucial to raise awareness of breast cancer and encourage women to take care of their health. By talking about this topic, we can not only spread knowledge, but also alleviate fear and encourage regular screening. Tell your friends, family, acquaintances or anyone you know. Education can save lives.", style={"padding" : "30px"})
            ])
    ])
])
            

# Function to process the uploaded image
def parse_data(content):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    np_img = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    resized_img = cv2.resize(img, (600, 600))  # Resize image for consistency
    return resized_img


@app.callback(
    Output("original", "figure"),
    Output("predicted", "figure"),
    Input("upload-data", "contents")
)


def show_images(content):
    if content is None:
        print("No content received.")
        return {}, {}

    img = parse_data(content)  # Process uploaded image
    frame = img.copy()

    # Load class names from file
    with open("kind_of_cancer_list.txt") as src:
        classlist = src.read().split("\n")

    # Generate random colors for each class
    detection_color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in classlist]
    
    # Load YOLO model for breast cancer detection
    model = YOLO("breastcancer_seg_model.pt")
    
    # Function to overlay masks on detected areas
    def overlay(original, masks, color, alpha=0.45):
        copy = original.copy()
        pts = np.array(masks, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(copy, [pts], color)
        return cv2.addWeighted(copy, alpha, original, 1 - alpha, 0)
    
    # Run model prediction on the uploaded image
    detect_param = model.predict(source=[frame], conf=0.45, save=False)
    results = detect_param[0]
    
    boxes = results.boxes
    masks = results.masks
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            clsID = int(box.cls[0])
            conf = float(box.conf[0])
            bb = box.xyxy[0].tolist()
        
            # Draw boxes around detected region
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_color[clsID], 2)
        
            # Overlay mask if available
            if masks is not None:
                mask = masks[i].xy
                frame = overlay(frame, mask, detection_color[clsID])
        
            # Display class name and confidence score
            cv2.putText(frame, f"{classlist[clsID]} {round(conf, 1)}%", (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display original and processed images
    fig1 = px.imshow(img)
    fig1.update_layout(coloraxis_showscale=False)

    fig2 = px.imshow(frame)
    fig2.update_layout(coloraxis_showscale=False)
    
    return fig1, fig2

if __name__ == "__main__":
    app.run_server(debug=True, port=1123)  # Run the Dash app
