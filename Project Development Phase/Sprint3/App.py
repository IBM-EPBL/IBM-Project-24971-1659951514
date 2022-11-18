from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("nutrition.h5")

app = Flask(__name__)

tf.saved_model.LoadOptions(
    allow_partial_checkpoint=False,
    experimental_io_device='/job:localhost',
    experimental_skip_checkpoint=False,
    experimental_variable_policy=None
)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/classify', methods=["GET", "POST"])
def classify():
    if request.method == 'POST':
      f = request.files['image']
      filename = f.filename
      f.save(secure_filename(f.filename))
    #   return 'file uploaded successfully'
    path = "D:\ibm\IBM-Project-24971-1659951514\Project Development Phase\Sprint3\\" + filename
    image = load_img(path, grayscale = False, target_size = (64,64))
    x = img_to_array(image)
#changing the shape
    x = np.expand_dims(x,axis = 0)
    predict_x = model.predict(x)
    classes_x = np.argmax(predict_x,axis = -1)
    index = ['BANANA','APPLES', 'ORANGE', 'PINEAPPLE', 'WATERMELON']
    result = str(index[classes_x[0]])

    print("worked", result)
    return result
if __name__ == "__main__":
    app.run(debug=True)