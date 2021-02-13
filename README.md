# MEP-detection-app :mag_right:
A web application for detection MEP assets in images.

![main window](readme_assets/screen.png)

## Running the application locally
* Put the MEP object detection frozen inference graph  in a model folder. I have
  not included mine here because its huge but you can train your own using my
  [model development repo](https://github.com/Jam516/MEP-object-detection)
* Replicate the development environment using `pip install -r requirements.txt`.
* Run the app using `streamlit run app.py`
