![](_page_0_Figure_2.jpeg)

Learn more [(https://developers.googleblog.com/en/gemini-2-family-expands/)](https://developers.googleblog.com/en/gemini-2-family-expands/)

# Explore vision capabilities with the Gemini API

![](_page_0_Picture_5.jpeg)

Gemini models are able to process images and videos, enabling many frontier developer use cases that would have historically required domain specic models. Some of Gemini's vision capabilities include the ability to:

- Caption and answer questions about images
- Transcribe and reason over PDFs, including up to 2 million tokens
- Describe, segment, and extract information from videos up to 90 minutes long
- Detect objects in an image and return bounding box coordinates for them

Gemini was built to be multimodal from the ground up and we continue to push the frontier of what is possible.

## Image input

For total image payload size less than 20MB, we recommend either uploading base64 encoded images or directly uploading locally stored image les.

## Working with local images

If you are using the Python imaging library (Pillow [(https://pypi.org/project/pillow/)](https://pypi.org/project/pillow/)), you can use PIL image objects too.

```
from google import genai
from google.genai import types
import PIL.Image
image = PIL.Image.open('/path/to/image.png')
client = genai.Client(api_key=" ")
response = client.models.generate_content(
   model="gemini-2.0-flash",
   contents=["What is this image?", image])
print(response.text)
                              GEMINI_API_KEY edit
```
### Base64 encoded images

You can upload public image URLs by encoding them as Base64 payloads. The following code example shows how to do this using only standard library tools:

```
from google import genai
from google.genai import types
import requests
image_path = "https://goo.gle/instrument-img"
image = requests.get(image_path)
client = genai.Client(api_key=" ")
response = client.models.generate_content(
   model="gemini-2.0-flash-exp",
   contents=["What is this image?",
             types.Part.from_bytes(data=image.content, mime_type="image/jpeg"
print(response.text)
                              GEMINI_API_KEY edit
```
## Multiple images

To prompt with multiple images, you can provide multiple images in the call to generate_content. These can be in any supported format, including base64 or PIL.

```
from google import genai
from google.genai import types
import pathlib
import PIL.Image
image_path_1 = "path/to/your/image1.jpeg" # Replace with the actual path to y
image_path_2 = "path/to/your/image2.jpeg" # Replace with the actual path to yo
image_url_1 = "https://goo.gle/instrument-img" # Replace with the actual URL t
pil_image = PIL.Image.open(image_path_1)
b64_image = types.Part.from_bytes(
   data=pathlib.Path(image_path_2).read_bytes(),
   mime_type="image/jpeg"
)
downloaded_image = requests.get(image_url_1)
client = genai.Client(api_key=" ")
response = client.models.generate_content(
   model="gemini-2.0-flash-exp",
   contents=["What do these images have in common?",
             pil_image, b64_image, downloaded_image])
print(response.text)
                               GEMINI_API_KEY edit
```
Note that these inline data calls don't include many of the features available through the File API, such as getting le metadata, [listing](#page-10-0)

[(https://ai.google.dev/gemini-api/docs/vision?lang=python#list-les)](#page-10-0), or [deleting](#page-10-1) les [(https://ai.google.dev/gemini-api/docs/vision?lang=python#delete-les)](#page-10-1).

## Large image payloads

When the combination of les and system instructions that you intend to send is larger than 20 MB in size, use the File API to upload those les.

Use the media.upload [(https://ai.google.dev/api/rest/v1beta/media/upload)](https://ai.google.dev/api/rest/v1beta/media/upload) method of the File API to upload an image of any size.

Note: The File API lets you store up to 20 GB of les per project, with a per-le maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but cannot be

downloaded from the API. It is available at no cost in all regions where the Gemini API is available.

After uploading the le, you can make GenerateContent requests that reference the File API URI. Select the generative model and provide it with a text prompt and the uploaded image.

```
from google import genai
client = genai.Client(api_key=" ")
img_path = "/path/to/Cajun_instruments.jpg"
file_ref = client.files.upload(file=img_path)
print(f'{file_ref=}')
client = genai.Client(api_key=" ")
response = client.models.generate_content(
   model="gemini-2.0-flash-exp",
   contents=["What can you tell me about these instruments?",
             file_ref])
print(response.text)
                             GEMINI_API_KEY edit
                             GEMINI_API_KEY edit
```
## OpenAI Compatibility

You can access Gemini's image understanding capabilities using the OpenAI libraries. This lets you integrate Gemini into existing OpenAI workows by updating three lines of code and using your Gemini API key. See the Image [understanding](https://ai.google.dev/gemini-api/docs/openai#image-understanding) example [(https://ai.google.dev/gemini-api/docs/openai#image-understanding)](https://ai.google.dev/gemini-api/docs/openai#image-understanding) for code demonstrating how to send images encoded as Base64 payloads.

## Prompting with images

In this tutorial, you will upload images using the File API or as inline data and generate content based on those images.

Technical details (images)

Gemini 2.0 Flash, 1.5 Pro, and 1.5 Flash support a maximum of 3,600 image les.

Images must be in one of the following image data MIME types:

- PNG image/png
- JPEG image/jpeg
- WEBP image/webp
- HEIC image/heic
- HEIF image/heif

#### Tokens

Here's how tokens are calculated for images:

- Gemini 1.0 Pro Vision: Each image accounts for 258 tokens.
- Gemini 1.5 Flash and Gemini 1.5 Pro: If both dimensions of an image are less than or equal to 384 pixels, then 258 tokens are used. If one dimension of an image is greater than 384 pixels, then the image is cropped into tiles. Each tile size defaults to the smallest dimension (width or height) divided by 1.5. If necessary, each tile is adjusted so that it's not smaller than 256 pixels and not greater than 768 pixels. Each tile is then resized to 768x768 and uses 258 tokens.
- Gemini 2.0 Flash: Image inputs with both dimensions <=384 pixels are counted as 258 tokens. Images larger in one or both dimensions are cropped and scaled as needed into tiles of 768x768 pixels, each counted as 258 tokens.

#### For best results

- Rotate images to the correct orientation before uploading.
- Avoid blurry images.
- If using a single image, place the text prompt after the image.

## Capabilities

This section outlines specic vision capabilities of the Gemini model, including object detection and bounding box coordinates.

Get a bounding box for an object

Gemini models are trained to return bounding box coordinates as relative widths or heights in the range of [0, 1]. These values are then scaled by 1000 and converted to integers. Effectively, the coordinates represent the bounding box on a 1000x1000 pixel version of the image. Therefore, you'll need to convert these coordinates back to the dimensions of your original image to accurately map the bounding boxes.

```
from google import genai
client = genai.Client(api_key=" ")
prompt = (
 "Return a bounding box for each of the objects in this image "
 "in [ymin, xmin, ymax, xmax] format.")
response = client.models.generate_content(
 model="gemini-1.5-pro",
 contents=[sample_file_1, prompt])
print(response.text)
                              GEMINI_API_KEY edit
```
You can use bounding boxes for object detection and localization within images and video. By accurately identifying and delineating objects with bounding boxes, you can unlock a wide range of applications and enhance the intelligence of your projects.

#### Key Benets

- Simple: Integrate object detection capabilities into your applications with ease, regardless of your computer vision expertise.
- Customizable: Produce bounding boxes based on custom instructions (e.g. "I want to see bounding boxes of all the green objects in this image"), without having to train a custom model.

#### Technical Details

- Input: Your prompt and associated images or video frames.
- Output: Bounding boxes in the [y_min, x_min, y_max, x_max] format. The top left corner is the origin. The x and y axis go horizontally and vertically, respectively. Coordinate values are normalized to 0-1000 for every image.
- Visualization: AI Studio users will see bounding boxes plotted within the UI.

For Python developers, try the 2D spatial [understanding](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb) notebook

[(https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb)](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb) or the [experimental](https://github.com/google-gemini/cookbook/blob/main/examples/Spatial_understanding_3d.ipynb) 3D pointing notebook

[(https://github.com/google-gemini/cookbook/blob/main/examples/Spatial_understanding_3d.ipynb)](https://github.com/google-gemini/cookbook/blob/main/examples/Spatial_understanding_3d.ipynb).

#### Normalize coordinates

The model returns bounding box coordinates in the format [y_min, x_min, y_max, x_max]. To convert these normalized coordinates to the pixel coordinates of your original image, follow these steps:

- 1. Divide each output coordinate by 1000.
- 2. Multiply the x-coordinates by the original image width.
- 3. Multiply the y-coordinates by the original image height.

To explore more detailed examples of generating bounding box coordinates and visualizing them on images, we encourage you to review our Object Detection [cookbook](https://github.com/google-gemini/cookbook/blob/main/examples/Object_detection.ipynb) example [(https://github.com/google-gemini/cookbook/blob/main/examples/Object_detection.ipynb)](https://github.com/google-gemini/cookbook/blob/main/examples/Object_detection.ipynb).

## Prompting with video

In this tutorial, you will upload a video using the File API and generate content based on those images.

Note: The File API is required to upload video les, due to their size. However, the File API is only available for Python, Node.js, Go, and REST.

### Technical details (video)

Gemini 1.5 Pro and Flash support up to approximately an hour of video data.

Video must be in one of the following video format MIME types:

- video/mp4
- video/mpeg
- video/mov
- video/avi
- video/x-flv
- video/mpg
- video/webm
- video/wmv
- video/3gpp

The File API service extracts image frames from videos at 1 frame per second (FPS) and audio at 1Kbps, single channel, adding timestamps every second. These rates are subject to change in the future for improvements in inference.

Note: The details of fast action sequences may be lost at the 1 FPS frame sampling rate. Consider slowing down high-speed clips for improved inference quality.

Individual frames are 258 tokens, and audio is 32 tokens per second. With metadata, each second of video becomes ~300 tokens, which means a 1M context window can t slightly less than an hour of video.

To ask questions about time-stamped locations, use the format MM:SS, where the rst two digits represent minutes and the last two digits represent seconds.

For best results:

- Use one video per prompt.
- If using a single video, place the text prompt after the video.

## Upload a video le using the File API

Note: The File API lets you store up to 20 GB of les per project, with a per-le maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but they cannot be downloaded using any API. It is available at no cost in all regions where the Gemini API is available.

The File API accepts video le formats directly. This example uses the short NASA lm "Jupiter's Great Red Spot Shrinks and Grows" [(https://www.youtube.com/watch?v=JDi4IdtvDVE0)](https://www.youtube.com/watch?v=JDi4IdtvDVE0). Credit: Goddard Space Flight Center (GSFC)/David Ladd (2018).

"Jupiter's Great Red Spot Shrinks and Grows" is in the public domain and does not show identiable people. (NASA image and media usage [guidelines.](https://www.nasa.gov/nasa-brand-center/images-and-media/)

[(https://www.nasa.gov/nasa-brand-center/images-and-media/)](https://www.nasa.gov/nasa-brand-center/images-and-media/))

Start by retrieving the short video:

\$ wget https://storage.googleapis.com/generativeai-downloads/images/GreatRedSp

Upload the video using the File API and print the URI.

```
from google import genai
client = genai.Client(api_key=" ")
print("Uploading file...")
video_file = client.files.upload(file="GreatRedSpot.mp4")
print(f"Completed upload: {video_file.uri}")
                             GEMINI_API_KEY edit
```
## Verify le upload and check state

Verify the API has successfully received the les by calling the [files.get](https://ai.google.dev/api/rest/v1beta/files/get) [(https://ai.google.dev/api/rest/v1beta/les/get)](https://ai.google.dev/api/rest/v1beta/files/get) method.

Note: Video les have a State eld in the File API. When a video is uploaded, it will be in the PROCESSING state until it is ready for inference. Only ACTIVE les can be used for model inference.

```
import time
# Check whether the file is ready to be used.
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(1)
    video_file = client.files.get(name=video_file.name)
if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)
print('Done')
```
### Prompt with a video and text

Once the uploaded video is in the ACTIVE state, you can make GenerateContent requests that specify the File API URI for that video. Select the generative model and provide it with the uploaded video and a text prompt.

```
from IPython.display import Markdown
# Pass the video file reference like any other media part.
response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[
        video_file,
        "Summarize this video. Then create a quiz with answer key "
        "based on the information in the video."])
# Print the response, rendering any Markdown
Markdown(response.text)
```
### Refer to timestamps in the content

You can use timestamps of the form HH:MM:SS to refer to specic moments in the video.

```
prompt = "What are the examples given at 01:05 and 01:19 supposed to show us?"
response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[video_file, prompt])
print(response.text)
```
### Transcribe video and provide visual descriptions

The Gemini models can transcribe and provide visual descriptions of video content by processing both the audio track and visual frames. For visual descriptions, the model samples the video at a rate of 1 frame per second. This sampling rate may affect the level of detail in the descriptions, particularly for videos with rapidly changing visuals.

```
prompt = (
    "Transcribe the audio from this video, giving timestamps for "
    "salient events in the video. Also provide visual descriptions.")
```

```
response = client.models.generate_content(
    model="gemini-1.5-pro",
    contents=[video_file, prompt])
print(response.text)
```
## List les

<span id="page-10-0"></span>You can list all les uploaded using the File API and their URIs using [files.list](https://ai.google.dev/api/files#method:-files.list) [(/api/les#method:-les.list)](https://ai.google.dev/api/files#method:-files.list).

```
from google import genai
client = genai.Client(api_key=" ")
print('My files:')
for f in client.files.list():
 print(" ", f'{f.name}: {f.uri}')
                             GEMINI_API_KEY edit
```
## Delete les

<span id="page-10-1"></span>Files uploaded using the File API are automatically deleted after 2 days. You can also manually delete them using files.delete [(/api/les#method:-les.delete)](https://ai.google.dev/api/files#method:-files.delete).

```
from google import genai
client = genai.Client(api_key=" ")
# Upload a file
poem_file = client.files.upload(file="poem.txt")
# Files will auto-delete after a period.
print(poem_file.expiration_time)
# Or they can be deleted explicitly.
dr = client.files.delete(name=poem_file.name)
try:
 client.models.generate_content(
                              GEMINI_API_KEY edit
```

```
model="gemini-2.0-flash-exp",
      contents=['Finish this poem:', poem_file])
except genai.errors.ClientError as e:
  print(e.code) # 403
  print(e.status) # PERMISSION_DENIED
  print(e.message) # You do not have permission to access the File .. or it m
```
## What's next

This guide shows how to upload image and video les using the File API and then generate text outputs from image and video inputs. To learn more, see the following resources:

- File prompting strategies [(/gemini-api/docs/le-prompting-strategies)](https://ai.google.dev/gemini-api/docs/file-prompting-strategies): The Gemini API supports prompting with text, image, audio, and video data, also known as multimodal prompting.
- System instructions [(/gemini-api/docs/system-instructions)](https://ai.google.dev/gemini-api/docs/system-instructions): System instructions let you steer the behavior of the model based on your specic needs and use cases.
- Safety guidance [(/gemini-api/docs/safety-guidance)](https://ai.google.dev/gemini-api/docs/safety-guidance): Sometimes generative AI models produce unexpected outputs, such as outputs that are inaccurate, biased, or offensive. Post-processing and human evaluation are essential to limit the risk of harm from such outputs.

Except as otherwise noted, the content of this page is licensed under the Creative Commons [Attribution](https://creativecommons.org/licenses/by/4.0/) 4.0 License [(https://creativecommons.org/licenses/by/4.0/)](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache](https://www.apache.org/licenses/LICENSE-2.0) 2.0 License [(https://www.apache.org/licenses/LICENSE-2.0)](https://www.apache.org/licenses/LICENSE-2.0). For details, see the Google [Developers](https://developers.google.com/site-policies) Site Policies [(https://developers.google.com/site-policies)](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its aliates.

Last updated 2025-03-04 UTC.