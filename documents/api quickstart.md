Gemini 2.0 Flash is now production ready! Learn more [(https://developers.googleblog.com/en/gemini-2-family-expands/)](https://developers.googleblog.com/en/gemini-2-family-expands/)

## Gemini API quickstart

This quickstart shows you how to install your SDK of choice and then make your rst Gemini API request.

checkedPython Node.js REST Go

## Install the Gemini API library

Note: We're rolling out a new set of Gemini API libraries, the Google Gen AI SDK [(/gemini-api/docs/sdks)](https://ai.google.dev/gemini-api/docs/sdks).

Using Python 3.9+ [(https://www.python.org/downloads/)](https://www.python.org/downloads/), install the [google-genai](https://pypi.org/project/google-genai/) package [(https://pypi.org/project/google-genai/)](https://pypi.org/project/google-genai/) using the following pip [command](https://packaging.python.org/en/latest/tutorials/installing-packages/) [(https://packaging.python.org/en/latest/tutorials/installing-packages/)](https://packaging.python.org/en/latest/tutorials/installing-packages/):

pip install -q -U google-genai

## Make your rst request

Get a Gemini API key in Google AI Studio [(https://aistudio.google.com/app/apikey)](https://aistudio.google.com/app/apikey)

Use the generateContent [(/api/generate-content#method:-models.generatecontent)](https://ai.google.dev/api/generate-content#method:-models.generatecontent) method to send a request to the Gemini API.

from google import genai

```
client = genai.Client(api_key=" ")
response = client.models.generate_content(
   model="gemini-2.0-flash", contents="Explain how AI works"
                             YOUR_API_KEY edit
```

```
)
print(response.text)
```
## What's next

Now that you made your rst API request, you might want to explore the following guides which showcase Gemini in action:

- Text generation [(/gemini-api/docs/text-generation)](https://ai.google.dev/gemini-api/docs/text-generation)
- Vision [(/gemini-api/docs/vision)](https://ai.google.dev/gemini-api/docs/vision)
- Long context [(/gemini-api/docs/long-context)](https://ai.google.dev/gemini-api/docs/long-context)

Except as otherwise noted, the content of this page is licensed under the Creative Commons [Attribution](https://creativecommons.org/licenses/by/4.0/) 4.0 License [(https://creativecommons.org/licenses/by/4.0/)](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache](https://www.apache.org/licenses/LICENSE-2.0) 2.0 License [(https://www.apache.org/licenses/LICENSE-2.0)](https://www.apache.org/licenses/LICENSE-2.0). For details, see the Google [Developers](https://developers.google.com/site-policies) Site Policies [(https://developers.google.com/site-policies)](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its aliates.

Last updated 2025-02-28 UTC.