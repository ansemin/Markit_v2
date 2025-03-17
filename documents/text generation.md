| Gemini 2.0 Flash is now production ready! |                                                                 |
|-------------------------------------------|-----------------------------------------------------------------|
| Learn<br>more                             | (https://developers.googleblog.com/en/gemini-2-family-expands/) |
|                                           |                                                                 |
|                                           |                                                                 |
|                                           |                                                                 |

# Text generation

checkedPython Node.js Go REST

The Gemini API can generate text output when provided text, images, video, and audio as input.

This guide shows you how to generate text using the [generateContent](https://ai.google.dev/api/rest/v1/models/generateContent) [(/api/rest/v1/models/generateContent)](https://ai.google.dev/api/rest/v1/models/generateContent) and [streamGenerateContent](https://ai.google.dev/api/rest/v1/models/streamGenerateContent) [(/api/rest/v1/models/streamGenerateContent)](https://ai.google.dev/api/rest/v1/models/streamGenerateContent) methods. To learn about working with Gemini's vision and audio capabilities, refer to the Vision [(/gemini-api/docs/vision)](https://ai.google.dev/gemini-api/docs/vision) and [Audio](https://ai.google.dev/gemini-api/docs/audio) [(/gemini-api/docs/audio)](https://ai.google.dev/gemini-api/docs/audio) guides.

### Generate text from text-only input

The simplest way to generate text using the Gemini API is to provide the model with a single text-only input, as shown in this example:

```
from google import genai
client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"])
print(response.text)
```
In this case, the prompt ("Explain how AI works") doesn't include any output examples, system instructions, or formatting information. It's a [zero-shot](https://ai.google.dev/gemini-api/docs/models/generative-models#zero-shot-prompts)

[(/gemini-api/docs/models/generative-models#zero-shot-prompts)](https://ai.google.dev/gemini-api/docs/models/generative-models#zero-shot-prompts) approach. For some use cases, a one-shot [(/gemini-api/docs/models/generative-models#one-shot-prompts)](https://ai.google.dev/gemini-api/docs/models/generative-models#one-shot-prompts) or [few-shot](https://ai.google.dev/gemini-api/docs/models/generative-models#few-shot-prompts) [(/gemini-api/docs/models/generative-models#few-shot-prompts)](https://ai.google.dev/gemini-api/docs/models/generative-models#few-shot-prompts) prompt might produce output that's more aligned with user expectations. In some cases, you might also want to provide

system instructions [(/gemini-api/docs/text-generation#system-instructions)](https://ai.google.dev/gemini-api/docs/text-generation#system-instructions) to help the model understand the task or follow specic guidelines.

## Generate text from text-and-image input

The Gemini API supports multimodal inputs that combine text and media les. The following example shows how to generate text from text-and-image input:

```
from PIL import Image
from google import genai
client = genai.Client(api_key="GEMINI_API_KEY")
image = Image.open("/path/to/organ.png")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, "Tell me about this instrument"])
print(response.text)
```
# Generate a text stream

By default, the model returns a response after completing the entire text generation process. You can achieve faster interactions by not waiting for the entire result, and instead use streaming to handle partial results.

The following example shows how to implement streaming using the streamGenerateContent [(/api/rest/v1/models/streamGenerateContent)](https://ai.google.dev/api/rest/v1/models/streamGenerateContent) method to generate text from a text-only input prompt.

```
from google import genai
client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"])
for chunk in response:
    print(chunk.text, end="")
```
# Create a chat conversation

The Gemini SDK lets you collect multiple rounds of questions and responses, allowing users to step incrementally toward answers or get help with multipart problems. This SDK feature provides an interface to keep track of conversations history, but behind the scenes uses the same generateContent method to create the response.

The following code example shows a basic chat implementation:

```
from google import genai
client = genai.Client(api_key="GEMINI_API_KEY")
chat = client.chats.create(model="gemini-2.0-flash")
response = chat.send_message("I have 2 dogs in my house.")
print(response.text)
response = chat.send_message("How many paws are in my house?")
print(response.text)
for message in chat._curated_history:
    print(f'role - {message.role}' end=": ")
    print(message.parts[0].text)
```
You can also use streaming with chat, as shown in the following example:

```
from google import genai
client = genai.Client(api_key="GEMINI_API_KEY")
chat = client.chats.create(model="gemini-2.0-flash")
response = chat.send_message_stream("I have 2 dogs in my house.")
for chunk in response:
    print(chunk.text, end="")
response = chat.send_message_stream("How many paws are in my house?")
for chunk in response:
    print(chunk.text, end="")
for message in chat._curated_history:
    print(f'role - {message.role}', end=": ")
    print(message.parts[0].text)
```
# Congure text generation

Every prompt you send to the model includes parameters that control how the model generates responses. You can use GenerationConfig [(/api/rest/v1/GenerationCong)](https://ai.google.dev/api/rest/v1/GenerationConfig) to congure these parameters. If you don't congure the parameters, the model uses default options, which can vary by model.

The following example shows how to congure several of the available options.

```
from google import genai
from google.genai import types
client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"],
    config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.1
    )
)
print(response.text)
```
# Add system instructions

System instructions let you steer the behavior of a model based on your specic needs and use cases.

By giving the model system instructions, you provide the model additional context to understand the task, generate more customized responses, and adhere to specic guidelines over the full user interaction with the model. You can also specify product-level behavior by setting system instructions, separate from prompts provided by end users.

You can set system instructions when you initialize your model:

```
sys_instruct="You are a cat. Your name is Neko."
client = genai.Client(api_key="GEMINI_API_KEY")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=sys_instruct),
```
)

.

```
contents=["your prompt here"]
```
Then, you can send requests to the model as usual.

For an interactive end to end example of using system instructions, see the [system](https://colab.sandbox.google.com/github/google-gemini/cookbook/blob/main/quickstarts/System_instructions.ipynb) [instructions](https://colab.sandbox.google.com/github/google-gemini/cookbook/blob/main/quickstarts/System_instructions.ipynb) colab

[(https://colab.sandbox.google.com/github/google](https://colab.sandbox.google.com/github/google-gemini/cookbook/blob/main/quickstarts/System_instructions.ipynb)[gemini/cookbook/blob/main/quickstarts/System_instructions.ipynb)](https://colab.sandbox.google.com/github/google-gemini/cookbook/blob/main/quickstarts/System_instructions.ipynb)

### What's next

Now that you have explored the basics of the Gemini API, you might want to try:

- Vision understanding [(/gemini-api/docs/vision)](https://ai.google.dev/gemini-api/docs/vision): Learn how to use Gemini's native vision understanding to process images and videos.
- Audio understanding [(/gemini-api/docs/audio)](https://ai.google.dev/gemini-api/docs/audio): Learn how to use Gemini's native audio understanding to process audio les.

Except as otherwise noted, the content of this page is licensed under the Creative Commons [Attribution](https://creativecommons.org/licenses/by/4.0/) 4.0 License [(https://creativecommons.org/licenses/by/4.0/)](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache](https://www.apache.org/licenses/LICENSE-2.0) 2.0 License [(https://www.apache.org/licenses/LICENSE-2.0)](https://www.apache.org/licenses/LICENSE-2.0). For details, see the Google [Developers](https://developers.google.com/site-policies) Site Policies [(https://developers.google.com/site-policies)](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its aliates.

Last updated 2025-03-04 UTC.