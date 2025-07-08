from groq import Groq
import base64
import os
import json
import re
import time


from dotenv import load_dotenv
import os

# Load from .env file
load_dotenv()

# --- CONFIG ---
API_KEY = os.getenv("GROQ_API_KEY")


# --- FUNCTIONS ---

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_json_blocks(text):
    return re.findall(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)

def process_image(image_path, client, prompt_type):
    base64_image = encode_image(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    if prompt_type == "clothes":
        prompt_text = (
            f"Describe only the clothes or accessories in this image: "
            f"include in this sequence ID  where the ID is '{image_name}.jpg, material, texture, style, color, type. "
            f"Return only one JSON block."
        )
    elif prompt_type == "person":
        prompt_text = (
            f"Describe only the person in the image in this sequence: IDwhere the ID is '{image_name}.jpg , age, gender, height, weight, hair color, hairstyle, skin color, and body type, "
            f"'. Return only one JSON block."
        )
    else:
        raise ValueError("Invalid prompt type. Use 'clothes' or 'person'.")

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    output = chat_completion.choices[0].message.content
    json_blocks = extract_json_blocks(output)

    if json_blocks:
        return json.loads(json_blocks[0])
    else:
        raise ValueError("No JSON block found in model output.")

# --- MAIN ---
client = Groq(api_key=API_KEY)

all_clothes_data = []
all_persons_data = []

# --- Process clothes ---

class GROQ_METADATA():
    def __init__(self, cloth_dir ,image_dir, out_cloth_dir,out_person_dir):
        self.output_clothes_json = out_cloth_dir
        self.output_persons_json = out_person_dir


        for filename in os.listdir(cloth_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(cloth_dir, filename)
                print(f"Processing clothes image: {filename}")
                
                flag = False
                while not flag:
                    try:
                        clothes_json = process_image(image_path, client, prompt_type="clothes")
                        all_clothes_data.append(clothes_json)
                        flag = True
                        print(f" Successfully processed: {filename}")
                    except Exception as e:
                        print(f" Failed to process {filename}, retrying in 2 seconds... Error: {e}")
                        # wait before retryin

        # --- Process persons ---
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(image_dir, filename)
                print(f"Processing person image: {filename}")

                flag = False
                while not flag:
                    try:
                        person_json = process_image(image_path, client, prompt_type="person")
                        all_persons_data.append(person_json)
                        flag = True
                        print(f" Successfully processed: {filename}")
                    except Exception as e:
                        print(f" Failed to process {filename}, retrying in 2 seconds... Error: {e}")
                        # Wait before retrying
    def generate_meta_data(self):
        # --- Save final JSON files ---
        with open( self.output_clothes_json, "w") as f:
            json.dump(all_clothes_data, f, indent=4)

        with open(self.output_persons_json, "w") as f:
            json.dump(all_persons_data, f, indent=4)

        # print(f"\n Saved all clothes to '{self.output_clothes_json}' and all persons to '{self.output_persons_json}'")
