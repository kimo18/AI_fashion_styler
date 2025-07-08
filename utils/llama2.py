
import subprocess
import json
import re
import shutil
import sys
import urllib.request

def is_ollama_installed():
    return shutil.which("ollama") is not None

def install_ollama_windows():
    print("Ollama not found. Downloading installer...")

    installer_url = "https://ollama.com/download/OllamaSetup.exe"  # hypothetical URL, please confirm the actual one
    installer_path = "OllamaSetup.exe"

    try:
        urllib.request.urlretrieve(installer_url, installer_path)
        print("Download completed. Running installer...")

        # Run the installer silently if supported (/S is common for silent install)
        # Otherwise, just run it normally
        subprocess.run([installer_path, "/S"], check=True)

        print("Installation complete. Please restart your terminal or computer.")
        sys.exit(0)  # Exit after installation

    except Exception as e:
        print(f"Failed to install Ollama: {e}")
        sys.exit(1)




def load_json_file(file_path,):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_to_file(data, filename="styled_output.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f" Output saved to {filename}")

def format_prompt(json1, json2, question):
    prompt = "You are given two JSON documents:\n\n"
    prompt += "JSON 1:\n" + json.dumps(json1, indent=2) + "\n\n"
    prompt += "JSON 2:\n" + json.dumps(json2, indent=2) + "\n\n"
    prompt += question
    return prompt

def extract_json(response_text):
    """
    Extracts the first valid JSON object from a larger text blob.
    """
    try:
        match = re.search(r"\{[\s\S]*\}", response_text)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        print(" Failed to decode JSON.")
    return None

def run_ollama(model: str, prompt: str):
    try:
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode()

    except subprocess.CalledProcessError as e:
        print(" Error running ollama:", e.stderr.decode())
        return None

# === Main ===

class LLAMA_Styler():
    def __init__(self,person_json_dir,clothes_json_dir):

        if not is_ollama_installed():
            install_ollama_windows()
        self.model_name = "llama3"  # or "mistral", etc.

        # Load JSON files
        self.json1 = load_json_file(person_json_dir,)
        self.json2 = load_json_file(clothes_json_dir,)

        # Define your strict prompt
        self.user_question = """You are a professional fashion stylist. You are given one or more people, each with a list of available clothing items. Check the 'type' of clothes to know which is 'pants' and which is 'shirt'. Each clothing item is represented by its unique ID.
        Your task:
        - For each person, select exactly one item with type 'shirt or top clothes'** and exactly one item with type 'pants or bottom clothes'**.
        - You are **strictly forbidden** from selecting more than one shirt or more than one pants per person.
        - You are **not allowed** to include any other clothing types (e.g. jackets, shoes, accessories).
        - Do not invent or assume any clothing items beyond what is provided.
        - Your answer must contain **only the selected clothing item IDs, nothing else.
        - Do not include any explanation, description, reasoning, or extra text.
        - The output must be a valid JSON object in this exact format
        {
            "Person 1": ["shirt_id", "pants_id"],
            "Person 2": ["shirt_id", "pants_id"]
        }
        """
    def style(self):
        full_prompt = format_prompt(self.json1, self.json2, self.user_question)
        response = run_ollama(self.model_name, full_prompt)

        print("\n--- Raw Response ---\n")
        print(response)

        parsed_output = extract_json(response)
        if parsed_output:
            save_json_to_file(parsed_output)
        else:
            print(" Could not extract valid JSON from the response.")
