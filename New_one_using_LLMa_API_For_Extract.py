import os
import re
import json
import docx2txt
import requests

# Regular expressions to detect GitHub & LinkedIn links
GITHUB_PATTERN = r"https?://(?:www\.)?github\.com/[a-zA-Z0-9_-]+"
LINKEDIN_PATTERN = r"https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+"

# Replace with your actual DeepInfra API Key
OPENAI_BASE_URL = "https://api.deepinfra.com/v1/openai"
API_KEY = "E:\\Python\\New1\\API_key"

def extract_text_using_llama(file_path):
    """Extracts text from a .docx file and processes it with LLaMA-3."""
    try:
        text = docx2txt.process(file_path)
        if not text:
            return ""
        
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 512
            }
        )

        return response.json()["choices"][0]["message"]["content"].strip() if response.status_code == 200 else text
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return text

def validate_link(link, pattern):
    return bool(re.fullmatch(pattern, link))

def extract_links(text):
    """Extracts GitHub and LinkedIn links from text."""
    github_links = re.findall(GITHUB_PATTERN, text)
    linkedin_links = re.findall(LINKEDIN_PATTERN, text)

    return {
        "GitHub": {link: "Valid" if validate_link(link, GITHUB_PATTERN) else "Invalid" for link in github_links} or {"Not Available": ""},
        "LinkedIn": {link: "Valid" if validate_link(link, LINKEDIN_PATTERN) else "Invalid" for link in linkedin_links} or {"Not Available": ""}
    }

def extract_experience_and_summary(text):
    """Extracts experience and summary from text using simple heuristics."""
    # Example: Look for keywords like "Experience" and "Summary"
    experience = ""
    summary = ""

    # Split text into sections
    sections = re.split(r'\n\s*\n', text)  # Split by double newlines
    for section in sections:
        if "experience" in section.lower():
            experience = section.strip()
        elif "summary" in section.lower():
            summary = section.strip()

    return {
        "Experience": experience or "Not Available",
        "Summary": summary or "Not Available"
    }

def process_resumes(folder_or_file_path):
    """Processes .docx files in a folder or a single file."""
    results = []
    
    if os.path.isdir(folder_or_file_path):
        files = [os.path.join(folder_or_file_path, f) for f in os.listdir(folder_or_file_path) if f.lower().endswith(".docx")]
    elif os.path.isfile(folder_or_file_path) and folder_or_file_path.lower().endswith(".docx"):
        files = [folder_or_file_path]
    else:
        print("Error: Invalid file or folder path.")
        return

    for file_path in files:
        text = extract_text_using_llama(file_path)
        extracted_links = extract_links(text)
        experience_and_summary = extract_experience_and_summary(text)
        
        # Combine all extracted data
        result = {
            "File Name": os.path.basename(file_path),
            **extracted_links,
            **experience_and_summary
        }
        results.append(result)

    # Output results in JSON format
    output_dir = os.path.dirname(folder_or_file_path) if os.path.isfile(folder_or_file_path) else folder_or_file_path
    os.makedirs(output_dir, exist_ok=True)
    
    json_output_file = os.path.join(output_dir, "extracted_data.json")
    with open(json_output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4)
    
    print(f"Extraction completed! Results saved in '{json_output_file}'")

if __name__ == "__main__":
    folder_or_file = input("Enter the full path of the folder or .docx file: ")
    process_resumes(folder_or_file)

