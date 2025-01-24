import os
import re
import docx
import requests


# Extract specific information from text
def extract_info(text):
    # Patterns for extracting data
    name_pattern = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)"  # Supports multi-word names
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"(\+\d{1,4}\s?\d{7,14})"  # Supports international phone numbers
    address_pattern = r"[A-Za-z0-9\s,.-]+(?:,\s[A-Za-z\s]+)+"
    github_pattern = r"https?://(?:www\.)?github\.com/[a-zA-Z0-9_-]+"
    linkedin_pattern = r"https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+"

    # Extracting data
    name = re.search(name_pattern, text)
    address = re.search(address_pattern, text)
    phones = re.findall(phone_pattern, text)
    email = re.search(email_pattern, text)
    github = re.search(github_pattern, text)
    linkedin = re.search(linkedin_pattern, text)

    # Organize extracted data
    primary_phone = phones[0] if len(phones) > 0 else None
    secondary_phone = phones[1] if len(phones) > 1 else None

    return {
        "name": name.group(1) if name else None,
        "address": address.group(0).strip() if address else None,
        "primary_phone": primary_phone,
        "secondary_phone": secondary_phone,
        "email": email.group(0) if email else None,
        "github": github.group(0) if github else None,
        "linkedin": linkedin.group(0) if linkedin else None,
    }

# Validate URLs
def validate_url(url):
    if not url:
        return "No URL provided"
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            return "Valid"
        else:
            return "Invalid"
    except requests.RequestException:
        return "Invalid"

# Process resumes
def process_resumes(input_path):
    # Check if input is a file or folder
    if os.path.isfile(input_path):
        docx_files = [input_path]
    elif os.path.isdir(input_path):
        docx_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith('.docx')]
    else:
        print("Invalid path provided.")
        return

    if not docx_files:
        print("No .docx files found.")
        return

    # Process each file
    for file_path in docx_files:
        try:
            # Open and read the document
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])

            # Extract and format the information
            data = extract_info(text)
            
            # Validate GitHub and LinkedIn URLs
            github_status = validate_url(data['github'])
            linkedin_status = validate_url(data['linkedin'])

            # Print the formatted output
            print(f"File: {file_path}")
            print(f"Name: {data['name']}")
            print(f"Address: {data['address']}")
            print(f"Primary Mobile No: {data['primary_phone']}")
            print(f"Secondary Mobile No: {data['secondary_phone']}")
            print(f"Primary Email: {data['email']}")
            print(f"GitHub: {data['github']} ({github_status})")
            print(f"LinkedIn: {data['linkedin']} ({linkedin_status})")
            print('-' * 50)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Main process
if __name__ == "__main__":
    input_path = input("Enter the path to the folder or .docx file: ").strip()
    process_resumes(input_path)
