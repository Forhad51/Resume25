import os
import re
import csv
import pdfplumber
import docx2txt
from llama_cpp import Llama

# Load the LLaMA model locally
llm = Llama(model_path="E:\\LLMa\\llama-2-7b.Q2_K.gguf", n_ctx=512)

# Regular expressions to detect GitHub & LinkedIn links (full URLs)
GITHUB_PATTERN = r"https?://(?:www\.)?github\.com/[a-zA-Z0-9_-]+"
LINKEDIN_PATTERN = r"https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)

# Function to extract text from a TXT file
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to validate extracted links
def validate_linkedin(link):
    return bool(re.fullmatch(LINKEDIN_PATTERN, link))

def validate_github(link):
    return bool(re.fullmatch(GITHUB_PATTERN, link))

# Function to extract GitHub & LinkedIn links using LLaMA 2
def extract_links(text):
    github_links = re.findall(GITHUB_PATTERN, text)
    linkedin_links = re.findall(LINKEDIN_PATTERN, text)

    github_results = {link: "Valid" if validate_github(link) else "Invalid" for link in github_links} if github_links else {"Not Available": ""}
    linkedin_results = {link: "Valid" if validate_linkedin(link) else "Invalid" for link in linkedin_links} if linkedin_links else {"Not Available": ""}

    return {"GitHub": github_results, "LinkedIn": linkedin_results}

# Process a single resume or multiple resumes from a given folder
def process_resumes(folder_or_file_path):
    results = []

    if os.path.isdir(folder_or_file_path):
        for file_name in os.listdir(folder_or_file_path):
            file_path = os.path.join(folder_or_file_path, file_name)
            if file_name.lower().endswith(".docx"):
                text = extract_text_from_docx(file_path)
                if text:
                    extracted_links = extract_links(text)
                    results.append({"File Name": file_name, **extracted_links})
    elif os.path.isfile(folder_or_file_path) and folder_or_file_path.lower().endswith(".docx"):
        file_path = folder_or_file_path
        text = extract_text_from_docx(file_path)
        if text:
            extracted_links = extract_links(text)
            results.append({"File Name": os.path.basename(file_path), **extracted_links})
    else:
        print("Error: The specified path is neither a valid file nor a folder containing .docx files.")
        return

    output_file = os.path.join(os.path.dirname(folder_or_file_path), "extracted_links.csv")
    with open(output_file, "w", encoding="utf-8", newline='') as csv_file:
        fieldnames = ["File Name", "GitHub", "LinkedIn"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            github_links = ', '.join([f"{link}: {status}" for link, status in result.get("GitHub", {}).items()])
            linkedin_links = ', '.join([f"{link}: {status}" for link, status in result.get("LinkedIn", {}).items()])
            writer.writerow({"File Name": result["File Name"], "GitHub": github_links, "LinkedIn": linkedin_links})
    
    print(f"Extraction completed! Results saved in '{output_file}'.")

# Run the script
if __name__ == "__main__":
    folder_or_file = input("Enter the full path of the folder or .docx file: ")
    process_resumes(folder_or_file)
