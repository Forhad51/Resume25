import os
import re
import csv
import pdfplumber
import docx2txt
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor

# Load the LLaMA model locally
llm = Llama(model_path="E:\\LLMa\\llama-2-7b.Q2_K.gguf", n_ctx=1024)

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

# Function to extract GitHub & LinkedIn links using regex
def extract_links(text):
    github_links = re.findall(GITHUB_PATTERN, text)
    linkedin_links = re.findall(LINKEDIN_PATTERN, text)

    github_results = {link: "Valid" if validate_github(link) else "Invalid" for link in github_links} if github_links else {"Not Available": ""}
    linkedin_results = {link: "Valid" if validate_linkedin(link) else "Invalid" for link in linkedin_links} if linkedin_links else {"Not Available": ""}

    return {"GitHub": github_results, "LinkedIn": linkedin_results}

# Function to extract full summary using LLaMA model
def extract_summary(text):
    chunk_size = 500  # Process smaller chunks for faster processing
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    full_summary = ""
    for chunk in chunks:
        prompt = ("Summarize the following resume content concisely. Extract key skills, experiences, "
                  "and important details. Keep it within a short paragraph.\n\n"
                  f"Resume Text:\n{chunk}")
        response = llm(prompt)
        if "choices" in response:
            full_summary += response["choices"][0]["text"].strip() + " "
    
    return full_summary.strip() if full_summary else "Summary not available"

# Function to process a single resume or multiple resumes from a given folder
def process_resume(file_path):
    text = ""
    if file_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif file_path.lower().endswith(".txt"):
        text = extract_text_from_txt(file_path)
    
    if text:
        extracted_links = extract_links(text)
        summary = extract_summary(text)
        return {"File Name": os.path.basename(file_path), "Summary": summary, **extracted_links}

    return None

# Process resumes in parallel
def process_resumes(folder_or_file_path):
    results = []
    if os.path.isdir(folder_or_file_path):
        files = [os.path.join(folder_or_file_path, file_name) for file_name in os.listdir(folder_or_file_path) if file_name.lower().endswith((".pdf", ".docx", ".txt"))]
    elif os.path.isfile(folder_or_file_path) and folder_or_file_path.lower().endswith((".pdf", ".docx", ".txt")):
        files = [folder_or_file_path]
    else:
        print("Error: The specified path is neither a valid file nor a folder containing supported files.")
        return

    # Process resumes concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_resume, files))

    # Filter out None results (in case of any failed processing)
    results = [result for result in results if result]

    # Save results to CSV
    output_file = os.path.join(os.path.dirname(folder_or_file_path), "extracted_links.csv")
    with open(output_file, "w", encoding="utf-8", newline='') as csv_file:
        fieldnames = ["File Name", "Summary", "GitHub", "LinkedIn"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            github_links = ', '.join([f"{link}: {status}" for link, status in result.get("GitHub", {}).items()])
            linkedin_links = ', '.join([f"{link}: {status}" for link, status in result.get("LinkedIn", {}).items()])
            writer.writerow({"File Name": result["File Name"], "Summary": result["Summary"], "GitHub": github_links, "LinkedIn": linkedin_links})
    
    print(f"Extraction completed! Results saved in '{output_file}'.")

    # Save results in a text file
    output_text_file = os.path.join(os.path.dirname(folder_or_file_path), "extracted_links_summary.txt")
    with open(output_text_file, "w", encoding="utf-8") as text_file:
        for result in results:
            text_file.write(f"File Name: {result['File Name']}\n")
            text_file.write(f"Summary: {result['Summary']}\n")
            github_links = '\n'.join([f"{link}: {status}" for link, status in result.get("GitHub", {}).items()])
            linkedin_links = '\n'.join([f"{link}: {status}" for link, status in result.get("LinkedIn", {}).items()])
            text_file.write(f"GitHub Links:\n{github_links}\n")
            text_file.write(f"LinkedIn Links:\n{linkedin_links}\n")
            text_file.write("\n" + "-"*50 + "\n")

    print(f"Text results saved in '{output_text_file}'.")

# Run the script
if __name__ == "__main__":
    folder_or_file = input("Enter the full path of the folder or a resume file (.pdf, .docx, .txt): ")
    process_resumes(folder_or_file)