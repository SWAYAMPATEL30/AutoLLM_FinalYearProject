import os
import re
from typing import Tuple

# Regex allowing only alphanumeric, underscores, and hyphens for identifiers
SAFE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")
ALLOWED_DOC_EXTENSIONS = {".txt", ".md", ".json", ".csv"}

def sanitize_job_id(job_id: str) -> str:
    """
    OWASP A01: Broken Access Control & Path Traversal Prevention.
    Validates that a job_id contains strictly alphanumeric characters, underscores, or hyphens.
    Prevents path traversal sequences (e.g. '../', '..\\') and null byte injections.
    """
    if not job_id or not isinstance(job_id, str):
        raise ValueError("Invalid job_id: Identifier cannot be empty.")
    
    clean_id = job_id.strip()
    if not SAFE_ID_REGEX.match(clean_id):
        raise ValueError(f"Invalid job_id: '{clean_id}' contains prohibited path traversal or control characters.")
    
    if len(clean_id) > 100:
        raise ValueError("Invalid job_id: Identifier exceeds maximum allowed length (100 characters).")
        
    return clean_id

def safe_join_paths(base_dir: str, *paths: str) -> str:
    """
    Safely joins path components and guarantees that the resolved absolute path
    resides strictly within the intended base directory.
    """
    resolved_base = os.path.abspath(base_dir)
    joined_path = os.path.abspath(os.path.join(resolved_base, *paths))
    
    # Common path must match base directory
    try:
        common = os.path.commonpath([resolved_base, joined_path])
    except ValueError:
        # Happens on Windows across different drive letters
        raise PermissionError(f"Security Alert: Path '{joined_path}' attempts to traverse across storage volumes.")
        
    if common != resolved_base:
        raise PermissionError(f"Security Alert: Path traversal attempt blocked. Target '{joined_path}' escapes base directory '{resolved_base}'.")
        
    return joined_path

def sanitize_user_prompt(prompt: str, max_length: int = 2000) -> str:
    """
    OWASP A03/A04: Input Validation and Sanitization.
    Normalizes free-text user prompt and enforces safe operational length limits.
    """
    if not prompt or not isinstance(prompt, str):
        return ""
        
    # Strip null bytes and non-printable control characters except standard whitespace
    cleaned = "".join(ch for ch in prompt if ch.isprintable() or ch in "\n\r\t")
    cleaned = cleaned.strip()
    
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
        
    return cleaned

def validate_inference_params(temperature: float = 0.7, max_tokens: int = 150) -> Tuple[float, int]:
    """
    Prevents Denial of Service (DoS) and numerical instability during model sampling.
    """
    try:
        temp = float(temperature)
    except (TypeError, ValueError):
        temp = 0.7
    temp = max(0.01, min(temp, 2.0))
    
    try:
        tokens = int(max_tokens)
    except (TypeError, ValueError):
        tokens = 150
    tokens = max(1, min(tokens, 1024))
    
    return temp, tokens

def validate_uploaded_file(filename: str, content_bytes: bytes, max_size_mb: float = 5.0) -> str:
    """
    OWASP A03/A04: File Upload & Document Content Validation.
    Enforces extension whitelisting, size limits, and UTF-8 decodability.
    Strips null bytes and harmful control characters.
    """
    if not filename or not isinstance(filename, str):
        raise ValueError("Invalid upload: Filename is required.")
        
    if "\x00" in filename:
        raise ValueError("Null byte injection detected in filename.")

    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise ValueError(f"Invalid file type '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_DOC_EXTENSIONS))}")
        
    max_bytes = int(max_size_mb * 1024 * 1024)
    if len(content_bytes) > max_bytes:
        raise ValueError(f"File size exceeds maximum allowed limit of {max_size_mb} MB.")
        
    if len(content_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
        
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content_bytes.decode("latin-1")
        except Exception:
            raise ValueError("Uploaded file could not be decoded. Please upload a UTF-8 text file.")
            
    # Strip null bytes
    text = text.replace("\x00", "")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\r\t")
    clean_text = text.strip()
    
    if not clean_text:
        raise ValueError("Uploaded file contains no readable text.")
        
    return clean_text
