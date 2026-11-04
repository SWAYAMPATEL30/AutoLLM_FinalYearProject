import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autollm.security import (
    sanitize_job_id,
    safe_join_paths,
    sanitize_user_prompt,
    validate_inference_params,
    validate_uploaded_file
)

class TestSecurity(unittest.TestCase):
    def test_sanitize_job_id_valid(self):
        self.assertEqual(sanitize_job_id("job_12345"), "job_12345")
        self.assertEqual(sanitize_job_id("job-abc_def-01"), "job-abc_def-01")

    def test_sanitize_job_id_traversal(self):
        with self.assertRaises(ValueError):
            sanitize_job_id("../secret")
        with self.assertRaises(ValueError):
            sanitize_job_id("..\\Windows\\System32")
        with self.assertRaises(ValueError):
            sanitize_job_id("job/subfolder")
        with self.assertRaises(ValueError):
            sanitize_job_id("")

    def test_safe_join_paths_contained(self):
        base = os.path.abspath("C:/temp/base")
        joined = safe_join_paths(base, "job_123", "export")
        self.assertTrue(joined.startswith(base))

    def test_safe_join_paths_escape_raises(self):
        base = os.path.abspath("C:/temp/base")
        with self.assertRaises(PermissionError):
            safe_join_paths(base, "..", "secret.txt")

    def test_validate_inference_params_clamping(self):
        temp, tokens = validate_inference_params(temperature=5.0, max_tokens=5000)
        self.assertEqual(temp, 2.0)
        self.assertEqual(tokens, 1024)

        temp, tokens = validate_inference_params(temperature=-1.0, max_tokens=0)
        self.assertEqual(temp, 0.01)
        self.assertEqual(tokens, 1)

    def test_sanitize_user_prompt_length(self):
        long_prompt = "a" * 3000
        cleaned = sanitize_user_prompt(long_prompt, max_length=2000)
        self.assertEqual(len(cleaned), 2000)

    def test_validate_uploaded_file_valid(self):
        content = b"Medical triage guidelines for diabetes treatment."
        clean_text = validate_uploaded_file("guidelines.txt", content)
        self.assertEqual(clean_text, content.decode("utf-8"))

    def test_validate_uploaded_file_invalid_extension(self):
        with self.assertRaises(ValueError):
            validate_uploaded_file("malicious.exe", b"binary executable content")
        with self.assertRaises(ValueError):
            validate_uploaded_file("script.py", b"import os; os.system('calc')")

    def test_validate_uploaded_file_null_byte_rejection(self):
        with self.assertRaises(ValueError):
            validate_uploaded_file("exploit\x00.txt", b"safe content")

    def test_validate_uploaded_file_size_limit(self):
        huge_content = b"A" * (6 * 1024 * 1024)  # 6 MB (limit is 5 MB)
        with self.assertRaises(ValueError):
            validate_uploaded_file("large.txt", huge_content, max_size_mb=5.0)

if __name__ == "__main__":
    unittest.main()
