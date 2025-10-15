import os
import ssl
import urllib3
from huggingface_hub import snapshot_download

# --- Disable SSL verification globally ---
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# ----------------------------------------

snapshot_download(
    repo_id="vblagoje/bert-english-uncased-finetuned-pos",
    local_dir="models/bert-english-uncased-finetuned-pos",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("✅ Model downloaded to models/bert-english-uncased-finetuned-pos")
