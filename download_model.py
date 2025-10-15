import ssl
from huggingface_hub import snapshot_download

# Disable SSL verification (temporary workaround)
ssl._create_default_https_context = ssl._create_unverified_context

snapshot_download(
    repo_id="vblagoje/bert-english-uncased-finetuned-pos",
    local_dir="models/bert-english-uncased-finetuned-pos",
    local_dir_use_symlinks=False,
)

print("✅ Model downloaded successfully to models/bert-english-uncased-finetuned-pos")
