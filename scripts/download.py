import kagglehub

# Download latest version
path = kagglehub.dataset_download("mindsetvision/mindset-lite", "data/")

print("Path to dataset files:", path)
