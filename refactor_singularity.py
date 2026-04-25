import os
import sys

TARGET_STR = "42d_singularity"
REPLACE_STR = "singularity_42d"
ROOT_DIR = os.getcwd()

def replace_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if TARGET_STR in content:
            print(f"Updating {file_path}...")
            new_content = content.replace(TARGET_STR, REPLACE_STR)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

def main():
    print(f"Refactoring {TARGET_STR} -> {REPLACE_STR}...")
    
    # 1. Update file contents
    count = 0
    for root, dirs, files in os.walk(ROOT_DIR):
        if '.git' in root or 'node_modules' in root:
            continue
            
        for file in files:
            if file.endswith(('.py', '.bat', '.md', '.txt', '.json')):
                full_path = os.path.join(root, file)
                # avoid self
                if 'refactor_singularity.py' in full_path: continue
                
                replace_in_file(full_path)
                count += 1
                
    print(f"Scanned {count} files.")
    
    # 2. Rename directories
    # We need to do this carefully. 
    # Find all directories named 42d_singularity
    dirs_to_rename = []
    for root, dirs, files in os.walk(ROOT_DIR):
        if TARGET_STR in dirs:
            dirs_to_rename.append(os.path.join(root, TARGET_STR))
            
    for d in dirs_to_rename:
        new_path = d.replace(TARGET_STR, REPLACE_STR)
        print(f"Renaming directory: {d} -> {new_path}")
        try:
            os.rename(d, new_path)
        except Exception as e:
            print(f"Error renaming {d}: {e}")

    print("Refactor complete.")

if __name__ == "__main__":
    main()
