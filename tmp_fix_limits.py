import re
import sys

filepath = r'c:\Users\corys\The-Cosmic-Davis-12D-Hebbian-Transformer--1\Cosmos\cosmos\web\server.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Count before
predict_count = len(re.findall(r'"num_predict":\s*\d+', content))
maxlen_count = len(re.findall(r'max_length=\d+', content))
maxtok_count = len(re.findall(r'max_tokens=300', content))
print(f'BEFORE: num_predict restrictions: {predict_count}')
print(f'BEFORE: max_length truncations: {maxlen_count}')
print(f'BEFORE: max_tokens=300: {maxtok_count}')

# 1. Replace all num_predict values (80-500 range) with 4096
content = re.sub(r'"num_predict":\s*\d{2,3}(?=\s*[,})])', '"num_predict": 4096', content)

# 2. Remove max_length args from extract_ollama_content calls
content = re.sub(r'extract_ollama_content\(([^,]+),\s*max_length=\d+\)', r'extract_ollama_content(\1)', content)

# 3. Replace remaining max_tokens=300 calls
content = content.replace('max_tokens=300', 'max_tokens=4096')

# Count after
predict_count2 = len(re.findall(r'"num_predict":\s*4096', content))
maxlen_count2 = len(re.findall(r'max_length=\d+', content))
maxtok_count2 = len(re.findall(r'max_tokens=4096', content))
print(f'AFTER: num_predict=4096: {predict_count2}')
print(f'AFTER: max_length truncations remaining: {maxlen_count2}')
print(f'AFTER: max_tokens=4096: {maxtok_count2}')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('DONE - All limits removed')
