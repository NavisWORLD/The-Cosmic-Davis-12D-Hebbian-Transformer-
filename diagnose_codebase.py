import os
import sys
import ast
import importlib
import traceback
import subprocess

# Set up paths exactly like the launcher
ROOT_DIR = os.getcwd()
PACKAGE_DIR = os.path.join(ROOT_DIR, 'packages', 'cosmic-synapse-transformer')
sys.path.insert(0, PACKAGE_DIR)

# Also add the root for imports relative to root (if any)
sys.path.insert(0, ROOT_DIR)

def check_syntax(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"ReadError: {e}"

def check_import(module_name):
    # run in subprocess to avoid crashing this script or polluting sys.modules
    code = f"import sys; sys.path.insert(0, r'{PACKAGE_DIR}'); sys.path.insert(0, r'{ROOT_DIR}'); import {module_name}"
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)

def get_module_name(file_path, root):
    # Convert file path to module name relative to root or package dir
    # This is tricky because some files are run as scripts, not modules.
    # We will try to guess.
    
    # attempt relative to package dir
    rel_pkg = os.path.relpath(file_path, PACKAGE_DIR)
    if not rel_pkg.startswith('..'):
        return rel_pkg.replace(os.path.sep, '.').replace('.py', '')
    
    # attempt relative to root
    rel_root = os.path.relpath(file_path, ROOT_DIR)
    return rel_root.replace(os.path.sep, '.').replace('.py', '')

def scan_directory(directory):
    report = []
    print(f"Scanning {directory}...")
    
    for root, dirs, files in os.walk(directory):
        # Skip some dirs
        if '.git' in dirs: dirs.remove('.git')
        if '__pycache__' in dirs: dirs.remove('__pycache__')
        if 'node_modules' in dirs: dirs.remove('node_modules')
        
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                
                # 1. Syntax Check
                valid_syntax, syntax_err = check_syntax(full_path)
                
                # 2. Import Check (only if syntax is valid)
                valid_import = False
                import_err = "Skipped due to syntax"
                if valid_syntax:
                   # Try to import it. If it's a script in root, often we skip import check, 
                   # but let's try importing it as a module to check dependencies.
                   module_name = get_module_name(full_path, directory)
                   valid_import, import_err = check_import(module_name)
                
                report.append({
                    'file': full_path,
                    'syntax': valid_syntax,
                    'syntax_err': syntax_err,
                    'import': valid_import,
                    'import_err': import_err
                })
    return report

def main():
    dirs_to_scan = [
        os.path.join(ROOT_DIR, 'production_12d'),
        os.path.join(ROOT_DIR, 'research_42d'),
        os.path.join(ROOT_DIR, 'packages', 'cosmic-synapse-transformer', 'cosmic_synapse')
    ]
    
    full_report = []
    for d in dirs_to_scan:
        if os.path.exists(d):
            full_report.extend(scan_directory(d))
        else:
            print(f"Warning: Directory {d} not found.")

    output_lines = []
    output_lines.append("====== DIAGNOSTIC REPORT ======")
    
    failed_syntax = [r for r in full_report if not r['syntax']]
    failed_import = [r for r in full_report if r['syntax'] and not r['import']]
    
    if failed_syntax:
        output_lines.append(f"\n[CRITICAL] SYNTAX ERRORS ({len(failed_syntax)} files):")
        for r in failed_syntax:
            output_lines.append(f"  [X] {os.path.basename(r['file'])}: {r['syntax_err']}")
            
    if failed_import:
        output_lines.append(f"\n[WARNING] IMPORT ERRORS ({len(failed_import)} files):")
        for r in failed_import:
            # Shorten output
            err = r['import_err'].split('\n')[-1] if r['import_err'] else "Unknown"
            output_lines.append(f"  [!] {os.path.basename(r['file'])}: {err}")
            
    if not failed_syntax and not failed_import:
        output_lines.append("\n[OK] All scanned files passed syntax and import checks!")
    else:
        output_lines.append(f"\nTotal Files Scanned: {len(full_report)}")
        output_lines.append(f"Passed: {len(full_report) - len(failed_syntax) - len(failed_import)}")

    with open('diagnostic_results_safe.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print("Diagnostic complete. Results written to diagnostic_results_safe.txt")

if __name__ == "__main__":
    main()
