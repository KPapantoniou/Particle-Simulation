import os
import ast
import sys

def get_py_info(filepath):
    """Extracts top-level classes and functions from a python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            node = ast.parse(f.read())

        info = []
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                info.append(f"  [C] {item.name}")
            elif isinstance(item, ast.FunctionDef):
                info.append(f"  [f] {item.name}")
        return info
    except Exception:
        return []

def list_files(startpath, max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return

    # Skip hidden folders and pycache
    entries = sorted([e for e in os.listdir(startpath)
                     if not e.startswith('.') and e != "__pycache__"])

    for i, entry in enumerate(entries):
        path = os.path.join(startpath, entry)
        is_last = (i == len(entries) - 1)
        prefix = "└── " if is_last else "├── "
        indent = "    " * current_depth

        print(f"{indent}{prefix}{entry}")

        # If it's a python file, extract info
        if entry.endswith(".py"):
            info = get_py_info(path)
            sub_indent = indent + ("    " if is_last else "│   ")
            for line in info:
                print(f"{sub_indent}    {line}")

        # Recurse into directories
        if os.path.isdir(path):
            list_files(path, max_depth, current_depth + 1)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print(target_dir)
    list_files(target_dir)
