#!/usr/bin/env python3
"""
Update ONNX proto version in CMakeLists.txt and regenerate types.

Usage:
    update_onnx_version.py <version>
    
Example:
    update_onnx_version.py 1.21.0
"""

import sys
import re
import subprocess
from pathlib import Path


def get_latest_onnx_version():
    """Fetch latest ONNX version from GitHub API"""
    try:
        import urllib.request
        import json
        url = "https://api.github.com/repos/onnx/onnx/releases/latest"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data["tag_name"].lstrip("v")
    except Exception as e:
        print(f"Warning: Could not fetch latest version: {e}")
        return None


def update_cmake_lists(version, cmake_path):
    """Update ONNX_PROTO_VERSION in CMakeLists.txt"""
    content = cmake_path.read_text()
    
    # Replace version string
    new_content = re.sub(
        r'set\(ONNX_PROTO_VERSION "[^"]+"',
        f'set(ONNX_PROTO_VERSION "{version}"',
        content
    )
    
    if new_content == content:
        print("No changes made to CMakeLists.txt")
        return False
    
    cmake_path.write_text(new_content)
    print(f"Updated CMakeLists.txt: ONNX_PROTO_VERSION = {version}")
    return True


def print_usage():
    print("Usage: update_onnx_version.py <version>")
    print("       update_onnx_version.py --latest")
    print("       update_onnx_version.py -h | --help")
    print("")
    print("Options:")
    print("  -h, --help    Show this help message")
    print("  --latest      Update to latest ONNX release")
    print("")
    print("Examples:")
    print("  update_onnx_version.py 1.21.0")
    print("  update_onnx_version.py --latest")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0)
    
    if sys.argv[1] == "--latest":
        # Try to get latest version
        latest = get_latest_onnx_version()
        if latest:
            print(f"Latest ONNX version: {latest}")
            response = input(f"Update to {latest}? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
            version = latest
        else:
            print("Error: Could not fetch latest version")
            sys.exit(1)
    else:
        version = sys.argv[1]
    
    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        print(f"Error: Invalid version format: {version}")
        print("Expected format: X.Y.Z (e.g., 1.21.0)")
        sys.exit(1)
    
    # Find CMakeLists.txt
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cmake_path = project_root / "CMakeLists.txt"
    
    if not cmake_path.exists():
        print(f"Error: CMakeLists.txt not found at {cmake_path}")
        sys.exit(1)
    
    # Update version
    if not update_cmake_lists(version, cmake_path):
        print("No update needed")
        sys.exit(0)
    
    # Clean build directory to force re-download
    build_dir = project_root / "build"
    proto_file = build_dir / "third_party" / "onnx" / "onnx.proto"
    
    if proto_file.exists():
        proto_file.unlink()
        print(f"Removed cached proto file: {proto_file}")
    
    # Optionally rebuild
    response = input("Rebuild now? (y/n): ")
    if response.lower() == 'y':
        if not build_dir.exists():
            build_dir.mkdir()
        
        subprocess.run(["cmake", "..", f"-DONNX_PROTO_VERSION={version}"], 
                      cwd=build_dir, check=True)
        subprocess.run(["make", "-j", "generate_types"], 
                      cwd=build_dir, check=True)
        
        print("\n✅ Types regenerated successfully!")
        print(f"\nCheck the new types:")
        print(f"  grep 'kFloat.*= ' src/core/types.hpp")
    else:
        print("\n✅ Version updated in CMakeLists.txt")
        print(f"\nNext steps:")
        print(f"  rm -rf build && mkdir build && cd build")
        print(f"  cmake .. -DONNX_PROTO_VERSION={version}")
        print(f"  make -j$(nproc)")


if __name__ == "__main__":
    main()
