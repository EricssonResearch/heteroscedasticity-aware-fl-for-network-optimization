#!/usr/bin/env python

import argparse
import subprocess
import shutil
import sys
from pathlib import Path

def main(appname, src=None):
    imagename = appname.replace('-', ':')
    project_root = Path(__file__).resolve().parents[2]
    docker_path = project_root / f"k8s/docker/{appname}.Dockerfile"

    # Check if the Dockerfile exists
    if not docker_path.is_file():
        print(f"Error: Dockerfile not found\n{docker_path}")
        sys.exit(1)

    # If src is provided, copy files and set temp_app_dir
    if src:
        src_path = project_root / src
        temp_app_dir = project_root / "k8s/docker/app"
        temp_src_dir = temp_app_dir / src_path.name

        # Check if src directory exists
        if not src_path.exists():
            print(f"Error: src directory not found\n{src_path}")
            sys.exit(1)

        # Create temporary app directory and copy src files
        temp_src_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_path, temp_src_dir, dirs_exist_ok=True)

    try:
        # Build the Docker image
        docker_image = f"registry.ailab.rnd.ki.sw.ericsson.se/fair-ai/main/fair-fl/{imagename}"
        subprocess.run(["docker", "build", "-t", docker_image, "-f", str(docker_path), str(docker_path.parent)], check=True)

        # Push the Docker image
        subprocess.run(["docker", "push", docker_image], check=True)
        print("Docker image updated and pushed successfully")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

    finally:
        # Remove the whole temporary app directory only if src was provided
        if src:
            shutil.rmtree(temp_app_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push Docker image")
    parser.add_argument("appname", help="Name of the application")
    parser.add_argument("--src", type=str, help="Source directory to copy files from")
    args = parser.parse_args()

    main(args.appname, args.src)
