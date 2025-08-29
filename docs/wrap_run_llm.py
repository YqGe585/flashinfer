import os

"""
HTML post-processing script to insert RunLLM widget into documentation.
Based on: https://github.com/sgl-project/sglang/blob/499f5e620c243b6a9980b63f7aa54d096a9a3ddd/docs/wrap_run_llm.py
Copyright (c) 2023 SGLang Project (Apache 2.0 License)
"""
import re


def insert_runllm_widget(html_content):
    widget_script = """
    <!-- RunLLM Widget Script -->
    <script type="module" id="runllm-widget-script" src="https://widget.runllm.com" crossorigin="true" version="stable" runllm-keyboard-shortcut="Mod+j" runllm-name="FlashInfer Assistant" runllm-position="BOTTOM_RIGHT" runllm-assistant-id="1052" async></script>
    """
    return re.sub("</body>", f"{widget_script}\n</body>", html_content)


def process_html_files(build_dir):
    processed_count = 0
    for root, _dirs, files in os.walk(build_dir):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                modified_content = insert_runllm_widget(content)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)
                processed_count += 1
    print(f"Processed {processed_count} HTML files")


def main():
    build_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_build", "html"
    )
    if os.path.exists(build_dir):
        print(f"Processing HTML files in: {build_dir}")
        process_html_files(build_dir)
    else:
        print(f"Build directory not found: {build_dir}")
        print("Please build the documentation first with 'make html'")


if __name__ == "__main__":
    main()
