import gradio as gr
import subprocess


def run_command(command):
    try:
        result = subprocess.check_output(command, shell=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"


iface = gr.Interface(
    fn=run_command,
    inputs="text",
    outputs="text",
    # live=True,
    title="Command Output Viewer",
    description="Enter a command and view its output.",
    examples=[["ls"], ["pwd"], ["echo 'Hello, Gradio!'"], ["python --version"]],
)
iface.launch(server_name="0.0.0.0", server_port=7860)
