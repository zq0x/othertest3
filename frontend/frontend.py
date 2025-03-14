import gradio as gr
import requests
import json
import time
import threading

class GradioFrontend:
    def __init__(self, backend_url="http://backend:8000/process"):
        self.backend_url = backend_url
        self.progress_text = ""
        self.stop_requested = False

    def process_request(self, *args):
        self.progress_text = ""
        self.stop_requested = False

        try:
            payload = {"args": args}
            response = requests.post(self.backend_url, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if self.stop_requested:
                    break  # Stop processing if stop is requested

                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        data = json.loads(decoded_line)
                        if "progress" in data:
                            self.progress_text += data["progress"] + "\n"
                            yield self.progress_text, gr.Button("Stop")
                        elif "result" in data:
                            self.progress_text += "Result: " + data["result"]
                            yield self.progress_text, gr.Button("Run")
                            break
                    except json.JSONDecodeError:
                        self.progress_text += decoded_line + "\n"
                        yield self.progress_text, gr.Button("Stop")

        except requests.exceptions.RequestException as e:
            self.progress_text = f"Error: {e}"
            yield self.progress_text, gr.Button("Run")

    def stop_processing(self):
        self.stop_requested = True
        return self.progress_text, gr.Button("Run")

    def launch_ui(self):
        inputs = [gr.Textbox(label="Input 1", default_value="Hello"),
                  gr.Number(label="Input 2", default_value=123),
                  gr.Checkbox(label="Input 3", default_value=True)]

        with gr.Blocks() as demo:
            gr.Markdown("# Gradio Frontend with FastAPI Backend")
            input_components = []
            for input_item in inputs:
                input_components.append(input_item)

            run_button = gr.Button("Run")
            stop_button = gr.Button("Stop")

            output_text = gr.Textbox(label="Output")

            run_button.click(fn=self.process_request, inputs=input_components, outputs=[output_text, run_button])
            stop_button.click(fn=self.stop_processing, outputs=[output_text, run_button])

        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    frontend = GradioFrontend()
    frontend.launch_ui()