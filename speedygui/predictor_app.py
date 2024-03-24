import threading
import toga
from toga.style import Pack
from toga.constants import COLUMN, ROW

# Styles for buttons
BUTTON_STYLE = Pack(
    padding=10,
    background_color="#4CAF50",
    color="white",
    font_size=14,
    font_weight="bold",
)

# Styles for the output label
OUTPUT_LABEL_STYLE = Pack(
    padding=10,
    background_color="#F0F0F0",
    font_size=12,
    flex=1,
)


class PredictorApp(toga.App):
    def __init__(self, app_name, formal_name, predictor):
        super().__init__(app_name, formal_name)
        self.predictor = predictor

    def startup(self):
        main_box = toga.Box(style=Pack(direction=COLUMN))

        self.progress_bar = toga.ProgressBar(max=100, value=0, running=False, style=Pack(padding=5))
        main_box.add(self.progress_bar)

        self.output_label = toga.MultilineTextInput(readonly=True, style=OUTPUT_LABEL_STYLE)
        main_box.add(self.output_label)

        button_box = toga.Box(style=Pack(direction=ROW, padding=10))
        self.select_folder_button = toga.Button('Select Folder', on_press=self.select_folder, style=BUTTON_STYLE)
        button_box.add(self.select_folder_button)
        self.run_predictor_button = toga.Button('Run Predictor', on_press=self.run_predictor, enabled=False, style=BUTTON_STYLE)
        button_box.add(self.run_predictor_button)
        main_box.add(button_box)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    async def select_folder(self, widget):
        try:
            self.select_folder_button.enabled = False
            folder_path = await toga.Window.select_folder_dialog(
                self.main_window, multiple_select=False, title="Select a Folder"
            )

            if folder_path is not None:
                self.folder_path = folder_path
                self.output_label.value = f"Selected folder: {self.folder_path}"
                self.run_predictor_button.enabled = True
            else:
                self.output_label.value = "No folder selected"
        except ValueError as e:
            self.output_label.value = str(e)
        finally:
            self.select_folder_button.enabled = True

    def run_predictor(self, widget):
        self.output_label.value = "Running predictor..."
        self.progress_bar.start()
        # Run the prediction in a separate thread
        threading.Thread(target=self.predict_thread).start()

    def predict_thread(self):
        data = {"test": self.predictor.dataset_creator(self.folder_path)}

        # Set the callback function for the progress bar update
        self.predictor.progress_callback = self.update_progress_callback

        # Perform inference
        predictions = self.predictor.predict(data)["test"]

        # Save the predictions
        predicted_masks_dir = self.predictor.save_predictions_fn(self.folder_path, predictions, data["test"])

        # Update the output label on the main UI thread
        self.loop.call_soon(lambda: self.update_output_label(f"Predicted masks saved in {predicted_masks_dir}"))
        self.progress_bar.stop()

    def update_output_label(self, message):
        self.output_label.value = message

    def update_progress_callback(self, progress):
        # Update the progress bar on the main UI thread
        self.loop.call_soon(lambda: self.update_progress_bar(progress))

    def update_progress_bar(self, progress):
        self.progress_bar.value = progress
