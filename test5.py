import os
import cv2
cv2.ocl.setUseOpenCL(True)
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import numpy as np
import time


class ParkingDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Spot Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")  # Light gray background


        # Initialize variables
        self.model = None
        self.video_path = ""
        self.processed_frame = None
        self.is_processing = False
        self.photo = None  # Keep a reference to prevent garbage collection
        self.video_capture = None
        self.stop_video = False
        self.paused = False
        self.confidence_threshold = 0.25  # Default confidence threshold
        self.parking_plots = []  # List to store manually added parking plots
       
        # Performance metrics
        self.last_fps = 0
        self.preprocess_time = 0
        self.inference_time = 0
        self.postprocess_time = 0
        self.frame_count = 0
        self.last_update_time = time.time()
       
        # Detection counts
        self.vacant_count = 0
        self.occupied_count = 0
        self.total_count = 0


        # Create UI elements
        self.create_ui()


        # Load the YOLOv8 model
        self.load_model()


    def create_ui(self):
        """Create the user interface with right-side panel."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
       
        # Create horizontal layout with canvas on left and controls on right
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
       
        # Left frame for canvas (larger portion)
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
       
        # Right frame for controls (smaller portion, fixed width)
        right_frame = ttk.Frame(content_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)  # Prevent shrinking
       
        # Style configuration
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10), padding=5)
        style.configure("TLabel", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("Stats.TLabel", font=("Arial", 11), padding=3)
        style.configure("Highlight.TLabel", font=("Arial", 11, "bold"), foreground="blue")
       
        # Image display canvas (in left frame)
        self.canvas = tk.Canvas(left_frame, bg="#111111", highlightthickness=1, highlightbackground="#444444")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
       
        # Results display in a frame above the canvas
        results_frame = ttk.Frame(left_frame, relief=tk.RIDGE, borderwidth=1)
        results_frame.pack(fill=tk.X, pady=(0, 5))
       
        # Style for the results display
        self.results_var = tk.StringVar(value="Vacant: 0/0 spaces")
        results_label = ttk.Label(
            results_frame,
            textvariable=self.results_var,
            font=("Arial", 14, "bold"),
            padding=8
        )
        results_label.pack(expand=True)
       
        # Right sidebar with controls
        ttk.Label(right_frame, text="CONTROLS", style="Header.TLabel").pack(pady=(0, 10), fill=tk.X)
       
        # Control frames with some styling
        controls_frame = ttk.Frame(right_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
       
        # Control buttons frame - vertical layout
        buttons_frame = ttk.LabelFrame(controls_frame, text="Source")
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
       
        # Upload Image button
        ttk.Button(buttons_frame, text="Upload Image", command=self.upload_image).pack(fill=tk.X, padx=5, pady=3)
       
        # Upload Video button
        ttk.Button(buttons_frame, text="Upload Video", command=self.upload_video).pack(fill=tk.X, padx=5, pady=3)
       
        # Start Camera button
        ttk.Button(buttons_frame, text="Start Camera", command=self.start_camera).pack(fill=tk.X, padx=5, pady=3)
       
        # Playback controls frame
        playback_frame = ttk.LabelFrame(controls_frame, text="Playback")
        playback_frame.pack(fill=tk.X, padx=5, pady=5)
       
        # Pause/Resume button
        self.pause_button = ttk.Button(playback_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(fill=tk.X, padx=5, pady=3)
       
        # Stop button
        ttk.Button(playback_frame, text="Stop", command=self.stop_processing).pack(fill=tk.X, padx=5, pady=3)
       
        # Parking Plot controls
        plot_frame = ttk.LabelFrame(controls_frame, text="Parking Plots")
        plot_frame.pack(fill=tk.X, padx=5, pady=5)
       
        # Add Parking Plot button
        ttk.Button(plot_frame, text="Add Parking Plot", command=self.add_parking_plot).pack(fill=tk.X, padx=5, pady=3)
       
        # Remove Parking Plot button
        ttk.Button(plot_frame, text="Remove Parking Plot", command=self.remove_parking_plot).pack(fill=tk.X, padx=5, pady=3)
       
        # Settings frame
        settings_frame = ttk.LabelFrame(controls_frame, text="Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
       
        # Confidence threshold slider
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=3)
       
        ttk.Label(threshold_frame, text="Confidence Threshold:").pack(anchor=tk.W)
       
        slider_frame = ttk.Frame(threshold_frame)
        slider_frame.pack(fill=tk.X, pady=2)
       
        self.confidence_slider = ttk.Scale(
            slider_frame,
            from_=0.1,
            to=0.9,
            value=self.confidence_threshold,
            command=self.update_confidence
        )
        self.confidence_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
       
        self.conf_value_label = ttk.Label(slider_frame, text=f"{self.confidence_threshold:.2f}")
        self.conf_value_label.pack(side=tk.RIGHT, padx=(5, 0))
       
        # Video Processing Information Panel
        info_frame = ttk.LabelFrame(right_frame, text="Processing Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
       
        # Create labels for displaying processing info
        self.info_frame = ttk.Frame(info_frame)
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)
       
        # FPS display
        fps_frame = ttk.Frame(self.info_frame)
        fps_frame.pack(fill=tk.X, pady=2)
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="0.0")
        ttk.Label(fps_frame, textvariable=self.fps_var, style="Stats.TLabel").pack(side=tk.RIGHT)
       
        # Resolution display
        res_frame = ttk.Frame(self.info_frame)
        res_frame.pack(fill=tk.X, pady=2)
        ttk.Label(res_frame, text="Resolution:").pack(side=tk.LEFT)
        self.resolution_var = tk.StringVar(value="N/A")
        ttk.Label(res_frame, textvariable=self.resolution_var, style="Stats.TLabel").pack(side=tk.RIGHT)
       
        # Processing time information
        time_frame = ttk.Frame(self.info_frame)
        time_frame.pack(fill=tk.X, pady=2)
        ttk.Label(time_frame, text="Processing Time:").pack(side=tk.LEFT)
        self.process_time_var = tk.StringVar(value="N/A")
        ttk.Label(time_frame, textvariable=self.process_time_var, style="Stats.TLabel").pack(side=tk.RIGHT)
       
        # Detection counts
        counts_frame = ttk.Frame(self.info_frame)
        counts_frame.pack(fill=tk.X, pady=2)
        ttk.Label(counts_frame, text="Detections:").pack(side=tk.LEFT)
        self.detections_var = tk.StringVar(value="0 vacant, 0 occupied")
        ttk.Label(counts_frame, textvariable=self.detections_var, style="Stats.TLabel").pack(side=tk.RIGHT)
       
        # Parking Space counts - NEW
        spaces_frame = ttk.Frame(self.info_frame)
        spaces_frame.pack(fill=tk.X, pady=2)
        ttk.Label(spaces_frame, text="Parking Spaces:").pack(side=tk.LEFT)
        self.spaces_var = tk.StringVar(value="0/0 vacant")
        ttk.Label(spaces_frame, textvariable=self.spaces_var, style="Highlight.TLabel").pack(side=tk.RIGHT)
       
        # Status bar at the bottom
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=(5, 0))


    def load_model(self):
        """Load the YOLOv8 model."""
        model_path = "runs/detect/parking_model/weights/best.pt"  # Update this path if needed
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return


        try:
            self.model = YOLO(model_path)
            self.status_var.set("Model loaded successfully")
            # Print class names from model to debug
            if hasattr(self.model, 'names'):
                class_names = self.model.names
                print(f"Class names from model: {class_names}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None


    def upload_image(self):
        """Allow the user to upload an image."""
        if self.is_processing:
            messagebox.showinfo("Info", "Processing another file. Please wait.")
            return


        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.video_path = file_path
            self.is_processing = True
            self.stop_video = False
            self.status_var.set(f"Processing image: {os.path.basename(file_path)}")
            threading.Thread(target=self.process_image, daemon=True).start()


    def upload_video(self):
        """Allow the user to upload a video."""
        if self.is_processing:
            messagebox.showinfo("Info", "Processing another file. Please wait.")
            return


        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.video_path = file_path
            self.is_processing = True
            self.stop_video = False
            self.status_var.set(f"Processing video: {os.path.basename(file_path)}")
            threading.Thread(target=self.process_video, daemon=True).start()


    def start_camera(self):
        """Start capturing from the camera."""
        if self.is_processing:
            messagebox.showinfo("Info", "Processing another file. Please wait.")
            return


        self.video_path = 0  # Use 0 for the default camera
        self.is_processing = True
        self.stop_video = False
        self.status_var.set("Processing camera feed")
        threading.Thread(target=self.process_video, daemon=True).start()


    def stop_processing(self):
        """Stop video or camera processing."""
        self.stop_video = True
        if self.video_capture:
            self.video_capture.release()
        self.status_var.set("Processing stopped")


    def toggle_pause(self):
        """Pause or resume video/camera processing."""
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
        self.status_var.set("Paused" if self.paused else "Resumed")


    def add_parking_plot(self):
        """Allow the user to manually add a parking plot."""
        messagebox.showinfo("Info", "Click and drag on the canvas to add a parking plot.")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)


    def on_press(self, event):
        """Handle mouse button press event."""
        self.start_x = event.x
        self.start_y = event.y


    def on_drag(self, event):
        """Handle mouse drag event."""
        self.canvas.delete("temp_rect")
        self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="blue", tags="temp_rect")


    def on_release(self, event):
        """Handle mouse button release event."""
        self.canvas.delete("temp_rect")
        self.parking_plots.append((self.start_x, self.start_y, event.x, event.y))
        self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="blue", tags="parking_plot")


    def remove_parking_plot(self):
        """Allow the user to remove the last added parking plot."""
        if self.parking_plots:
            self.parking_plots.pop()
            self.canvas.delete("parking_plot")


    def update_confidence(self, value):
        """Update the confidence threshold."""
        self.confidence_threshold = float(value)
        self.conf_value_label.config(text=f"{self.confidence_threshold:.2f}")


    def process_image(self):
        """Process the uploaded image using the YOLOv8 model."""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please check the model path.")
            self.is_processing = False
            return


        # Load the image
        image = cv2.imread(self.video_path)
        if image is None:
            messagebox.showerror("Error", "Failed to load the image.")
            self.is_processing = False
            return


        # Resize the image to 640x640
        start_preprocess = time.time()
        resized_image = cv2.resize(image, (640, 640))
        preprocess_time = (time.time() - start_preprocess) * 1000  # ms
       
        # Update resolution info
        self.resolution_var.set(f"{resized_image.shape[1]}x{resized_image.shape[0]}")


        # Perform YOLOv8 inference
        try:
            start_inference = time.time()
            results = self.model(resized_image, conf=self.confidence_threshold, verbose=False)
            inference_time = (time.time() - start_inference) * 1000  # ms
           
            # Update processing time info
            self.preprocess_time = preprocess_time
            self.inference_time = inference_time
            self.process_time_var.set(f"{inference_time:.1f}ms")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.is_processing = False
            return


        # Process results
        start_postprocess = time.time()
        self.process_results(results, resized_image)
        postprocess_time = (time.time() - start_postprocess) * 1000  # ms
       
        # Print terminal output similar to the sample
        shape_info = f"(1, 3, {resized_image.shape[0]}, {resized_image.shape[1]})"
        print(f"0: {resized_image.shape[1]}x{resized_image.shape[0]} {self.vacant_count} empty, {self.occupied_count} occupieds, {inference_time:.1f}ms")
        print(f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, {postprocess_time:.1f}ms postprocess per image at shape {shape_info}")
        print(f"Processed counts: {self.vacant_count} vacant, {self.occupied_count} occupied, {self.total_count} total")
       
        # Update detections info
        self.detections_var.set(f"{self.vacant_count} vacant, {self.occupied_count} occupied")
       
        # Update parking spaces info
        self.spaces_var.set(f"{self.vacant_count}/{self.total_count} vacant")


        # Display the image
        self.root.after(0, self.display_frame)
        self.is_processing = False


    def process_video(self):
        """Process the uploaded video or camera feed using the YOLOv8 model."""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please check the model path.")
            self.is_processing = False
            return


        # Open the video file or camera
        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Failed to open the video or camera.")
            self.is_processing = False
            return


        # Get video properties
        frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
       
        # Update resolution info
        self.resolution_var.set(f"{frame_width}x{frame_height}")
       
        # Reset frame counter and timer
        self.frame_count = 0
        self.last_update_time = time.time()


        while not self.stop_video:
            if not self.paused:
                ret, frame = self.video_capture.read()
                if not ret:
                    print("End of video or failed to read frame.")
                    break


                # Increase frame counter
                self.frame_count += 1
               
                # Start timing for this frame
                start_preprocess = time.time()
               
                # Resize the frame to 640x640
                resized_frame = cv2.resize(frame, (640, 640))
               
                # Calculate preprocess time
                preprocess_time = (time.time() - start_preprocess) * 1000  # in ms


                # Perform YOLOv8 inference
                try:
                    start_inference = time.time()
                    results = self.model(resized_frame, conf=self.confidence_threshold, verbose=False)
                    inference_time = (time.time() - start_inference) * 1000  # in ms
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process frame: {str(e)}")
                    self.is_processing = False
                    return


                # Process results
                start_postprocess = time.time()
                self.process_results(results, resized_frame)
                postprocess_time = (time.time() - start_postprocess) * 1000  # in ms
               
                # Store timing data
                self.preprocess_time = preprocess_time
                self.inference_time = inference_time
                self.postprocess_time = postprocess_time
               
                # Calculate and update FPS every 10 frames
                if self.frame_count % 10 == 0:
                    current_time = time.time()
                    elapsed_time = current_time - self.last_update_time
                    self.last_fps = 10 / elapsed_time if elapsed_time > 0 else 0
                    self.last_update_time = current_time
                   
                    # Update FPS display
                    self.fps_var.set(f"{self.last_fps:.1f}")
                   
                    # Update processing time info
                    total_process_time = preprocess_time + inference_time + postprocess_time
                    self.process_time_var.set(f"{total_process_time:.1f}ms")
                   
                    # Print terminal output similar to the sample
                    shape_info = f"(1, 3, {resized_frame.shape[0]}, {resized_frame.shape[1]})"
                    print(f"\nFrame {self.frame_count}: {resized_frame.shape[1]}x{resized_frame.shape[0]} {self.vacant_count} empty, {self.occupied_count} occupieds, {inference_time:.1f}ms")
                    print(f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, {postprocess_time:.1f}ms postprocess per image at shape {shape_info}")
                    print(f"Processed counts: {self.vacant_count} vacant, {self.occupied_count} occupied, {self.total_count} total")
                   
                    # Update detections info
                    self.detections_var.set(f"{self.vacant_count} vacant, {self.occupied_count} occupied")
                   
                    # Update parking spaces info
                    self.spaces_var.set(f"{self.vacant_count}/{self.total_count} vacant")


                # Display the frame
                self.root.after(0, self.display_frame)


        # Release the video capture object
        self.video_capture.release()
        self.is_processing = False


    def process_results(self, results, frame):
        """Process YOLOv8 results and draw bounding boxes."""
        if len(results) == 0:
            print("No results returned from the model.")
            return


        # Access the first result
        result = results[0]
       
        # Initialize counters for vacant and occupied spots
        vacant_spots = 0
        occupied_spots = 0
       
        # Get class mapping from the model
        class_names = result.names
       
        # Process detections
        boxes = []
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
               
                # Get coordinates
                xyxy = box.xyxy[0].tolist()
               
                # Count based on class ID
                class_name = class_names[cls_id].lower()
               
                # Check different possible class name variations
                if 'empty' in class_name or 'vacant' in class_name or cls_id == 0:
                    vacant_spots += 1
                else:
                    occupied_spots += 1
               
                boxes.append([*xyxy, conf, cls_id])
       
        # Calculate total spots
        total_spots = vacant_spots + occupied_spots
       
        # Store counts for terminal output
        self.vacant_count = vacant_spots
        self.occupied_count = occupied_spots
        self.total_count = total_spots
       
        # Draw bounding boxes and labels on the frame
        for det in boxes:
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
           
            # Determine class name and color
            class_name = class_names[cls_id]
            is_vacant = 'empty' in class_name.lower() or 'vacant' in class_name.lower() or cls_id == 0
            color = (0, 255, 0) if is_vacant else (0, 0, 255)  # Green for vacant, Red for occupied
           
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
           
            # Draw label
            label = f"{'Vacant' if is_vacant else 'Occupied'}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Update the results display
        self.results_var.set(f"Vacant: {vacant_spots}/{total_spots} spaces")
       
        # Update the spaces display in the info panel
        self.spaces_var.set(f"{vacant_spots}/{total_spots} vacant")


        # Convert the frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.processed_frame = Image.fromarray(frame_rgb)


    def display_frame(self):
        """Display the processed frame on the canvas."""
        if self.processed_frame is None:
            return


        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
       
        # Ensure canvas has dimensions (after it's drawn on screen)
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.update_idletasks()  # Force update of geometry
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
           
            # If still no valid dimensions, use default values
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 600
                canvas_height = 400


        # Resize the frame to fit the canvas while maintaining aspect ratio
        img_width, img_height = self.processed_frame.size
        img_ratio = img_width / img_height
        canvas_ratio = canvas_width / canvas_height


        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)


        # Use backward-compatible resampling method
        try:
            # For newer versions of Pillow (>= 9.1.0)
            resized_frame = self.processed_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except AttributeError:
            # For older versions of Pillow (< 9.1.0)
            resized_frame = self.processed_frame.resize((new_width, new_height), Image.ANTIALIAS)
        except Exception:
            # Fallback for any other issues
            resized_frame = self.processed_frame.resize((new_width, new_height))


        self.photo = ImageTk.PhotoImage(resized_frame)


        # Clear the canvas and update with new frame
        self.canvas.delete("all")
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
       
        # Re-draw any parking plots
        for plot in self.parking_plots:
            x1, y1, x2, y2 = plot
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", tags="parking_plot")


def main():
    # Create and start the app
    root = tk.Tk()
    app = ParkingDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
