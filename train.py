import os
# OpenMP and threading optimizations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# CUDA memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import libraries after setting environment variables
import torch
import gc
from ultralytics import YOLO

# Force garbage collection
gc.collect()
torch.cuda.empty_cache()

def main():
    try:
        # Enable CUDNN benchmark for potentially faster runtime
        torch.backends.cudnn.benchmark = True
        
        # Disable CUDNN determinism for better performance (if not needed)
        torch.backends.cudnn.deterministic = False
        
        # Check GPU availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Print available GPU memory before loading model
        if torch.cuda.is_available():
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Initial free memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            
        # Load a pre-trained YOLOv8 model
        model = YOLO("yolov8n.pt")
        
        # Validate dataset path
        dataset_path = "D:/VS code/project 1/dataset/data.yaml"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        # Force another garbage collection before training
        gc.collect()
        torch.cuda.empty_cache()
        
        # Train the model with optimized parameters
        results = model.train(
            data=dataset_path,  
            epochs=50,          
            imgsz=320,          # Reduced image size for memory efficiency
            batch=2,            # Very small batch size
            name="parking_model",
            device="0",         # Use GPU 0
            workers=0,          # Disable multiprocessing
            save=True,
            save_period=10,
            cache=False,        # Disable caching to reduce memory
            half=True,          # Use mixed precision (FP16)
            amp=True,           # Enable automatic mixed precision
            patience=0,         # Disable early stopping to save memory
            verbose=True,
            exist_ok=True,
            lr0=0.01,           # Slightly reduced learning rate
            cos_lr=True,        # Use cosine learning rate schedule
            overlap_mask=False, # Reduce memory for instance segmentation if not needed
            optimizer="AdamW",  # Memory-efficient optimizer
            multi_scale=False,  # Disable multi-scale training to save memory
            nbs=64,             # Nominal batch size for gradients 
            close_mosaic=10     # Disable mosaic augmentation in final epochs
        )
        
        # Print training results
        print("Training completed successfully!")
        print(f"Results saved in: {results.save_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        # Print helpful message for CUDA OOM errors
        if "CUDA out of memory" in str(e):
            print("\nCUDA memory error detected. Try further reducing batch size or image size.")
            if torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    finally:
        # Clean up memory at the end
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()