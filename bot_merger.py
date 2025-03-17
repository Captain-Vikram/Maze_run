import os
import numpy as np
from google.colab import drive

# 🚀 Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# 🔹 Directory where bot models are stored
SAVE_FOLDER = "/content/drive/MyDrive/MazeBot_Models"

def load_q_tables(grid_size):
    """Load all Q-tables for a given grid size."""
    q_tables = []
    model_files = sorted([
        f for f in os.listdir(SAVE_FOLDER) if f.startswith(f"bot_{grid_size}x{grid_size}_lvl_")
    ], key=lambda x: int(x.split("_lvl_")[1].split(".")[0]))  # Sort by level

    if not model_files:
        print(f"⚠️ No Q-tables found for {grid_size}x{grid_size}. Skipping.")
        return None, []

    for model in model_files:
        path = os.path.join(SAVE_FOLDER, model)
        q_table = np.load(path)
        q_tables.append(q_table)

    return q_tables, model_files

def merge_q_tables(q_tables):
    """Merge multiple Q-tables by averaging them."""
    return np.mean(q_tables, axis=0)  # Element-wise average

def save_combined_q_table(grid_size, combined_q_table):
    """Save the merged Q-table to Google Drive."""
    save_path = os.path.join(SAVE_FOLDER, f"bot_{grid_size}x{grid_size}_combined.npy")
    np.save(save_path, combined_q_table)
    print(f"✅ Saved combined bot at: {save_path}")
    return save_path

def delete_old_q_tables(model_files):
    """Delete all previous Q-tables after merging."""
    for file in model_files:
        path = os.path.join(SAVE_FOLDER, file)
        os.remove(path)  # Delete the file
        print(f"🗑️ Deleted: {file}")

def process_grid_size(grid_size):
    """Load, merge, save, and delete old models for a grid size."""
    print(f"🔄 Processing {grid_size}x{grid_size}...")
    q_tables, model_files = load_q_tables(grid_size)
    
    if q_tables:
        combined_q_table = merge_q_tables(q_tables)
        save_combined_q_table(grid_size, combined_q_table)
        delete_old_q_tables(model_files)  # Cleanup old files
    else:
        print(f"⚠️ No models found for {grid_size}x{grid_size}, skipping.")


process_grid_size(15)

# 🎉 Done!