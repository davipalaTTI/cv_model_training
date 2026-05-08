import os
import yaml
import logging

# Initialize Logger for this module
logger = logging.getLogger(__name__)


def prepare_output_folders(output_root):
    """
    Creates the standard YOLOv8/v11 directory structure with logging.
    Returns: A dictionary of paths for use by the inference thread.
    """
    try:
        logger.info(f"Preparing dataset directory structure at: {output_root}")

        paths = {
            'train_img': os.path.join(output_root, "train", "images"),
            'train_lbl': os.path.join(output_root, "train", "labels"),
            'val_img': os.path.join(output_root, "valid", "images"),
            'val_lbl': os.path.join(output_root, "valid", "labels"),
        }

        # Create each folder
        for folder_type, path in paths.items():
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Verified folder exists: {path}")

        logger.info("Directory structure verified/created successfully.")
        return paths

    except PermissionError:
        error_msg = f"Permission Denied: Cannot write to {output_root}. Try a different location."
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        logger.error(f"Critical error during folder preparation: {e}")
        raise


def generate_yaml(output_root, class_list):
    """
    Writes the data.yaml file needed for YOLO training.
    class_list: A list of names in order: ['car', 'truck', 'person']
    """
    try:
        if not class_list:
            logger.warning("Attempted to generate YAML with an empty class list.")
            raise ValueError("The Class Configuration table is empty. Please add at least one class.")

        yaml_path = os.path.join(output_root, "data.yaml")
        logger.info(f"Generating YOLO manifest (data.yaml) for {len(class_list)} classes.")

        yaml_data = {
            'path': output_root,  # Absolute path to the dataset root
            'train': 'train/images',
            'val': 'valid/images',
            'nc': len(class_list),
            'names': class_list
        }

        # Write to disk
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        logger.info(f"YAML manifest successfully saved to: {yaml_path}")
        return yaml_path

    except Exception as e:
        logger.error(f"Failed to write data.yaml: {e}")
        # Re-raise so the GUI knows the pipeline failed at the manifest stage
        raise Exception(f"YAML Generation Error: {str(e)}")