import os
import uuid
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, List


class DatasetService:
    def __init__(self):
        self.base_dataset_path = Path("/models/datasets")
        self.base_dataset_path.mkdir(parents=True, exist_ok=True)

    async def prepare_uploaded_dataset(
        self, uploaded_file_path: Path, dataset_info: Dict[str, Any]
    ) -> str:
        """
        Prepare uploaded dataset file for YOLO training

        Args:
            uploaded_file_path: Path to uploaded dataset file
            dataset_info: Dict containing dataset information
                - datasetId: int
                - modelName: str
                - filename: str

        Returns:
            str: Path to the prepared dataset YAML file
        """
        dataset_id = dataset_info.get("datasetId")
        model_name = dataset_info.get("modelName", "unnamed_model")
        filename = dataset_info.get("filename", "dataset.zip")

        print(f"Processing uploaded dataset {dataset_id}: {filename}", flush=True)

        # Create unique dataset directory
        dataset_uuid = uuid.uuid4().hex[:8]
        dataset_dir = self.base_dataset_path / f"dataset_{dataset_id}_{dataset_uuid}"
        dataset_dir.mkdir(exist_ok=True)

        try:
            # Copy uploaded file to dataset directory
            dataset_file = dataset_dir / filename
            shutil.copy2(uploaded_file_path, dataset_file)

            # Extract if it's a compressed file
            extracted_dir = await self._extract_dataset_if_needed(dataset_dir)

            # Validate and organize dataset structure
            organized_dir = await self._organize_dataset_structure(extracted_dir)

            # Detect classes from dataset
            classes = await self._detect_classes(organized_dir)

            # Create YOLO configuration file
            yaml_config_path = await self._create_yolo_config(
                organized_dir, classes, model_name, dataset_id
            )

            print(f"Dataset prepared successfully: {yaml_config_path}", flush=True)
            return str(yaml_config_path)

        except Exception as e:
            # Cleanup on error
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise e
        finally:
            # Clean up uploaded file
            if uploaded_file_path.exists():
                uploaded_file_path.unlink()

    async def _extract_dataset_if_needed(self, dataset_dir: Path) -> Path:
        """Extract dataset if it's in a compressed format"""
        # Look for zip files in the directory
        zip_files = list(dataset_dir.glob("*.zip"))

        if zip_files:
            zip_file = zip_files[0]
            extract_dir = dataset_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)

            print(f"Extracting dataset: {zip_file}", flush=True)

            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # Remove the zip file to save space
            zip_file.unlink()

            return extract_dir

        return dataset_dir

    async def _organize_dataset_structure(self, dataset_dir: Path) -> Path:
        """
        Organize dataset into YOLO format if needed
        Expected structure:
        dataset/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
        """
        # Check if it's already in full YOLO format (with train/val splits)
        if self._is_full_yolo_format(dataset_dir):
            print(
                "Dataset already in full YOLO format with train/val splits", flush=True
            )
            return dataset_dir

        # Check if it's in basic YOLO format (just images/ and labels/ folders)
        if self._is_basic_yolo_format(dataset_dir):
            print("Dataset in basic YOLO format, creating train/val splits", flush=True)
            return await self._create_train_val_split(dataset_dir)

        # Try to organize into YOLO format
        organized_dir = dataset_dir.parent / f"{dataset_dir.name}_organized"
        organized_dir.mkdir(exist_ok=True)

        # Create YOLO directory structure
        (organized_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (organized_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (organized_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (organized_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Try to organize files based on common patterns
        await self._organize_files_to_yolo(dataset_dir, organized_dir)

        print(f"Dataset organized into YOLO format: {organized_dir}", flush=True)
        return organized_dir

    def _is_full_yolo_format(self, dataset_dir: Path) -> bool:
        """Check if dataset is in full YOLO format with train/val splits"""
        required_dirs = [
            dataset_dir / "images" / "train",
            dataset_dir / "images" / "val",
            dataset_dir / "labels" / "train",
            dataset_dir / "labels" / "val",
        ]

        return all(
            dir_path.exists() and dir_path.is_dir() for dir_path in required_dirs
        )

    def _is_basic_yolo_format(self, dataset_dir: Path) -> bool:
        """Check if dataset is in basic YOLO format (images/ and labels/)"""
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"

        return (
            images_dir.exists()
            and images_dir.is_dir()
            and labels_dir.exists()
            and labels_dir.is_dir()
        )

    async def _create_train_val_split(self, dataset_dir: Path) -> Path:
        """Create train/val split from basic YOLO format"""
        split_dir = dataset_dir.parent / f"{dataset_dir.name}_split"
        split_dir.mkdir(exist_ok=True)

        # Create train/val structure
        (split_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (split_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

        # Copy classes.txt if it exists
        classes_files = list(dataset_dir.glob("classes.txt"))
        if classes_files:
            shutil.copy2(classes_files[0], split_dir / "classes.txt")

        # Get all image files
        images_dir = dataset_dir / "images"
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        # Split 80/20
        train_split = int(len(image_files) * 0.8)
        train_images = image_files[:train_split]
        val_images = image_files[train_split:]

        # Copy files to train/val splits
        await self._copy_split_files(dataset_dir, split_dir, train_images, "train")
        await self._copy_split_files(dataset_dir, split_dir, val_images, "val")

        print(
            f"Created train/val split: {len(train_images)} train, "
            f"{len(val_images)} val images",
            flush=True,
        )

        return split_dir

    async def _copy_split_files(
        self, source_dir: Path, target_dir: Path, image_files: List[Path], split: str
    ) -> None:
        """Copy image and label files to train or val split"""
        images_target = target_dir / "images" / split
        labels_target = target_dir / "labels" / split
        labels_source = source_dir / "labels"

        for image_file in image_files:
            # Copy image
            shutil.copy2(image_file, images_target / image_file.name)

            # Copy corresponding label if it exists
            label_name = image_file.stem + ".txt"
            label_file = labels_source / label_name

            if label_file.exists():
                shutil.copy2(label_file, labels_target / label_name)

    async def _organize_files_to_yolo(self, source_dir: Path, target_dir: Path) -> None:
        """Organize files into YOLO format"""
        # Find all image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        label_extensions = [".txt"]

        image_files = []
        label_files = []

        # Recursively find all image and label files
        for ext in image_extensions:
            image_files.extend(source_dir.rglob(f"*{ext}"))
            image_files.extend(source_dir.rglob(f"*{ext.upper()}"))

        for ext in label_extensions:
            label_files.extend(source_dir.rglob(f"*{ext}"))

        print(
            f"Found {len(image_files)} images and {len(label_files)} labels", flush=True
        )

        # Split into train/val (80/20 split)
        train_split = int(len(image_files) * 0.8)
        train_images = image_files[:train_split]
        val_images = image_files[train_split:]

        # Copy images and corresponding labels
        await self._copy_files_with_labels(
            train_images, label_files, target_dir, "train"
        )
        await self._copy_files_with_labels(val_images, label_files, target_dir, "val")

    async def _copy_files_with_labels(
        self,
        image_files: List[Path],
        label_files: List[Path],
        target_dir: Path,
        split: str,
    ) -> None:
        """Copy image files and their corresponding label files"""
        images_dir = target_dir / "images" / split
        labels_dir = target_dir / "labels" / split

        for image_file in image_files:
            # Copy image
            target_image = images_dir / image_file.name
            shutil.copy2(image_file, target_image)

            # Find corresponding label file
            label_name = image_file.stem + ".txt"
            corresponding_labels = [lf for lf in label_files if lf.name == label_name]

            if corresponding_labels:
                label_file = corresponding_labels[0]
                target_label = labels_dir / label_file.name
                shutil.copy2(label_file, target_label)

    async def _detect_classes(self, dataset_dir: Path) -> Dict[int, str]:
        """Detect classes from the dataset labels or classes.txt file"""
        classes = {}

        # Look for classes.txt file first (most common and reliable)
        class_files = (
            list(dataset_dir.glob("classes.txt"))
            + list(dataset_dir.glob("**/classes.txt"))
            + list(dataset_dir.glob("names.txt"))
            + list(dataset_dir.glob("**/names.txt"))
            + list(dataset_dir.glob("class_names.txt"))
            + list(dataset_dir.glob("**/class_names.txt"))
        )

        if class_files:
            classes_file = class_files[0]  # Use the first one found
            print(f"Found classes file: {classes_file}", flush=True)

            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        class_name = line.strip()
                        if class_name:  # Skip empty lines
                            classes[idx] = class_name

                print(f"Loaded {len(classes)} classes from file", flush=True)
                return classes

            except Exception as e:
                print(
                    f"Error reading classes file: {e}. "
                    f"Falling back to label detection",
                    flush=True,
                )

        # Fallback: extract classes from label files if no classes.txt
        print("No classes file found, detecting from labels", flush=True)
        label_files = list(dataset_dir.rglob("labels/**/*.txt"))
        class_ids = set()

        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_ids.add(class_id)
            except (ValueError, IndexError, UnicodeDecodeError):
                continue

        # Create generic class names
        for class_id in sorted(class_ids):
            classes[class_id] = f"class_{class_id}"

        print(f"Detected {len(classes)} classes from labels", flush=True)
        return classes

    async def _create_yolo_config(
        self,
        dataset_dir: Path,
        classes: Dict[int, str],
        model_name: str,
        dataset_id: int,
    ) -> str:
        """Create YOLO configuration YAML file"""
        # Create config in the temp_configs directory
        config_dir = Path("/models/temp_configs")
        config_dir.mkdir(exist_ok=True)

        config_uuid = uuid.uuid4().hex[:8]
        config_file = (
            config_dir / f"{model_name}_dataset_{dataset_id}_{config_uuid}.yaml"
        )

        # YOLO config structure with correct format
        config_data = {
            "path": str(dataset_dir),
            "train": "images",  # Simplified path - YOLO will look for train/val subdirs
            "val": "images",  # Simplified path - YOLO will look for train/val subdirs
            "test": None,  # Optional test set (commented out as per example)
            "names": classes,
        }

        # Write YAML configuration with proper formatting
        with open(config_file, "w") as f:
            f.write(f"path: {dataset_dir}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("# test:\n\n")
            f.write("# Classes\n")
            f.write("names:\n")
            for class_id, class_name in classes.items():
                f.write(f"  {class_id}: {class_name}\n")

        print(f"YOLO config created: {config_file}", flush=True)
        return str(config_file)

    def cleanup_dataset(self, yaml_config_path: str) -> None:
        """Clean up temporary dataset files and YAML config"""
        try:
            # Clean up YAML config file
            config_file = Path(yaml_config_path)
            if config_file.exists():
                config_file.unlink()
                print(f"Cleaned up YAML config: {config_file}", flush=True)

            # Extract dataset directory from YAML file content
            dataset_dir = None
            try:
                if config_file.exists():
                    with open(config_file, "r") as f:
                        content = f.read()
                        for line in content.split("\n"):
                            if line.startswith("path:"):
                                dataset_dir = Path(line.split("path:")[1].strip())
                                break
                else:
                    # Fallback: try to extract from filename pattern
                    config_name = config_file.name
                    parts = config_name.split("_")
                    if len(parts) >= 3 and parts[1] == "dataset":
                        dataset_id = parts[2]
                        # Find matching dataset directory
                        datasets_base = Path("/models/datasets")
                        for dir_path in datasets_base.glob(f"dataset_{dataset_id}_*"):
                            if dir_path.is_dir():
                                dataset_dir = dir_path
                                break
            except Exception as parse_error:
                print(
                    f"Warning: Could not parse dataset path from YAML: {parse_error}",
                    flush=True,
                )

            # Clean up dataset directory (both extracted and zip files)
            if dataset_dir and dataset_dir.exists():
                datasets_base_str = "/models/datasets/"
                if datasets_base_str in str(dataset_dir):
                    try:
                        # Remove the entire dataset directory (includes extracted folders and zip)
                        shutil.rmtree(dataset_dir)
                        print(
                            f"Cleaned up dataset directory: {dataset_dir}", flush=True
                        )

                        # Also check for any related temp files
                        dataset_parent = dataset_dir.parent
                        if dataset_parent.name == "datasets":
                            dataset_pattern = dataset_dir.name.split("_")[
                                :-1
                            ]  # Remove UUID part
                            for related_dir in dataset_parent.glob(
                                "_".join(dataset_pattern) + "_*"
                            ):
                                if related_dir != dataset_dir and related_dir.is_dir():
                                    try:
                                        shutil.rmtree(related_dir)
                                        print(
                                            f"Cleaned up related dataset: {related_dir}",
                                            flush=True,
                                        )
                                    except (FileNotFoundError, PermissionError) as e:
                                        print(
                                            f"Warning: Could not clean up {related_dir}: {e}",
                                            flush=True,
                                        )
                    except (FileNotFoundError, PermissionError) as cleanup_error:
                        print(
                            f"Warning: Could not clean up dataset directory {dataset_dir}: {cleanup_error}",
                            flush=True,
                        )

        except Exception as e:
            # Don't fail on cleanup errors, especially during cancellation
            print(
                f"Note: Cleanup encountered issues (normal during cancellation): {str(e)}",
                flush=True,
            )

    def cleanup_training_artifacts(
        self, model_name: str, dataset_id: int = None
    ) -> None:
        """Clean up all training artifacts for a specific model"""
        try:
            # Clean up YAML config files
            config_dir = Path("/models/temp_configs")
            if config_dir.exists():
                pattern = f"{model_name}_*"
                if dataset_id:
                    pattern = f"{model_name}_dataset_{dataset_id}_*"

                for config_file in config_dir.glob(pattern + ".yaml"):
                    try:
                        config_file.unlink()
                        print(f"Cleaned up config file: {config_file}", flush=True)
                    except (FileNotFoundError, PermissionError) as e:
                        print(
                            f"Warning: Could not clean up config {config_file}: {e}",
                            flush=True,
                        )

            # Clean up dataset directories if dataset_id is provided
            if dataset_id:
                datasets_dir = Path("/models/datasets")
                if datasets_dir.exists():
                    for dataset_dir in datasets_dir.glob(f"dataset_{dataset_id}_*"):
                        if dataset_dir.is_dir():
                            try:
                                shutil.rmtree(dataset_dir)
                                print(f"Cleaned up dataset: {dataset_dir}", flush=True)
                            except (FileNotFoundError, PermissionError) as e:
                                print(
                                    f"Warning: Could not clean up dataset {dataset_dir}: {e}",
                                    flush=True,
                                )

        except Exception as e:
            print(
                f"Note: Training artifact cleanup encountered issues: {str(e)}",
                flush=True,
            )

    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """List all downloaded datasets"""
        datasets = []

        if self.base_dataset_path.exists():
            for dataset_dir in self.base_dataset_path.iterdir():
                if dataset_dir.is_dir():
                    # Try to extract info from directory name
                    parts = dataset_dir.name.split("_")
                    if len(parts) >= 3:
                        dataset_size = sum(
                            f.stat().st_size
                            for f in dataset_dir.rglob("*")
                            if f.is_file()
                        )
                        datasets.append(
                            {
                                "path": str(dataset_dir),
                                "name": dataset_dir.name,
                                "size": dataset_size,
                            }
                        )

        return datasets
