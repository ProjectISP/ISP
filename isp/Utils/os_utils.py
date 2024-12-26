import os
import shutil


class OSutils:

    @staticmethod
    def copy_and_rename_file(src_path, dest_dir, new_name):
        """
        Copies a file from the source path to the destination directory
        with a new name.

        :param src_path: Path to the source file
        :param dest_dir: Path to the destination directory
        :param new_name: New name for the copied file
        """
        # Ensure destination directory exists
        os.makedirs(dest_dir, exist_ok=True)

        # Construct the full destination path
        dest_path = os.path.join(dest_dir, new_name)

        # Copy the file to the new location with the new name
        shutil.copy(src_path, dest_path)
        print(f"File copied to {dest_path}")