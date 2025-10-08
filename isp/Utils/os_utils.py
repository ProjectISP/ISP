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

    @staticmethod
    def create_symlink(target, link_path, overwrite=False):
        """
        Creates a symbolic link pointing to a target.

        :param target: Path to the existing file or directory
        :param link_path: Path for the symbolic link
        :param overwrite: If True, overwrite existing link/file
        """
        # Resolve absolute paths
        target = os.path.abspath(target)
        link_path = os.path.abspath(link_path)

        # If overwrite is allowed, remove existing file/link first
        if overwrite and os.path.lexists(link_path):
            os.remove(link_path)

        # Avoid trying to link a path to itself
        try:
            if os.path.exists(link_path) and os.path.samefile(target, link_path):
                print(f"Skipping symlink: {link_path} and {target} are the same file.")
                return
        except FileNotFoundError:
            pass  # link_path may not exist yet

        try:
            os.symlink(target, link_path)
            print(f"Symbolic link created: {link_path} -> {target}")
        except FileExistsError:
            print(f"Link already exists: {link_path}")
        except OSError as e:
            print(f"Error creating symlink: {e}")
