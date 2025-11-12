#!/usr/bin/env python3
"""
Compare two directories and copy files from server when Google Drive has 0-byte files.

This script:
1. Scans the Google Drive directory for files with 0 bytes
2. Checks if the same file exists on the Synology server with a larger size
3. Copies the server version to replace the 0-byte file
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple


class ZeroByteFileSyncer:
    def __init__(self, gdrive_path: str, server_path: str):
        """
        Initialize the syncer with source and destination paths.
        
        Args:
            gdrive_path: Path to Google Drive directory
            server_path: Path to Synology server directory
        """
        self.gdrive_path = Path(gdrive_path)
        self.server_path = Path(server_path)
        
        # Validate paths
        if not self.gdrive_path.exists():
            raise ValueError(f"Google Drive path does not exist: {gdrive_path}")
        if not self.server_path.exists():
            raise ValueError(f"Server path does not exist: {server_path}")
    
    def find_zero_byte_files(self) -> List[Path]:
        """
        Find all files with 0 bytes in the Google Drive directory.
        
        Returns:
            List of Path objects for 0-byte files
        """
        zero_byte_files = []
        
        print(f"Scanning {self.gdrive_path} for 0-byte files...")
        
        for root, dirs, files in os.walk(self.gdrive_path):
            for filename in files:
                filepath = Path(root) / filename
                try:
                    if filepath.stat().st_size == 0:
                        zero_byte_files.append(filepath)
                except (OSError, PermissionError) as e:
                    print(f"Warning: Could not access {filepath}: {e}")
        
        return zero_byte_files
    
    def get_relative_path(self, filepath: Path) -> Path:
        """
        Get the relative path from the Google Drive base directory.
        
        Args:
            filepath: Absolute path to a file
            
        Returns:
            Relative path from gdrive_path
        """
        return filepath.relative_to(self.gdrive_path)
    
    def find_matching_server_file(self, relative_path: Path) -> Tuple[bool, int]:
        """
        Check if a file exists on the server with a larger size.
        
        Args:
            relative_path: Relative path of the file
            
        Returns:
            Tuple of (exists, size) where exists is True if file found with size > 0
        """
        server_file = self.server_path / relative_path
        
        if server_file.exists() and server_file.is_file():
            try:
                size = server_file.stat().st_size
                return (size > 0, size)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not access {server_file}: {e}")
                return (False, 0)
        
        return (False, 0)
    
    def copy_file(self, relative_path: Path, dry_run: bool = True) -> bool:
        """
        Copy a file from server to Google Drive.
        
        Args:
            relative_path: Relative path of the file to copy
            dry_run: If True, only simulate the copy
            
        Returns:
            True if copy was successful (or would be in dry run mode)
        """
        source = self.server_path / relative_path
        destination = self.gdrive_path / relative_path
        
        try:
            if dry_run:
                print(f"[DRY RUN] Would copy: {source} -> {destination}")
                return True
            else:
                # Ensure destination directory exists
                destination.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(source, destination)
                print(f"✓ Copied: {relative_path} ({source.stat().st_size} bytes)")
                return True
                
        except (OSError, PermissionError, shutil.Error) as e:
            print(f"✗ Error copying {relative_path}: {e}")
            return False
    
    def sync(self, dry_run: bool = True) -> dict:
        """
        Main sync operation.
        
        Args:
            dry_run: If True, only report what would be done without making changes
            
        Returns:
            Dictionary with sync statistics
        """
        stats = {
            'zero_byte_files': 0,
            'files_to_copy': 0,
            'files_copied': 0,
            'files_failed': 0,
            'total_bytes': 0
        }
        
        print("=" * 80)
        print("Zero-Byte File Sync")
        print("=" * 80)
        print(f"Google Drive: {self.gdrive_path}")
        print(f"Server:       {self.server_path}")
        print(f"Mode:         {'DRY RUN' if dry_run else 'LIVE'}")
        print("=" * 80)
        print()
        
        # Find all 0-byte files
        zero_byte_files = self.find_zero_byte_files()
        stats['zero_byte_files'] = len(zero_byte_files)
        
        print(f"Found {len(zero_byte_files)} zero-byte file(s)\n")
        
        if not zero_byte_files:
            print("No zero-byte files found. Nothing to do.")
            return stats
        
        # Check each file against server
        files_to_process = []
        
        for filepath in zero_byte_files:
            relative_path = self.get_relative_path(filepath)
            exists, size = self.find_matching_server_file(relative_path)
            
            if exists:
                files_to_process.append((relative_path, size))
                stats['files_to_copy'] += 1
                stats['total_bytes'] += size
                print(f"→ {relative_path} (server has {size:,} bytes)")
            else:
                print(f"✗ {relative_path} (not found on server or also 0 bytes)")
        
        print()
        print(f"Files to copy: {stats['files_to_copy']}")
        print(f"Total size: {stats['total_bytes']:,} bytes ({stats['total_bytes'] / 1024 / 1024:.2f} MB)")
        print()
        
        if not files_to_process:
            print("No matching files found on server. Nothing to copy.")
            return stats
        
        # Copy files
        if dry_run:
            print("DRY RUN MODE - No files will be copied")
            print("Run with dry_run=False to perform actual copy")
        else:
            print("Copying files...")
        
        print()
        
        for relative_path, size in files_to_process:
            if self.copy_file(relative_path, dry_run):
                if not dry_run:
                    stats['files_copied'] += 1
            else:
                stats['files_failed'] += 1
        
        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Zero-byte files found:    {stats['zero_byte_files']}")
        print(f"Files available on server: {stats['files_to_copy']}")
        if not dry_run:
            print(f"Files copied successfully: {stats['files_copied']}")
            print(f"Files failed:              {stats['files_failed']}")
        print("=" * 80)
        
        return stats


def main():
    """Main entry point."""
    # Define paths
    gdrive_path = "/Users/heathweaver/Library/CloudStorage/GoogleDrive-heath@careerlearning.com/Shared drives/CareerLearning/Video Downloads"
    server_path = "/Volumes/docker/careerlearning/transcription/videos"
    
    try:
        # Create syncer instance
        syncer = ZeroByteFileSyncer(gdrive_path, server_path)
        
        # First run in dry-run mode
        print("Running in DRY RUN mode first...\n")
        stats = syncer.sync(dry_run=True)
        
        # Ask user if they want to proceed
        if stats['files_to_copy'] > 0:
            print("\n")
            response = input("Do you want to proceed with copying these files? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                print("\nProceeding with actual copy...\n")
                syncer.sync(dry_run=False)
            else:
                print("\nOperation cancelled by user.")
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
