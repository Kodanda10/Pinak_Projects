"""
File Deletion Quarantine System for Pinak.

Following the strict no-file-deletion policy, this system archives files instead of deleting them.
All file operations that would normally delete files are redirected to quarantine/archive.

Key Features:
- Archive files with metadata (timestamp, reason, user)
- Maintain complete audit trail
- Automatic cleanup policies
- Restore functionality
- Integration with Git for tracking
"""

import hashlib
import logging
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class QuarantineAction(Enum):
    """Types of quarantine actions."""

    DELETE_REQUESTED = "delete_requested"
    MOVE_REQUESTED = "move_requested"
    OVERWRITE_REQUESTED = "overwrite_requested"
    CLEANUP_REQUESTED = "cleanup_requested"
    AUTO_ARCHIVE = "auto_archive"


class QuarantinePriority(Enum):
    """Priority levels for quarantine operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QuarantineRecord:
    """Record of a quarantined file operation."""

    original_path: str
    quarantine_path: str
    action: QuarantineAction
    reason: str
    user: str
    timestamp: datetime.datetime
    file_hash: str
    file_size: int
    priority: QuarantinePriority
    metadata: Dict[str, Any]
    can_restore: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for JSON serialization."""
        data = asdict(self)
        data["action"] = self.action.value
        data["priority"] = self.priority.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuarantineRecord":
        """Create record from dictionary."""
        data["action"] = QuarantineAction(data["action"])
        data["priority"] = QuarantinePriority(data["priority"])
        data["timestamp"] = datetime.datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class FileQuarantineManager:
    """
    Manages file quarantine operations.

    This class provides a safe way to "delete" files by moving them to a quarantine
    directory with full audit trail and restoration capabilities.
    """

    def __init__(
        self,
        quarantine_base: Optional[Union[str, Path]] = None,
        auto_cleanup_days: int = 90,
        max_quarantine_size_gb: float = 10.0,
    ):
        """
        Initialize the quarantine manager.

        Args:
            quarantine_base: Base directory for quarantine storage
            auto_cleanup_days: Days after which to auto-cleanup low priority items
            max_quarantine_size_gb: Maximum size of quarantine directory
        """
        if quarantine_base is None:
            # Default to .quarantine in project root
            project_root = self._find_project_root()
            quarantine_base = project_root / ".quarantine"

        self.quarantine_base = Path(quarantine_base)
        self.auto_cleanup_days = auto_cleanup_days
        self.max_quarantine_size_gb = max_quarantine_size_gb

        # Create quarantine structure
        self._setup_quarantine_structure()

        # Load existing records
        self.records_file = self.quarantine_base / "records.json"
        self.records = self._load_records()

        logger.info(f"FileQuarantineManager initialized at {self.quarantine_base}")

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current = Path.cwd()
        # Look for common project markers
        markers = ["setup.py", "pyproject.toml", ".git", "README.md"]

        while current != current.parent:
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent

        return Path.cwd()  # Fallback to current directory

    def _setup_quarantine_structure(self):
        """Set up the quarantine directory structure."""
        # Create main quarantine directory
        self.quarantine_base.mkdir(parents=True, exist_ok=True)

        # Create subdirectories by priority
        for priority in QuarantinePriority:
            (self.quarantine_base / priority.value).mkdir(exist_ok=True)

        # Create metadata directory
        (self.quarantine_base / "metadata").mkdir(exist_ok=True)

        # Create gitkeep files to ensure directories are tracked
        for subdir in self.quarantine_base.rglob("*"):
            if subdir.is_dir():
                (subdir / ".gitkeep").touch(exist_ok=True)

    def _load_records(self) -> Dict[str, QuarantineRecord]:
        """Load quarantine records from disk."""
        if not self.records_file.exists():
            return {}

        try:
            with open(self.records_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    record_id: QuarantineRecord.from_dict(record_data)
                    for record_id, record_data in data.items()
                }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load quarantine records: {e}")
            return {}

    def _save_records(self):
        """Save quarantine records to disk."""
        data = {
            record_id: record.to_dict() for record_id, record in self.records.items()
        }

        # Write to temporary file first for atomicity
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.records_file.parent, delete=False, suffix=".tmp"
        ) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic move
        temp_file = Path(f.name)
        temp_file.replace(self.records_file)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        if not file_path.exists() or not file_path.is_file():
            return ""

        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except (OSError, IOError):
            return ""

    def _get_quarantine_path(
        self, original_path: Path, priority: QuarantinePriority
    ) -> Path:
        """Generate quarantine path for a file."""
        # Create a unique name based on original path and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path_hash = hashlib.md5(str(original_path).encode()).hexdigest()[:8]

        # Sanitize original filename
        safe_name = "".join(c for c in original_path.name if c.isalnum() or c in "._-")
        if not safe_name:
            safe_name = "unnamed_file"

        quarantine_name = f"{timestamp}_{path_hash}_{safe_name}"
        return self.quarantine_base / priority.value / quarantine_name

    def quarantine_file(
        self,
        file_path: Union[str, Path],
        action: QuarantineAction,
        reason: str,
        user: str = "system",
        priority: QuarantinePriority = QuarantinePriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        preserve_git_history: bool = True,
    ) -> Optional[str]:
        """
        Quarantine a file instead of deleting it.

        Args:
            file_path: Path to the file to quarantine
            action: Type of action being quarantined
            reason: Reason for quarantine
            user: User performing the action
            priority: Priority level for the quarantine
            metadata: Additional metadata
            preserve_git_history: Whether to preserve git history info

        Returns:
            Quarantine record ID if successful, None if failed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return None

        try:
            # Calculate file hash and size
            file_hash = self._calculate_file_hash(file_path)
            file_size = file_path.stat().st_size

            # Generate quarantine path
            quarantine_path = self._get_quarantine_path(file_path, priority)

            # Move file to quarantine
            shutil.move(str(file_path), str(quarantine_path))

            # Create quarantine record
            record = QuarantineRecord(
                original_path=str(file_path),
                quarantine_path=str(quarantine_path),
                action=action,
                reason=reason,
                user=user,
                timestamp=datetime.datetime.now(),
                file_hash=file_hash,
                file_size=file_size,
                priority=priority,
                metadata=metadata or {},
                can_restore=True,
            )

            # Add git information if available
            if preserve_git_history:
                record.metadata.update(self._get_git_info(file_path))

            # Generate record ID
            record_id = hashlib.md5(
                f"{record.original_path}_{record.timestamp.isoformat()}".encode()
            ).hexdigest()

            # Store record
            self.records[record_id] = record
            self._save_records()

            # Check if we need to cleanup old files
            self._check_auto_cleanup()

            logger.info(
                f"File quarantined: {file_path} -> {quarantine_path} (ID: {record_id})"
            )
            return record_id

        except Exception as e:
            logger.error(f"Failed to quarantine file {file_path}: {e}")
            return None

    def restore_file(
        self, record_id: str, target_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Restore a quarantined file.

        Args:
            record_id: ID of the quarantine record
            target_path: Path to restore to (defaults to original path)

        Returns:
            True if successful, False otherwise
        """
        if record_id not in self.records:
            logger.warning(f"Quarantine record not found: {record_id}")
            return False

        record = self.records[record_id]

        if not record.can_restore:
            logger.warning(f"File cannot be restored: {record_id}")
            return False

        try:
            quarantine_path = Path(record.quarantine_path)
            if not quarantine_path.exists():
                logger.error(f"Quarantined file not found: {quarantine_path}")
                return False

            # Determine target path
            if target_path is None:
                target_path = Path(record.original_path)
            else:
                target_path = Path(target_path)

            # Create target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # Move file back
            shutil.move(str(quarantine_path), str(target_path))

            # Update record
            record.can_restore = False
            record.metadata["restored_at"] = datetime.datetime.now().isoformat()
            record.metadata["restored_to"] = str(target_path)
            self._save_records()

            logger.info(f"File restored: {quarantine_path} -> {target_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore file {record_id}: {e}")
            return False

    def list_quarantined_files(
        self,
        action_filter: Optional[QuarantineAction] = None,
        priority_filter: Optional[QuarantinePriority] = None,
        user_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List quarantined files with optional filtering.

        Returns:
            List of quarantine records as dictionaries
        """
        records = []

        for record_id, record in self.records.items():
            # Apply filters
            if action_filter and record.action != action_filter:
                continue
            if priority_filter and record.priority != priority_filter:
                continue
            if user_filter and record.user != user_filter:
                continue

            record_dict = record.to_dict()
            record_dict["record_id"] = record_id
            records.append(record_dict)

            if len(records) >= limit:
                break

        return records

    def cleanup_old_files(
        self,
        max_age_days: Optional[int] = None,
        priority_filter: Optional[QuarantinePriority] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Clean up old quarantined files.

        Args:
            max_age_days: Maximum age in days (defaults to auto_cleanup_days)
            priority_filter: Only cleanup specific priority
            dry_run: If True, only report what would be cleaned

        Returns:
            Cleanup statistics
        """
        if max_age_days is None:
            max_age_days = self.auto_cleanup_days

        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)

        stats = {
            "scanned": 0,
            "to_cleanup": 0,
            "cleaned_up": 0,
            "total_size_freed": 0,
            "errors": 0,
        }

        for record_id, record in list(self.records.items()):
            stats["scanned"] += 1

            # Apply filters
            if record.timestamp > cutoff_date:
                continue
            if priority_filter and record.priority != priority_filter:
                continue
            if not record.can_restore:  # Already processed
                continue

            stats["to_cleanup"] += 1

            if dry_run:
                continue

            try:
                quarantine_path = Path(record.quarantine_path)
                if quarantine_path.exists():
                    size = quarantine_path.stat().st_size
                    quarantine_path.unlink()
                    stats["total_size_freed"] += size
                    stats["cleaned_up"] += 1

                # Remove record
                del self.records[record_id]

            except Exception as e:
                logger.error(f"Failed to cleanup {record_id}: {e}")
                stats["errors"] += 1

        if not dry_run:
            self._save_records()

        return stats

    def _check_auto_cleanup(self):
        """Check if auto cleanup is needed."""
        total_size = sum(
            Path(record.quarantine_path).stat().st_size
            for record in self.records.values()
            if Path(record.quarantine_path).exists()
        )

        total_size_gb = total_size / (1024**3)

        if total_size_gb > self.max_quarantine_size_gb:
            logger.info(
                f"Quarantine size ({total_size_gb:.2f}GB) exceeds limit. Running cleanup..."
            )
            stats = self.cleanup_old_files(priority_filter=QuarantinePriority.LOW)
            logger.info(f"Auto cleanup completed: {stats}")

    def _get_git_info(self, file_path: Path) -> Dict[str, Any]:
        """Get git information for a file."""
        git_info = {}

        try:

            # Get last commit info for the file
            result = subprocess.run(
                [
                    "git",
                    "log",
                    "-1",
                    "--pretty=format:%H,%an,%ae,%ad",
                    "--date=iso",
                    "--",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                cwd=file_path.parent if file_path.parent.exists() else None,
            )

            if result.returncode == 0 and result.stdout.strip():
                commit_hash, author_name, author_email, date = (
                    result.stdout.strip().split(",")
                )
                git_info.update(
                    {
                        "last_commit_hash": commit_hash,
                        "last_author": author_name,
                        "last_author_email": author_email,
                        "last_modified": date,
                    }
                )

        except Exception as e:
            logger.debug(f"Could not get git info for {file_path}: {e}")

        return git_info

    def get_quarantine_stats(self) -> Dict[str, Any]:
        """Get comprehensive quarantine statistics."""
        total_size = 0
        priority_counts = {p.value: 0 for p in QuarantinePriority}
        action_counts = {a.value: 0 for a in QuarantineAction}
        age_distribution = {
            "1_day": 0,
            "7_days": 0,
            "30_days": 0,
            "90_days": 0,
            "older": 0,
        }

        now = datetime.datetime.now()

        for record in self.records.values():
            # Size
            try:
                if Path(record.quarantine_path).exists():
                    total_size += Path(record.quarantine_path).stat().st_size
            except OSError:
                pass

            # Counts
            priority_counts[record.priority.value] += 1
            action_counts[record.action.value] += 1

            # Age distribution
            age_days = (now - record.timestamp).days
            if age_days <= 1:
                age_distribution["1_day"] += 1
            elif age_days <= 7:
                age_distribution["7_days"] += 1
            elif age_days <= 30:
                age_distribution["30_days"] += 1
            elif age_days <= 90:
                age_distribution["90_days"] += 1
            else:
                age_distribution["older"] += 1

        return {
            "total_files": len(self.records),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "priority_distribution": priority_counts,
            "action_distribution": action_counts,
            "age_distribution": age_distribution,
            "auto_cleanup_days": self.auto_cleanup_days,
            "max_size_gb": self.max_quarantine_size_gb,
        }


# Global quarantine manager instance
_quarantine_manager = None


def get_quarantine_manager() -> FileQuarantineManager:
    """Get the global quarantine manager instance."""
    global _quarantine_manager
    if _quarantine_manager is None:
        _quarantine_manager = FileQuarantineManager()
    return _quarantine_manager


def quarantine_file(*args, **kwargs) -> Optional[str]:
    """Convenience function to quarantine a file."""
    return get_quarantine_manager().quarantine_file(*args, **kwargs)


def safe_delete(file_path: Union[str, Path], reason: str, user: str = "system") -> bool:
    """
    Safe delete function that quarantines instead of deleting.

    This function should be used instead of os.remove() or Path.unlink().
    """
    return (
        quarantine_file(
            file_path=file_path,
            action=QuarantineAction.DELETE_REQUESTED,
            reason=reason,
            user=user,
            priority=QuarantinePriority.MEDIUM,
        )
        is not None
    )


def safe_move(
    src: Union[str, Path], dst: Union[str, Path], reason: str, user: str = "system"
) -> bool:
    """
    Safe move function that quarantines the source file.

    This should be used when a file would be overwritten or removed during a move.
    """
    # First quarantine the destination if it exists
    dst_path = Path(dst)
    if dst_path.exists():
        quarantine_file(
            file_path=dst,
            action=QuarantineAction.OVERWRITE_REQUESTED,
            reason=f"Overwritten by move from {src}: {reason}",
            user=user,
            priority=QuarantinePriority.HIGH,
        )

    # Then perform the move
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        logger.error(f"Failed to move {src} to {dst}: {e}")
        return False


# Monkey patch common deletion functions for safety
def patch_file_operations():
    """Patch common file operations to use quarantine system."""

    # Store original functions
    original_remove = os.remove
    original_unlink = os.unlink if hasattr(os, "unlink") else None
    original_rmdir = os.rmdir

    def safe_os_remove(path):
        """Safe os.remove that quarantines instead of deleting."""
        if Path(path).exists():
            safe_delete(path, "os.remove() called", "system")
        else:
            original_remove(path)

    def safe_os_unlink(path):
        """Safe os.unlink that quarantines instead of deleting."""
        if Path(path).exists():
            safe_delete(path, "os.unlink() called", "system")
        else:
            original_unlink(path)

    def safe_os_rmdir(path):
        """Safe os.rmdir that quarantines directory contents."""
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_dir():
            # Quarantine all files in directory first
            for file_path in path_obj.rglob("*"):
                if file_path.is_file():
                    safe_delete(file_path, f"Directory removal: {path}", "system")

        original_rmdir(path)

    # Apply patches
    os.remove = safe_os_remove
    if original_unlink:
        os.unlink = safe_os_unlink
    os.rmdir = safe_os_rmdir

    logger.info("File operations patched for quarantine safety")


# Auto-patch on import if enabled
if os.getenv("PINAK_SAFE_FILE_OPS", "true").lower() == "true":
    patch_file_operations()
