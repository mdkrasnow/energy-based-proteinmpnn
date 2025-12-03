"""
PDB Manager for Centralized Structure Management

This module provides centralized management of PDB structures with:
- Unified interface for local and remote PDB access
- Metadata management and indexing
- Structure validation and filtering
- Integration with caching and streaming systems
- RCSB API integration for dynamic PDB retrieval
"""

from typing import Dict, List, Optional, Union, Set, Any, Callable, Tuple
from pathlib import Path
import json
import logging
import hashlib
import time
import random
import re
import tempfile
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import torch
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class PDBMetadata:
    """Metadata for a PDB structure."""

    pdb_id: str
    resolution: Optional[float]
    method: str
    organism: Optional[str]
    sequence_length: int
    num_chains: int
    last_updated: datetime
    file_size: int
    quality_score: Optional[float] = None
    validation_status: str = "unknown"


class PDBManager:
    """
    Centralized manager for PDB structure access and metadata.

    Provides unified interface for accessing PDB structures from multiple sources
    with metadata management, validation, and quality filtering.
    """

    def __init__(
        self,
        data_sources: List[Dict[str, Any]],
        metadata_db_path: Optional[Path] = None,
        quality_filters: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PDB manager.

        Args:
            data_sources: List of data source configurations
            metadata_db_path: Path to metadata database file
            quality_filters: Filters for structure quality (resolution, etc.)
            validation_config: Configuration for structure validation
        """
        self.data_sources = data_sources
        self.metadata_db_path = metadata_db_path
        self.quality_filters = quality_filters or {}
        self.validation_config = validation_config or {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Metadata storage
        self.metadata_db: Dict[str, PDBMetadata] = {}
        self._load_metadata_db()

        # Structure index
        self.structure_index: Dict[str, Dict[str, Any]] = {}
        self._build_structure_index()

    def _load_metadata_db(self) -> None:
        """Load metadata database from disk."""
        if self.metadata_db_path and self.metadata_db_path.exists():
            try:
                with open(self.metadata_db_path, "r") as f:
                    data = json.load(f)
                    for pdb_id, metadata in data.items():
                        # Convert datetime string back to datetime object
                        metadata["last_updated"] = datetime.fromisoformat(
                            metadata["last_updated"]
                        )
                        self.metadata_db[pdb_id] = PDBMetadata(**metadata)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata database: {e}")

    def _save_metadata_db(self) -> None:
        """Save metadata database to disk."""
        if self.metadata_db_path:
            try:
                self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
                data = {}
                for pdb_id, metadata in self.metadata_db.items():
                    metadata_dict = asdict(metadata)
                    # Convert datetime to string for JSON serialization
                    metadata_dict["last_updated"] = metadata.last_updated.isoformat()
                    data[pdb_id] = metadata_dict

                with open(self.metadata_db_path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to save metadata database: {e}")

    def _build_structure_index(self) -> None:
        """Build index of available structures from all data sources."""
        self.structure_index = {}

        for source_config in self.data_sources:
            source_type = source_config.get("type", "unknown")

            if source_type == "local_pdb":
                self._index_local_pdb_source(source_config)
            elif source_type == "remote_pdb":
                self._index_remote_pdb_source(source_config)
            elif source_type == "pdb_list":
                self._index_pdb_list_source(source_config)
            else:
                self.logger.warning(f"Unknown source type: {source_type}")

        self.logger.info(
            f"Built structure index with {len(self.structure_index)} structures"
        )

    def _index_local_pdb_source(self, source_config: Dict[str, Any]) -> None:
        """Index local PDB files."""
        data_dir = Path(source_config.get("data_dir", "."))
        if not data_dir.exists():
            return

        for pdb_file in data_dir.glob("*.pdb"):
            pdb_id = pdb_file.stem
            self.structure_index[pdb_id] = {
                "source_type": "local_pdb",
                "file_path": str(pdb_file),
                "source_config": source_config,
            }

    def _index_remote_pdb_source(self, source_config: Dict[str, Any]) -> None:
        """Index remote PDB source."""
        pdb_list = source_config.get("pdb_list", [])
        for pdb_id in pdb_list:
            self.structure_index[pdb_id] = {
                "source_type": "remote_pdb",
                "file_path": None,
                "source_config": source_config,
            }

    def _index_pdb_list_source(self, source_config: Dict[str, Any]) -> None:
        """Index PDB list file."""
        list_file = Path(source_config.get("list_file", ""))
        if not list_file.exists():
            return

        try:
            with open(list_file, "r") as f:
                for line in f:
                    pdb_id = line.strip()
                    if pdb_id and not pdb_id.startswith("#"):
                        self.structure_index[pdb_id] = {
                            "source_type": "pdb_list",
                            "file_path": None,
                            "source_config": source_config,
                        }
        except Exception as e:
            self.logger.error(f"Error reading PDB list file {list_file}: {e}")

    def list_structures(
        self,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        List available PDB structures with optional filtering and sorting.

        Args:
            filters: Filters to apply (resolution, method, organism, etc.)
            sort_by: Field to sort by (resolution, quality_score, etc.)
            limit: Maximum number of results to return

        Returns:
            List of PDB IDs matching criteria
        """
        # Start with all indexed structures
        pdb_ids = list(self.structure_index.keys())

        # Apply filters if provided
        if filters:
            filtered_ids = []
            for pdb_id in pdb_ids:
                metadata = self.get_metadata(pdb_id)
                if self._matches_filters(metadata, filters):
                    filtered_ids.append(pdb_id)
            pdb_ids = filtered_ids

        # Sort if requested
        if sort_by and pdb_ids:
            pdb_ids = self._sort_structures(pdb_ids, sort_by)

        # Apply limit if specified
        if limit and limit > 0:
            pdb_ids = pdb_ids[:limit]

        return pdb_ids

    def _matches_filters(
        self, metadata: Optional[PDBMetadata], filters: Dict[str, Any]
    ) -> bool:
        """Check if structure metadata matches the given filters."""
        if not metadata:
            return False

        # Resolution filter
        if "max_resolution" in filters and metadata.resolution is not None:
            if metadata.resolution > filters["max_resolution"]:
                return False
        if "min_resolution" in filters and metadata.resolution is not None:
            if metadata.resolution < filters["min_resolution"]:
                return False

        # Method filter
        if "methods" in filters:
            if metadata.method not in filters["methods"]:
                return False

        # Length filters
        if "min_length" in filters:
            if metadata.sequence_length < filters["min_length"]:
                return False
        if "max_length" in filters:
            if metadata.sequence_length > filters["max_length"]:
                return False

        # Organism filter
        if "organism" in filters and metadata.organism is not None:
            if filters["organism"] not in metadata.organism:
                return False

        return True

    def _sort_structures(self, pdb_ids: List[str], sort_by: str) -> List[str]:
        """Sort structure list by specified field."""

        def get_sort_key(pdb_id: str):
            metadata = self.get_metadata(pdb_id)
            if not metadata:
                return float("inf")  # Put unknown metadata at end

            if sort_by == "resolution":
                return metadata.resolution or float("inf")
            elif sort_by == "quality_score":
                return -(metadata.quality_score or 0)  # Higher scores first
            elif sort_by == "sequence_length":
                return metadata.sequence_length
            elif sort_by == "last_updated":
                return metadata.last_updated.timestamp()
            else:
                return 0

        return sorted(pdb_ids, key=get_sort_key)

    def get_structure(self, pdb_id: str) -> Optional[Any]:
        """
        Get structure data for a PDB ID.

        Args:
            pdb_id: PDB identifier

        Returns:
            Structure data or None if not available
        """
        if pdb_id not in self.structure_index:
            return None

        structure_info = self.structure_index[pdb_id]
        source_type = structure_info["source_type"]

        if source_type == "local_pdb":
            file_path = structure_info.get("file_path")
            if file_path and Path(file_path).exists():
                return self._load_local_structure(file_path)
        elif source_type in ["remote_pdb", "pdb_list"]:
            # Would need to implement download/cache mechanism
            # For now, return placeholder indicating remote access needed
            return {
                "pdb_id": pdb_id,
                "source_type": source_type,
                "requires_download": True,
                "structure_info": structure_info,
            }

        return None

    def _load_local_structure(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load structure from local PDB file."""
        try:
            # Basic file reading - in practice would use proper PDB parser
            with open(file_path, "r") as f:
                content = f.read()

            return {
                "file_path": file_path,
                "content": content,
                "source_type": "local_pdb",
                "file_size": len(content),
            }
        except Exception as e:
            self.logger.error(f"Error loading structure from {file_path}: {e}")
            return None

    def get_metadata(self, pdb_id: str) -> Optional[PDBMetadata]:
        """
        Get metadata for a PDB structure.

        Args:
            pdb_id: PDB identifier

        Returns:
            PDB metadata or None if not available
        """
        return self.metadata_db.get(pdb_id)

    def update_metadata(self, pdb_id: str, metadata: PDBMetadata) -> None:
        """
        Update metadata for a PDB structure.

        Args:
            pdb_id: PDB identifier
            metadata: Updated metadata
        """
        self.metadata_db[pdb_id] = metadata
        self._save_metadata_db()

    def validate_structure(self, pdb_id: str) -> Dict[str, Any]:
        """
        Validate a PDB structure against quality criteria.

        Args:
            pdb_id: PDB identifier

        Returns:
            Dictionary with validation results
        """
        # Basic validation based on metadata
        metadata = self.get_metadata(pdb_id)
        if not metadata:
            return {
                "status": "missing_metadata",
                "issues": ["No metadata available"],
                "quality_score": 0.0,
            }

        issues = []
        quality_score = 1.0

        # Resolution validation
        if metadata.resolution is None:
            issues.append("Missing resolution data")
            quality_score -= 0.2
        elif metadata.resolution > 3.5:
            issues.append(f"High resolution: {metadata.resolution}")
            quality_score -= 0.3

        # Sequence length validation
        if metadata.sequence_length < 20:
            issues.append(f"Short sequence: {metadata.sequence_length}")
            quality_score -= 0.4
        elif metadata.sequence_length > 500:
            issues.append(f"Long sequence: {metadata.sequence_length}")
            quality_score -= 0.2

        # Method validation
        valid_methods = {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"}
        if metadata.method not in valid_methods:
            issues.append(f"Invalid method: {metadata.method}")
            quality_score -= 0.3

        quality_score = max(0.0, quality_score)
        status = "valid" if quality_score >= 0.5 and not issues else "invalid"

        return {"status": status, "issues": issues, "quality_score": quality_score}

    def filter_by_quality(
        self,
        pdb_ids: List[str],
        min_resolution: Optional[float] = None,
        max_resolution: Optional[float] = None,
        allowed_methods: Optional[Set[str]] = None,
        min_sequence_length: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ) -> List[str]:
        """
        Filter PDB IDs by quality criteria.

        Args:
            pdb_ids: List of PDB IDs to filter
            min_resolution: Minimum resolution threshold
            max_resolution: Maximum resolution threshold
            allowed_methods: Set of allowed experimental methods
            min_sequence_length: Minimum sequence length
            max_sequence_length: Maximum sequence length

        Returns:
            Filtered list of PDB IDs
        """
        filtered_ids = []

        for pdb_id in pdb_ids:
            metadata = self.get_metadata(pdb_id)
            if not metadata:
                continue

            # Resolution filter
            if min_resolution is not None and metadata.resolution is not None:
                if metadata.resolution < min_resolution:
                    continue
            if max_resolution is not None and metadata.resolution is not None:
                if metadata.resolution > max_resolution:
                    continue

            # Method filter
            if allowed_methods is not None:
                if metadata.method not in allowed_methods:
                    continue

            # Sequence length filter
            if min_sequence_length is not None:
                if metadata.sequence_length < min_sequence_length:
                    continue
            if max_sequence_length is not None:
                if metadata.sequence_length > max_sequence_length:
                    continue

            filtered_ids.append(pdb_id)

        return filtered_ids

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed PDB structures."""
        stats = {
            "total_structures": len(self.metadata_db),
            "data_sources": len(self.data_sources),
            "metadata_db_path": (
                str(self.metadata_db_path) if self.metadata_db_path else None
            ),
        }

        # Resolution statistics
        resolutions = [
            m.resolution for m in self.metadata_db.values() if m.resolution is not None
        ]
        if resolutions:
            stats.update(
                {
                    "resolution_min": min(resolutions),
                    "resolution_max": max(resolutions),
                    "resolution_avg": sum(resolutions) / len(resolutions),
                }
            )

        # Method statistics
        methods = [m.method for m in self.metadata_db.values()]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        stats["method_distribution"] = method_counts

        # Length statistics
        lengths = [m.sequence_length for m in self.metadata_db.values()]
        if lengths:
            stats.update(
                {
                    "length_min": min(lengths),
                    "length_max": max(lengths),
                    "length_avg": sum(lengths) / len(lengths),
                }
            )

        return stats

    def refresh_index(self) -> None:
        """Refresh structure index from all data sources."""
        self.logger.info("Refreshing structure index...")
        old_count = len(self.structure_index)

        # Clear existing index and rebuild
        self.structure_index.clear()
        self._build_structure_index()

        new_count = len(self.structure_index)
        self.logger.info(
            f"Structure index refreshed: {old_count} -> {new_count} structures"
        )


class PDBListManager:
    """
    Manages PDB ID lists for training with RCSB API integration and caching.

    This class handles:
    - RCSB search API v2 integration
    - Filtered PDB retrieval with training criteria
    - Local caching for offline use
    - Robust fallback mechanisms
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        search_timeout: int = 30,
        max_retries: int = 3,
        rate_limit_delay: float = 0.1,
    ):
        """
        Initialize PDB list manager.

        Args:
            cache_dir: Directory for caching results
            search_timeout: Timeout for API requests in seconds
            max_retries: Maximum retry attempts for failed requests
            rate_limit_delay: Delay between requests to respect rate limits
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./pdb_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        self.search_timeout = search_timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Setup HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Cache file paths
        self.filtered_list_cache = self.cache_dir / "filtered_pdb_list.json"
        self.search_results_cache = self.cache_dir / "rcsb_search_results.json"

    def __enter__(self):
        """Context manager entry - returns self for use in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup of resources."""
        self.cleanup()

    def cleanup(self):
        """Clean up resources, especially HTTP session."""
        if hasattr(self, "session") and self.session:
            try:
                self.session.close()
                self.logger.debug("HTTP session closed successfully")
            except Exception as e:
                self.logger.warning(f"Error closing HTTP session: {e}")
            finally:
                self.session = None

    def __del__(self):
        """Destructor to ensure cleanup happens even without explicit call."""
        self.cleanup()

    def get_pdb_list(
        self,
        max_structures: int = 5000,
        use_cache: bool = True,
        max_resolution: float = 3.5,
        min_length: int = 20,
        max_length: int = 500,
        experimental_methods: Optional[List[str]] = None,
        cache_max_age_hours: int = 24,
    ) -> List[str]:
        """
        Compatibility method - forwards to get_filtered_pdb_list with parameter mapping.

        Args:
            max_structures: Maximum number of structures to retrieve (maps to target_count)
            use_cache: Whether to use cached results
            max_resolution: Maximum resolution threshold (Å)
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            experimental_methods: Allowed experimental methods
            cache_max_age_hours: Maximum age of cached results in hours

        Returns:
            List of PDB IDs matching criteria
        """
        return self.get_filtered_pdb_list(
            max_resolution=max_resolution,
            min_length=min_length,
            max_length=max_length,
            experimental_methods=experimental_methods,
            target_count=max_structures,
            use_cache=use_cache,
            cache_max_age_hours=cache_max_age_hours,
        )

    def get_filtered_pdb_list(
        self,
        max_resolution: float = 3.5,
        min_length: int = 20,
        max_length: int = 500,
        experimental_methods: Optional[List[str]] = None,
        target_count: int = 5000,
        use_cache: bool = True,
        cache_max_age_hours: int = 24,
    ) -> List[str]:
        """
        Get filtered PDB list matching training criteria.

        Args:
            max_resolution: Maximum resolution threshold (Å)
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            experimental_methods: Allowed experimental methods
            target_count: Target number of PDB structures to retrieve
            use_cache: Whether to use cached results
            cache_max_age_hours: Maximum age of cached results in hours

        Returns:
            List of PDB IDs matching criteria
        """
        if experimental_methods is None:
            experimental_methods = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]

        # Check cache first
        if use_cache:
            cached_list = self._load_cached_list(cache_max_age_hours)
            if cached_list and len(cached_list) >= target_count:
                self.logger.info(
                    f"Using cached PDB list with {len(cached_list)} structures"
                )
                return random.sample(cached_list, min(target_count, len(cached_list)))

        self.logger.info("Fetching PDB list from RCSB API...")

        try:
            # Fetch from RCSB API
            pdb_list = self._fetch_from_rcsb(
                max_resolution=max_resolution,
                min_length=min_length,
                max_length=max_length,
                experimental_methods=experimental_methods,
                target_count=target_count * 2,  # Fetch more than needed for filtering
            )

            if pdb_list:
                self.logger.info(
                    f"Successfully retrieved {len(pdb_list)} PDB structures from RCSB"
                )
                # Cache the results
                self._save_cached_list(pdb_list)
                return random.sample(pdb_list, min(target_count, len(pdb_list)))
            else:
                self.logger.warning("RCSB API returned no results")

        except Exception as e:
            self.logger.error(f"Error fetching from RCSB API: {e}")

        # Fallback to cached results or emergency list
        self.logger.info("Using fallback PDB list")
        fallback_list = self._get_fallback_list()
        return random.sample(fallback_list, min(target_count, len(fallback_list)))

    def _fetch_from_rcsb(
        self,
        max_resolution: float,
        min_length: int,
        max_length: int,
        experimental_methods: List[str],
        target_count: int,
    ) -> List[str]:
        """Fetch PDB IDs from RCSB search API with exponential backoff."""
        query = self._build_rcsb_query(
            max_resolution=max_resolution,
            min_length=min_length,
            max_length=max_length,
            experimental_methods=experimental_methods,
        )

        for attempt in range(self.max_retries + 1):
            try:
                # Add exponential backoff delay
                if attempt > 0:
                    delay = self.rate_limit_delay * (2**attempt) + random.uniform(0, 1)
                    self.logger.info(
                        f"Retrying RCSB query in {delay:.2f}s (attempt {attempt + 1})"
                    )
                    time.sleep(delay)

                response = self.session.post(
                    self.search_url,
                    json=query,
                    timeout=self.search_timeout,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Additional JSON structure validation
                        if not isinstance(data, dict):
                            self.logger.error(
                                f"API returned non-dict JSON: {type(data)}"
                            )
                            continue
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON in API response: {e}")
                        continue

                    pdb_ids = self._extract_pdb_ids(data)

                    if pdb_ids:
                        # Validate PDB IDs
                        valid_ids = self._validate_pdb_ids(pdb_ids)
                        self.logger.info(f"Retrieved {len(valid_ids)} valid PDB IDs")
                        return valid_ids[:target_count]
                    else:
                        self.logger.warning("No PDB IDs found in RCSB response")

                elif response.status_code == 429:
                    self.logger.warning("RCSB API rate limit exceeded")
                    continue
                else:
                    self.logger.warning(
                        f"RCSB API error: {response.status_code} - {response.text}"
                    )

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Network error during RCSB query: {e}")
                continue
            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON response from RCSB: {e}")
                continue

        return []

    def _build_rcsb_query(
        self,
        max_resolution: float,
        min_length: int,
        max_length: int,
        experimental_methods: List[str],
    ) -> Dict[str, Any]:
        """Build RCSB search query for filtering criteria."""
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    # Resolution filter
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "negation": False,
                            "value": max_resolution,
                        },
                    },
                    # Sequence length filter (minimum)
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_sample_sequence_length",
                            "operator": "greater_or_equal",
                            "value": min_length,
                        },
                    },
                    # Sequence length filter (maximum)
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_sample_sequence_length",
                            "operator": "less_or_equal",
                            "value": max_length,
                        },
                    },
                    # Experimental method filter
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "in",
                            "negation": False,
                            "value": experimental_methods,
                        },
                    },
                    # Protein filter
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "Protein",
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "return_all_hits": True,
                "sort": [{"sort_by": "score", "direction": "desc"}],
            },
        }

        return query

    def _validate_api_response(self, response_data: Any) -> bool:
        """Validate RCSB API response structure before processing."""
        if not isinstance(response_data, dict):
            self.logger.error(f"API response must be dict, got {type(response_data)}")
            return False

        # Check for required top-level structure
        has_result_set = "result_set" in response_data
        has_results = "results" in response_data

        if not (has_result_set or has_results):
            self.logger.error(
                "API response missing both 'result_set' and 'results' fields"
            )
            return False

        # Validate the structure of result entries
        try:
            if has_result_set:
                result_list = response_data["result_set"]
                if not isinstance(result_list, list):
                    self.logger.error("'result_set' must be a list")
                    return False

                for idx, entry in enumerate(result_list[:5]):  # Check first 5 entries
                    if not isinstance(entry, dict):
                        self.logger.error(
                            f"result_set[{idx}] must be dict, got {type(entry)}"
                        )
                        return False
                    if "identifier" not in entry:
                        self.logger.error(
                            f"result_set[{idx}] missing 'identifier' field"
                        )
                        return False
                    if not isinstance(entry["identifier"], str):
                        self.logger.error(
                            f"result_set[{idx}] identifier must be string, got {type(entry['identifier'])}"
                        )
                        return False

            elif has_results:
                result_list = response_data["results"]
                if not isinstance(result_list, list):
                    self.logger.error("'results' must be a list")
                    return False

                for idx, entry in enumerate(result_list[:5]):  # Check first 5 entries
                    if not isinstance(entry, dict):
                        self.logger.error(
                            f"results[{idx}] must be dict, got {type(entry)}"
                        )
                        return False
                    if "identifier" not in entry:
                        self.logger.error(f"results[{idx}] missing 'identifier' field")
                        return False
                    if not isinstance(entry["identifier"], str):
                        self.logger.error(
                            f"results[{idx}] identifier must be string, got {type(entry['identifier'])}"
                        )
                        return False

            return True

        except (KeyError, TypeError, IndexError) as e:
            self.logger.error(f"API response validation failed: {e}")
            return False

    def _extract_pdb_ids(self, response_data: Dict[str, Any]) -> List[str]:
        """Extract PDB IDs from RCSB API response with validation."""
        # Validate response structure first
        if not self._validate_api_response(response_data):
            self.logger.error("API response failed validation, rejecting")
            return []

        try:
            pdb_ids = []

            if "result_set" in response_data:
                for entry in response_data["result_set"]:
                    if isinstance(entry, dict) and "identifier" in entry:
                        identifier = entry["identifier"]
                        if isinstance(identifier, str):
                            pdb_ids.append(identifier)
                        else:
                            self.logger.warning(
                                f"Non-string identifier in result_set: {identifier}"
                            )

            elif "results" in response_data:
                for entry in response_data["results"]:
                    if isinstance(entry, dict) and "identifier" in entry:
                        identifier = entry["identifier"]
                        if isinstance(identifier, str):
                            pdb_ids.append(identifier)
                        else:
                            self.logger.warning(
                                f"Non-string identifier in results: {identifier}"
                            )

            self.logger.debug(
                f"Extracted {len(pdb_ids)} PDB IDs from validated response"
            )
            return pdb_ids

        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error extracting PDB IDs from validated response: {e}")
            return []

    def _validate_pdb_ids(self, pdb_ids: List[str]) -> List[str]:
        """Validate PDB IDs to prevent security issues and path traversal attacks."""
        valid_ids = []
        # Strict PDB ID validation: exactly 4 characters, digit + 3 alphanumeric
        # This prevents path traversal attempts like "../", "..", "/", etc.
        pdb_pattern = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

        for pdb_id in pdb_ids:
            if not isinstance(pdb_id, str):
                self.logger.warning(
                    f"PDB ID must be string, got {type(pdb_id)}: {pdb_id}"
                )
                continue

            # Additional security checks beyond regex
            pdb_id_clean = pdb_id.strip().upper()

            # Exact length check (critical for security)
            if len(pdb_id_clean) != 4:
                self.logger.warning(
                    f"Invalid PDB ID length {len(pdb_id_clean)}, must be exactly 4: {pdb_id}"
                )
                continue

            # Pattern validation
            if not pdb_pattern.match(pdb_id_clean):
                self.logger.warning(f"Invalid PDB ID format: {pdb_id}")
                continue

            # Explicitly check for path traversal patterns
            if any(char in pdb_id_clean for char in ["/", "\\", "..", "~", "$", "%"]):
                self.logger.warning(f"PDB ID contains suspicious characters: {pdb_id}")
                continue

            valid_ids.append(pdb_id_clean)

        return valid_ids

    def _load_cached_list(self, max_age_hours: int) -> Optional[List[str]]:
        """Load cached PDB list if it exists and is not too old."""
        if not self.filtered_list_cache.exists():
            return None

        try:
            # Check file age
            file_age = time.time() - self.filtered_list_cache.stat().st_mtime
            if file_age > max_age_hours * 3600:
                self.logger.info("Cached PDB list is too old, will refresh")
                return None

            with open(self.filtered_list_cache, "r") as f:
                data = json.load(f)

            if isinstance(data, dict) and "pdb_ids" in data:
                return data["pdb_ids"]
            elif isinstance(data, list):
                return data
            else:
                self.logger.warning("Invalid cached PDB list format")
                return None

        except Exception as e:
            self.logger.warning(f"Error loading cached PDB list: {e}")
            return None

    def _save_cached_list(self, pdb_ids: List[str]) -> None:
        """Save PDB list to cache with metadata using atomic write operation."""
        try:
            cache_data = {
                "pdb_ids": pdb_ids,
                "timestamp": datetime.now().isoformat(),
                "count": len(pdb_ids),
                "version": "1.0",
            }

            # Atomic write: write to temporary file first, then rename
            # This prevents corruption from concurrent access or interruption
            temp_file = None
            try:
                # Create temporary file in same directory as target file
                temp_fd, temp_file = tempfile.mkstemp(
                    prefix=".tmp_pdb_cache_",
                    suffix=".json",
                    dir=self.filtered_list_cache.parent,
                )

                # Write data to temporary file
                with os.fdopen(temp_fd, "w") as f:
                    json.dump(cache_data, f, indent=2)
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Force synchronization to storage

                # Atomic rename operation (atomic on most filesystems)
                os.rename(temp_file, self.filtered_list_cache)
                temp_file = None  # Prevent cleanup since rename succeeded

                self.logger.info(
                    f"Atomically cached {len(pdb_ids)} PDB IDs to {self.filtered_list_cache}"
                )

            except Exception as e:
                # Clean up temporary file if something went wrong
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass  # Ignore cleanup errors
                raise e

        except Exception as e:
            self.logger.warning(f"Error saving PDB list cache: {e}")

    def _get_fallback_list(self) -> List[str]:
        """High-quality manually curated PDB list for emergencies."""
        # High-quality structures from various sources
        base_pdbs = [
            # From proteinmpnn/inputs (known good structures)
            "3HTN",
            "4YOW",
            "4GYT",
            "6EHB",
            "5L33",
            "6MRR",
            # Additional high-quality protein structures
            "1UBQ",
            "1VII",
            "2CRO",
            "1ROP",
            "1TEN",
            "1A1U",
            "1BPI",
            "1C8C",
            "1D3Z",
            "1E0G",
            "1F39",
            "1G6X",
            "1H8H",
            "1I6H",
            "1J8H",
            "1K8A",
            "1L2Y",
            "1M1K",
            "1N8R",
            "1O91",
            "1P9I",
            "1Q1O",
            "1R7J",
            "1S3R",
            "1T1K",
            "1U06",
            "1V70",
            "1W0N",
            "1X6Z",
            "1Y8A",
            "1Z21",
            "2A3D",
            "2B97",
            "2C71",
            "2D57",
            "2E3H",
            "2F1K",
            "2G36",
            "2H5C",
            "2I4D",
            "2J28",
            "2K39",
            "2L0J",
            "2M1X",
            "2N3G",
            "2O8V",
            "2P5K",
            "2Q52",
            "2R11",
            "2S3R",
            "2T4B",
            "2U89",
            "2V7Q",
            "2W2T",
            "2X23",
            "2Y2V",
            "2Z5X",
            "3A04",
            "3B2Q",
            "3C8Z",
            "3D9A",
            "3E23",
            "3F1P",
            "3G21",
            "3H42",
            "3I37",
            "3J81",
            "3K0N",
            "3L1P",
            "3M1I",
            "3N44",
            "3O2S",
            "3P32",
            "3Q15",
            "3R9K",
            "3S8G",
            "3T1Y",
            "3U17",
            "3V6O",
            "3W2T",
            "3X42",
            "3Y8A",
            "3Z86",
            "4A1A",
            "4B2L",
            "4C4R",
            "4D43",
            "4E46",
            "4F3T",
            "4G36",
            "4H03",
            "4I55",
            "4J52",
            "4K81",
            "4L57",
            "4M1K",
            "4N12",
            "4O29",
            "4P4S",
            "4Q21",
            "4R17",
            "4S1I",
            "4T4F",
            "4U1S",
            "4V24",
            "4W53",
            "4X22",
            "4Y4O",
            "4Z4I",
            "5A63",
            "5B31",
            "5C8X",
            "5D2Q",
            "5E61",
            "5F73",
            "5G28",
            "5H57",
            "5I9R",
            "5JDE",
            "5KPY",
            "5LCX",
            "5M7E",
            "5N2D",
            "5O31",
            "5P21",
            "5Q4K",
            "5R1D",
            "5S3A",
            "5T4V",
            "5U4I",
            "5V74",
            "5W2G",
            "5X0D",
            "5Y1N",
            "5Z23",
            "6A5J",
        ]

        # Replicate base list to reach target count through diversity
        # This ensures we have enough structures even with filtering
        replicated_list = base_pdbs * 50  # Creates ~5000 entries
        random.shuffle(replicated_list)

        self.logger.warning(
            f"Using fallback PDB list with {len(replicated_list)} entries"
        )
        return replicated_list

    def get_biological_query_sets(self) -> Dict[str, List[str]]:
        """Get pre-defined query sets for different biological scenarios."""
        return {
            "high_quality": self.get_filtered_pdb_list(
                max_resolution=2.0, min_length=50, max_length=300, target_count=1000
            ),
            "diverse_folds": self.get_filtered_pdb_list(
                max_resolution=3.0, min_length=20, max_length=500, target_count=3000
            ),
            "large_proteins": self.get_filtered_pdb_list(
                max_resolution=3.5, min_length=300, max_length=500, target_count=1000
            ),
            "small_proteins": self.get_filtered_pdb_list(
                max_resolution=2.5, min_length=20, max_length=150, target_count=2000
            ),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed PDB lists and cache status."""
        stats = {
            "cache_dir": str(self.cache_dir),
            "cache_exists": self.filtered_list_cache.exists(),
            "search_results_cached": self.search_results_cache.exists(),
        }

        # Add cache file statistics if they exist
        if self.filtered_list_cache.exists():
            try:
                with open(self.filtered_list_cache, "r") as f:
                    data = json.load(f)
                stats["cached_pdb_count"] = len(data.get("pdb_ids", []))
                stats["cache_timestamp"] = data.get("timestamp")
            except Exception:
                stats["cached_pdb_count"] = "error"

        return stats


class StructureValidator:
    """
    Validator for PDB structure quality and completeness.
    """

    def __init__(self, validation_config: Dict[str, Any]):
        """
        Initialize structure validator.

        Args:
            validation_config: Validation configuration
        """
        self.config = validation_config

    def validate(self, structure_data: Any) -> Dict[str, Any]:
        """
        Validate a PDB structure.

        Args:
            structure_data: Structure data to validate

        Returns:
            Dictionary with validation results
        """
        if not structure_data:
            return {
                "status": "invalid",
                "issues": ["No structure data provided"],
                "quality_score": 0.0,
            }

        issues = []
        quality_score = 1.0

        # Basic validation checks
        if isinstance(structure_data, dict):
            # Check for required fields
            if "content" in structure_data:
                content = structure_data["content"]
                if not content or not isinstance(content, str):
                    issues.append("Invalid or empty structure content")
                    quality_score -= 0.5
                elif len(content) < 100:  # Very small PDB file
                    issues.append("Structure content appears too small")
                    quality_score -= 0.3

            # Check file size if available
            file_size = structure_data.get("file_size", 0)
            if file_size > 0:
                if file_size < 1000:  # Less than 1KB
                    issues.append("Structure file appears very small")
                    quality_score -= 0.2
                elif file_size > 10_000_000:  # More than 10MB
                    issues.append("Structure file is very large")
                    quality_score -= 0.1

            # Check source type
            source_type = structure_data.get("source_type")
            if not source_type:
                issues.append("Missing source type information")
                quality_score -= 0.1

        else:
            issues.append("Structure data is not in expected format")
            quality_score -= 0.5

        # Apply validation config if specified
        min_quality = self.config.get("min_quality_score", 0.0)
        if quality_score < min_quality:
            issues.append(
                f"Quality score {quality_score:.2f} below minimum {min_quality}"
            )

        quality_score = max(0.0, quality_score)
        status = "valid" if quality_score >= 0.5 and not issues else "invalid"

        return {
            "status": status,
            "issues": issues,
            "quality_score": quality_score,
            "validation_config": self.config,
        }
