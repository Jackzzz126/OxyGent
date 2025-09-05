# oxygent/databases/db_vector_v2/es_db.py
import logging
from abc import ABC
from typing import Dict, List, Optional, Callable, Any

import numpy as np
import pandas as pd
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import RequestError

from oxygent.databases.db_vector_v2.base_vector_db import BaseVectorDB
from oxygent.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class EsVectorDB(BaseVectorDB, ABC):
    """Elasticsearch Vector Database Implementation.

    Implements vector storage and similarity search using Elasticsearch 8.x+ dense_vector
    type, providing compatibility with BaseVectorDB interface while leveraging ES's
    distributed search capabilities.
    """

    def __init__(self,
                 hosts: List[str],
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 emb_func: Optional[Callable] = None,
                 timeout: int = 30,
                 maxsize: int = 200):
        """Initialize Elasticsearch vector database client.

        Args:
            hosts: List of ES cluster addresses (e.g., ["http://localhost:9200"])
            user: Authentication username
            password: Authentication password
            emb_func: Asynchronous embedding function
            timeout: Connection timeout in seconds
            maxsize: Maximum size of the connection pool
        """
        super().__init__()
        self.emb_func = emb_func
        self.emb_cache = EmbeddingCache()

        # Initialize ES client
        try:
            self.client = AsyncElasticsearch(
                hosts=hosts,
                http_auth=(user, password) if user and password else None,
                timeout=timeout,
                maxsize=maxsize
            )
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
            self.client = None

    async def create_space(self, index_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Create a vector index (space) in Elasticsearch.

        The index mapping must include a dense_vector field (typically named "vector").

        Args:
            index_name: Name of the index to create
            body: Index configuration including mappings and settings

        Returns:
            Dictionary containing operation result
        """
        if not self.client:
            raise RuntimeError("Elasticsearch client not initialized")

        if await self._index_exists(index_name):
            logger.warning(f"Index {index_name} already exists")
            return {"status": "exists", "index": index_name}

        try:
            response = await self.client.indices.create(
                index=index_name,
                body=body
            )
            logger.info(f"Successfully created index {index_name}")
            return response
        except RequestError as e:
            logger.error(f"Failed to create index {index_name}: {str(e)}")
            raise

    async def query_search(self,
                           index_name: str,
                           query: str,
                           retrieval_nums: int = 10,
                           fields: List[str] = [],
                           threshold: Optional[float] = None) -> pd.DataFrame:
        """Search by text query (converts to embedding first).

        Args:
            index_name: Target index name
            query: Text query to embed and search
            retrieval_nums: Maximum number of results to return
            fields: List of fields to include in results
            threshold: Minimum similarity score threshold

        Returns:
            DataFrame containing search results with scores
        """
        if not self.emb_func:
            raise ValueError("Embedding function not provided")

        # Get embedding from cache or generate new
        emb = await self.emb_cache.get_embedding(
            text=query,
            emb_func=self.emb_func
        )
        return await self.emb_search(
            index_name=index_name,
            emb=emb,
            retrieval_nums=retrieval_nums,
            fields=fields,
            threshold=threshold
        )

    async def emb_search(self,
                         index_name: str,
                         emb: np.ndarray,
                         retrieval_nums: int = 10,
                         fields: List[str] = [],
                         threshold: Optional[float] = None) -> pd.DataFrame:
        """Search by direct vector embedding.

        Args:
            index_name: Target index name
            emb: Embedding vector for similarity search
            retrieval_nums: Maximum number of results to return
            fields: List of fields to include in results
            threshold: Minimum similarity score threshold

        Returns:
            DataFrame containing search results with scores
        """
        if not self.client:
            raise RuntimeError("Elasticsearch client not initialized")

        # Convert numpy array to list
        vector = emb.tolist() if isinstance(emb, np.ndarray) else emb

        # Build k-NN query
        query_body = {
            "knn": {
                "vector": {  # Assumes vector field is named "vector"
                    "vector": vector,
                    "k": retrieval_nums
                }
            },
            "_source": fields if fields else ["*"]
        }

        # Add score threshold filter if specified
        if threshold is not None:
            query_body["post_filter"] = {
                "range": {"_score": {"gte": threshold}}
            }

        try:
            response = await self.client.search(
                index=index_name,
                body=query_body,
                size=retrieval_nums
            )
            return self._format_search_results(response)
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return pd.DataFrame()

    async def filter_and_emb_search(self,
                                    index_name: str,
                                    emb: np.ndarray,
                                    retrieval_nums: int = 10,
                                    fields: List[str] = [],
                                    filter: Dict[str, Any] = {}) -> pd.DataFrame:
        """Hybrid search with vector similarity and metadata filters.

        Args:
            index_name: Target index name
            emb: Embedding vector for similarity search
            retrieval_nums: Maximum number of results to return
            fields: List of fields to include in results
            filter: Dictionary of metadata filters (e.g., {"app_name": "test"})

        Returns:
            DataFrame containing filtered search results
        """
        if not self.client:
            raise RuntimeError("Elasticsearch client not initialized")

        vector = emb.tolist() if isinstance(emb, np.ndarray) else emb
        filter_clauses = self._build_filter_clauses(filter)

        query_body = {
            "knn": {
                "vector": {
                    "vector": vector,
                    "k": retrieval_nums,
                    "filter": {"bool": {"must": filter_clauses}} if filter_clauses else None
                }
            },
            "_source": fields if fields else ["*"]
        }

        try:
            response = await self.client.search(
                index=index_name,
                body=query_body,
                size=retrieval_nums
            )
            return self._format_search_results(response)
        except Exception as e:
            logger.error(f"Filtered vector search failed: {str(e)}")
            return pd.DataFrame()

    async def upload_by_df(self,
                           index_name: str,
                           df: pd.DataFrame,
                           vector_col: str = "vector",
                           id_col: Optional[str] = None) -> str:
        """Bulk upload documents from DataFrame.

        Args:
            index_name: Target index name
            df: DataFrame containing documents to upload
            vector_col: Name of column containing vectors
            id_col: Optional column name to use as document ID

        Returns:
            Status message indicating success or failure
        """
        if not self.client:
            raise RuntimeError("Elasticsearch client not initialized")

        if vector_col not in df.columns:
            raise ValueError(f"Vector column '{vector_col}' not found in DataFrame")

        # Prepare bulk operations
        bulk_ops = []
        for idx, row in df.iterrows():
            # Create index operation
            doc_id = row[id_col] if id_col and id_col in row else str(idx)
            bulk_ops.append({
                "index": {
                    "_index": index_name,
                    "_id": doc_id
                }
            })

            # Prepare document (convert vector to list)
            doc = row.to_dict()
            if isinstance(doc[vector_col], np.ndarray):
                doc[vector_col] = doc[vector_col].tolist()
            bulk_ops.append(doc)

        # Execute bulk operation
        try:
            response = await self.client.bulk(operations=bulk_ops)

            if response["errors"]:
                errors = [item["index"]["error"] for item in response["items"] if "error" in item["index"]]
                error_msg = f"Failed to insert {len(errors)} documents: {errors[:5]}"  # Show first 5 errors
                logger.error(error_msg)
                return error_msg

            success_count = len([item for item in response["items"] if item["index"]["status"] in (200, 201)])
            return f"Successfully uploaded {success_count}/{len(df)} documents to {index_name}"
        except Exception as e:
            error_msg = f"Bulk upload failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def delete_by_filter(self, index_name: str, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Delete documents matching filter criteria.

        Args:
            index_name: Target index name
            filter: Filter criteria for deletion

        Returns:
            Dictionary containing deletion result
        """
        if not self.client:
            raise RuntimeError("Elasticsearch client not initialized")

        query = {
            "query": {
                "bool": {
                    "must": self._build_filter_clauses(filter)
                }
            }
        }

        try:
            response = await self.client.delete_by_query(
                index=index_name,
                body=query
            )
            logger.info(f"Deleted {response['deleted']} documents from {index_name}")
            return response
        except Exception as e:
            logger.error(f"Delete by filter failed: {str(e)}")
            raise

    async def check_space_exist(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        if not self.client:
            return False
        return await self._index_exists(index_name)

    async def drop_space(self, index_name: str) -> str:
        """Delete an existing index.

        Args:
            index_name: Name of the index to delete

        Returns:
            Status message
        """
        if not self.client:
            raise RuntimeError("Elasticsearch client not initialized")

        if not await self._index_exists(index_name):
            return f"Index {index_name} does not exist"

        try:
            await self.client.indices.delete(index=index_name)
            return f"Index {index_name} deleted successfully"
        except Exception as e:
            error_msg = f"Failed to delete index {index_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def close(self):
        """Close the Elasticsearch client connection."""
        if self.client:
            await self.client.close()
            logger.info("Elasticsearch client closed")

    # Helper methods
    async def _index_exists(self, index_name: str) -> bool:
        """Check if an index exists (internal helper)."""
        try:
            return await self.client.indices.exists(index=index_name)
        except Exception as e:
            logger.warning(f"Error checking index existence: {str(e)}")
            return False

    def _build_filter_clauses(self, filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert filter dict to ES bool query clauses."""
        clauses = []
        for key, value in filter.items():
            if isinstance(value, list):
                clauses.append({"terms": {key: value}})
            else:
                clauses.append({"term": {key: value}})
        return clauses

    def _format_search_results(self, response: Dict[str, Any]) -> pd.DataFrame:
        """Convert ES search response to DataFrame."""
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return pd.DataFrame()

        data = []
        for hit in hits:
            doc = hit["_source"].copy()
            doc["_id"] = hit["_id"]
            doc["_score"] = hit["_score"]
            data.append(doc)

        return pd.DataFrame(data)