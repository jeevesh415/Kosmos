"""
Neo4j knowledge graph interface for scientific literature.

Provides full CRUD operations for nodes (Paper, Concept, Method, Author)
and relationships (CITES, USES_METHOD, DISCUSSES, AUTHORED, RELATED_TO).
"""

import logging
import subprocess
import threading
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path

from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from py2neo.errors import Neo4jError

from kosmos.config import get_config
from kosmos.literature.base_client import PaperMetadata

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Neo4j-based knowledge graph for scientific literature.

    Manages a graph database with:
    - Node types: Paper, Concept, Method, Author
    - Relationship types: CITES, USES_METHOD, DISCUSSES, AUTHORED, RELATED_TO

    Supports full CRUD operations and complex graph queries.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        auto_start_container: bool = True,
        create_indexes: bool = True
    ):
        """
        Initialize knowledge graph connection.

        Args:
            uri: Neo4j connection URI (default: from config)
            user: Neo4j username (default: from config)
            password: Neo4j password (default: from config)
            database: Neo4j database name (default: from config)
            auto_start_container: Whether to auto-start Docker container
            create_indexes: Whether to create indexes on initialization

        Example:
            ```python
            graph = KnowledgeGraph()

            # Add a paper
            paper_node = graph.create_paper(paper_metadata)

            # Query citations
            citations = graph.get_citations(paper_id)
            ```
        """
        config = get_config()

        # Connection settings
        self.uri = uri or config.neo4j.uri
        self.user = user or config.neo4j.user
        self.password = password or config.neo4j.password
        self.database = database or config.neo4j.database

        # Auto-start container if needed
        if auto_start_container:
            self._ensure_container_running()

        # Connect to Neo4j
        self._connected = False
        try:
            self.graph = Graph(
                self.uri,
                auth=(self.user, self.password),
                name=self.database
            )

            # Test connection
            self.graph.run("RETURN 1").data()
            self._connected = True

            logger.info(
                f"Connected to Neo4j at {self.uri} "
                f"(database={self.database})"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.graph = None
            # Don't raise - let callers check self.connected

        # Create matchers for queries (only if connected)
        if self._connected:
            self.node_matcher = NodeMatcher(self.graph)
            self.rel_matcher = RelationshipMatcher(self.graph)

            # Create indexes for performance
            if create_indexes:
                self._create_indexes()
        else:
            self.node_matcher = None
            self.rel_matcher = None

    @property
    def connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self._connected

    def _ensure_container_running(self):
        """
        Ensure Neo4j Docker container is running.

        Starts the container via docker-compose if not already running.
        """
        try:
            # Check if container is running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=kosmos-neo4j", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if "kosmos-neo4j" in result.stdout:
                logger.info("Neo4j container already running")
                return

            # Start container
            logger.info("Starting Neo4j container...")
            subprocess.run(
                ["docker-compose", "up", "-d", "neo4j"],
                check=True,
                capture_output=True,
                timeout=60
            )

            # Wait for container to be ready
            logger.info("Waiting for Neo4j to be ready...")
            max_retries = 30
            for i in range(max_retries):
                try:
                    result = subprocess.run(
                        ["docker", "exec", "kosmos-neo4j", "cypher-shell",
                         "-u", "neo4j", "-p", "kosmos-password", "RETURN 1"],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        logger.info("Neo4j container ready")
                        return
                except subprocess.TimeoutExpired:
                    pass

                time.sleep(2)

            logger.warning("Neo4j container started but health check timed out")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Neo4j container: {e}")
            logger.warning("Proceeding without auto-start - ensure Neo4j is running manually")
        except FileNotFoundError:
            logger.warning("docker or docker-compose not found - skipping auto-start")

    def _create_indexes(self):
        """Create indexes for efficient queries."""
        indexes = [
            # Paper indexes
            "CREATE INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)",
            "CREATE INDEX paper_doi IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
            "CREATE INDEX paper_arxiv IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id)",
            "CREATE INDEX paper_pubmed IF NOT EXISTS FOR (p:Paper) ON (p.pubmed_id)",

            # Author indexes
            "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",

            # Concept indexes
            "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX concept_domain IF NOT EXISTS FOR (c:Concept) ON (c.domain)",

            # Method indexes
            "CREATE INDEX method_name IF NOT EXISTS FOR (m:Method) ON (m.name)",
            "CREATE INDEX method_category IF NOT EXISTS FOR (m:Method) ON (m.category)",
        ]

        for index_query in indexes:
            try:
                self.graph.run(index_query)
            except Exception as e:
                logger.debug(f"Index creation note: {e}")

        logger.info("Created graph indexes")

    # ==================== Paper CRUD ====================

    def create_paper(
        self,
        paper: PaperMetadata,
        merge: bool = True
    ) -> Node:
        """
        Create or update a Paper node.

        Args:
            paper: PaperMetadata object
            merge: If True, merge with existing node; if False, create new

        Returns:
            Created/merged Paper node

        Example:
            ```python
            node = graph.create_paper(paper_metadata)
            print(f"Created paper node: {node['title']}")
            ```
        """
        properties = {
            "id": paper.primary_identifier,
            "title": paper.title,
            "abstract": paper.abstract or "",
            "year": paper.year or 0,
            "citation_count": paper.citation_count,
            "domain": paper.fields[0] if paper.fields else "unknown",
            "created_at": datetime.now().isoformat()
        }

        # Add identifiers
        if paper.doi:
            properties["doi"] = paper.doi
        if paper.arxiv_id:
            properties["arxiv_id"] = paper.arxiv_id
        if paper.pubmed_id:
            properties["pubmed_id"] = paper.pubmed_id
        if paper.url:
            properties["url"] = paper.url

        if merge:
            node = self.node_matcher.match("Paper", id=paper.primary_identifier).first()
            if node:
                node.update(properties)
                self.graph.push(node)
                logger.debug(f"Updated paper node: {paper.title}")
            else:
                node = Node("Paper", **properties)
                self.graph.create(node)
                logger.debug(f"Created paper node: {paper.title}")
        else:
            node = Node("Paper", **properties)
            self.graph.create(node)
            logger.debug(f"Created paper node: {paper.title}")

        return node

    def get_paper(self, paper_id: str) -> Optional[Node]:
        """
        Get a Paper node by ID.

        Args:
            paper_id: Paper identifier (DOI, arXiv ID, or PubMed ID)

        Returns:
            Paper node or None if not found
        """
        # Try by primary ID
        node = self.node_matcher.match("Paper", id=paper_id).first()
        if node:
            return node

        # Try by DOI
        node = self.node_matcher.match("Paper", doi=paper_id).first()
        if node:
            return node

        # Try by arXiv ID
        node = self.node_matcher.match("Paper", arxiv_id=paper_id).first()
        if node:
            return node

        # Try by PubMed ID
        node = self.node_matcher.match("Paper", pubmed_id=paper_id).first()
        return node

    def update_paper(self, paper_id: str, properties: Dict[str, Any]) -> Optional[Node]:
        """
        Update a Paper node.

        Args:
            paper_id: Paper identifier
            properties: Properties to update

        Returns:
            Updated node or None if not found
        """
        node = self.get_paper(paper_id)
        if node:
            node.update(properties)
            self.graph.push(node)
            logger.debug(f"Updated paper {paper_id}")
            return node

        logger.warning(f"Paper {paper_id} not found for update")
        return None

    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a Paper node and all its relationships.

        Args:
            paper_id: Paper identifier

        Returns:
            True if deleted, False if not found
        """
        node = self.get_paper(paper_id)
        if node:
            self.graph.delete(node)
            logger.info(f"Deleted paper {paper_id}")
            return True

        logger.warning(f"Paper {paper_id} not found for deletion")
        return False

    # ==================== Author CRUD ====================

    def create_author(
        self,
        name: str,
        affiliation: Optional[str] = None,
        h_index: Optional[int] = None,
        merge: bool = True
    ) -> Node:
        """
        Create or update an Author node.

        Args:
            name: Author name
            affiliation: Author affiliation
            h_index: h-index
            merge: If True, merge with existing; if False, create new

        Returns:
            Created/merged Author node
        """
        properties = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "paper_count": 0
        }

        if affiliation:
            properties["affiliation"] = affiliation
        if h_index is not None:
            properties["h_index"] = h_index

        if merge:
            node = self.node_matcher.match("Author", name=name).first()
            if node:
                node.update(properties)
                self.graph.push(node)
                logger.debug(f"Updated author node: {name}")
            else:
                node = Node("Author", **properties)
                self.graph.create(node)
                logger.debug(f"Created author node: {name}")
        else:
            node = Node("Author", **properties)
            self.graph.create(node)
            logger.debug(f"Created author node: {name}")

        return node

    def get_author(self, name: str) -> Optional[Node]:
        """Get an Author node by name."""
        return self.node_matcher.match("Author", name=name).first()

    def update_author(self, name: str, properties: Dict[str, Any]) -> Optional[Node]:
        """Update an Author node."""
        node = self.get_author(name)
        if node:
            node.update(properties)
            self.graph.push(node)
            logger.debug(f"Updated author {name}")
            return node
        return None

    def delete_author(self, name: str) -> bool:
        """Delete an Author node and relationships."""
        node = self.get_author(name)
        if node:
            self.graph.delete(node)
            logger.info(f"Deleted author {name}")
            return True
        return False

    # ==================== Concept CRUD ====================

    def create_concept(
        self,
        name: str,
        description: Optional[str] = None,
        domain: Optional[str] = None,
        merge: bool = True
    ) -> Node:
        """
        Create or update a Concept node.

        Args:
            name: Concept name
            description: Concept description
            domain: Scientific domain
            merge: If True, merge with existing

        Returns:
            Created/merged Concept node
        """
        properties = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "frequency": 0
        }

        if description:
            properties["description"] = description
        if domain:
            properties["domain"] = domain

        if merge:
            node = self.node_matcher.match("Concept", name=name).first()
            if node:
                node.update(properties)
                self.graph.push(node)
            else:
                node = Node("Concept", **properties)
                self.graph.create(node)
        else:
            node = Node("Concept", **properties)
            self.graph.create(node)

        return node

    def get_concept(self, name: str) -> Optional[Node]:
        """Get a Concept node by name."""
        return self.node_matcher.match("Concept", name=name).first()

    def update_concept(self, name: str, properties: Dict[str, Any]) -> Optional[Node]:
        """Update a Concept node."""
        node = self.get_concept(name)
        if node:
            node.update(properties)
            self.graph.push(node)
            return node
        return None

    def delete_concept(self, name: str) -> bool:
        """Delete a Concept node."""
        node = self.get_concept(name)
        if node:
            self.graph.delete(node)
            logger.info(f"Deleted concept {name}")
            return True
        return False

    # ==================== Method CRUD ====================

    def create_method(
        self,
        name: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        merge: bool = True
    ) -> Node:
        """
        Create or update a Method node.

        Args:
            name: Method name
            description: Method description
            category: Method category (experimental, computational, analytical)
            merge: If True, merge with existing

        Returns:
            Created/merged Method node
        """
        properties = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }

        if description:
            properties["description"] = description
        if category:
            properties["category"] = category

        if merge:
            node = self.node_matcher.match("Method", name=name).first()
            if node:
                node.update(properties)
                self.graph.push(node)
            else:
                node = Node("Method", **properties)
                self.graph.create(node)
        else:
            node = Node("Method", **properties)
            self.graph.create(node)

        return node

    def get_method(self, name: str) -> Optional[Node]:
        """Get a Method node by name."""
        return self.node_matcher.match("Method", name=name).first()

    def update_method(self, name: str, properties: Dict[str, Any]) -> Optional[Node]:
        """Update a Method node."""
        node = self.get_method(name)
        if node:
            node.update(properties)
            self.graph.push(node)
            return node
        return None

    def delete_method(self, name: str) -> bool:
        """Delete a Method node."""
        node = self.get_method(name)
        if node:
            self.graph.delete(node)
            logger.info(f"Deleted method {name}")
            return True
        return False

    # ==================== Relationship CRUD ====================

    def create_citation(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        merge: bool = True
    ) -> Optional[Relationship]:
        """
        Create a CITES relationship between papers.

        Args:
            citing_paper_id: ID of citing paper
            cited_paper_id: ID of cited paper
            merge: If True, merge with existing

        Returns:
            Created relationship or None if papers not found
        """
        citing = self.get_paper(citing_paper_id)
        cited = self.get_paper(cited_paper_id)

        if not citing or not cited:
            logger.warning(f"Cannot create citation: papers not found")
            return None

        # Check if relationship exists
        existing = self.rel_matcher.match([citing, cited], r_type="CITES").first()

        if existing and merge:
            return existing

        rel = Relationship(citing, "CITES", cited, created_at=datetime.now().isoformat())
        self.graph.create(rel)
        logger.debug(f"Created CITES: {citing_paper_id} -> {cited_paper_id}")

        return rel

    def create_authored(
        self,
        author_name: str,
        paper_id: str,
        order: Optional[int] = None,
        role: Optional[str] = None,
        merge: bool = True
    ) -> Optional[Relationship]:
        """
        Create an AUTHORED relationship.

        Args:
            author_name: Author name
            paper_id: Paper ID
            order: Author order in paper
            role: Author role (first, corresponding, etc.)
            merge: If True, merge with existing

        Returns:
            Created relationship or None if nodes not found
        """
        author = self.get_author(author_name)
        paper = self.get_paper(paper_id)

        if not author or not paper:
            logger.warning(f"Cannot create AUTHORED: nodes not found")
            return None

        properties = {"created_at": datetime.now().isoformat()}
        if order is not None:
            properties["order"] = order
        if role:
            properties["role"] = role

        rel = Relationship(author, "AUTHORED", paper, **properties)
        created = not merge
        if merge:
            existing = self.graph.match((author, paper), r_type="AUTHORED").first()
            created = existing is None
        self.graph.merge(rel) if merge else self.graph.create(rel)

        # Only update author paper count when a new relationship is created
        if created:
            author["paper_count"] = author.get("paper_count", 0) + 1
            self.graph.push(author)

        return rel

    def create_discusses(
        self,
        paper_id: str,
        concept_name: str,
        relevance_score: float = 1.0,
        section: Optional[str] = None,
        merge: bool = True
    ) -> Optional[Relationship]:
        """
        Create a DISCUSSES relationship between paper and concept.

        Args:
            paper_id: Paper ID
            concept_name: Concept name
            relevance_score: Relevance score (0-1)
            section: Paper section where discussed
            merge: If True, merge with existing

        Returns:
            Created relationship or None if nodes not found
        """
        paper = self.get_paper(paper_id)
        concept = self.get_concept(concept_name)

        if not paper or not concept:
            logger.warning(f"Cannot create DISCUSSES: nodes not found")
            return None

        properties = {
            "relevance_score": relevance_score,
            "created_at": datetime.now().isoformat()
        }
        if section:
            properties["section"] = section

        rel = Relationship(paper, "DISCUSSES", concept, **properties)
        created = not merge
        if merge:
            existing = self.graph.match((paper, concept), r_type="DISCUSSES").first()
            created = existing is None
        self.graph.merge(rel) if merge else self.graph.create(rel)

        # Only update concept frequency when a new relationship is created
        if created:
            concept["frequency"] = concept.get("frequency", 0) + 1
            self.graph.push(concept)

        return rel

    def create_uses_method(
        self,
        paper_id: str,
        method_name: str,
        confidence: float = 1.0,
        context: Optional[str] = None,
        merge: bool = True
    ) -> Optional[Relationship]:
        """
        Create a USES_METHOD relationship.

        Args:
            paper_id: Paper ID
            method_name: Method name
            confidence: Confidence score (0-1)
            context: Context where method is used
            merge: If True, merge with existing

        Returns:
            Created relationship or None if nodes not found
        """
        paper = self.get_paper(paper_id)
        method = self.get_method(method_name)

        if not paper or not method:
            logger.warning(f"Cannot create USES_METHOD: nodes not found")
            return None

        properties = {
            "confidence": confidence,
            "created_at": datetime.now().isoformat()
        }
        if context:
            properties["context"] = context

        rel = Relationship(paper, "USES_METHOD", method, **properties)
        created = not merge
        if merge:
            existing = self.graph.match((paper, method), r_type="USES_METHOD").first()
            created = existing is None
        self.graph.merge(rel) if merge else self.graph.create(rel)

        # Only update method usage count when a new relationship is created
        if created:
            method["usage_count"] = method.get("usage_count", 0) + 1
            self.graph.push(method)

        return rel

    def create_related_to(
        self,
        concept1_name: str,
        concept2_name: str,
        similarity: float = 0.5,
        source: str = "manual",
        merge: bool = True
    ) -> Optional[Relationship]:
        """
        Create a RELATED_TO relationship between concepts.

        Args:
            concept1_name: First concept name
            concept2_name: Second concept name
            similarity: Similarity score (0-1)
            source: Source of relationship (manual, semantic, cooccurrence)
            merge: If True, merge with existing

        Returns:
            Created relationship or None if nodes not found
        """
        concept1 = self.get_concept(concept1_name)
        concept2 = self.get_concept(concept2_name)

        if not concept1 or not concept2:
            logger.warning(f"Cannot create RELATED_TO: concepts not found")
            return None

        properties = {
            "similarity": similarity,
            "source": source,
            "created_at": datetime.now().isoformat()
        }

        rel = Relationship(concept1, "RELATED_TO", concept2, **properties)
        self.graph.merge(rel) if merge else self.graph.create(rel)

        return rel

    # ==================== Graph Queries ====================

    def get_citations(self, paper_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """
        Get papers cited by a given paper.

        Args:
            paper_id: Paper ID
            depth: Citation depth (1 = direct citations, 2 = citations of citations, etc.)

        Returns:
            List of cited papers with metadata
        """
        query = f"""
        MATCH (p:Paper {{id: $paper_id}})-[:CITES*1..{depth}]->(cited:Paper)
        RETURN cited, length((p)-[:CITES*]->(cited)) as depth
        ORDER BY depth, cited.citation_count DESC
        """

        results = self.graph.run(query, paper_id=paper_id).data()
        return [{"paper": dict(r["cited"]), "depth": r["depth"]} for r in results]

    def get_citing_papers(self, paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get papers that cite a given paper.

        Args:
            paper_id: Paper ID
            limit: Maximum number of results

        Returns:
            List of citing papers
        """
        query = """
        MATCH (citing:Paper)-[:CITES]->(p:Paper {id: $paper_id})
        RETURN citing
        ORDER BY citing.year DESC, citing.citation_count DESC
        LIMIT $limit
        """

        results = self.graph.run(query, paper_id=paper_id, limit=limit).data()
        return [dict(r["citing"]) for r in results]

    def get_author_papers(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Get all papers by an author.

        Args:
            author_name: Author name

        Returns:
            List of papers
        """
        query = """
        MATCH (a:Author {name: $author_name})-[r:AUTHORED]->(p:Paper)
        RETURN p, r.order as author_order
        ORDER BY p.year DESC
        """

        results = self.graph.run(query, author_name=author_name).data()
        return [{"paper": dict(r["p"]), "order": r["author_order"]} for r in results]

    def get_concept_papers(
        self,
        concept_name: str,
        min_relevance: float = 0.5,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get papers discussing a concept.

        Args:
            concept_name: Concept name
            min_relevance: Minimum relevance score
            limit: Maximum results

        Returns:
            List of papers with relevance scores
        """
        query = """
        MATCH (p:Paper)-[r:DISCUSSES]->(c:Concept {name: $concept_name})
        WHERE r.relevance_score >= $min_relevance
        RETURN p, r.relevance_score as relevance
        ORDER BY r.relevance_score DESC, p.citation_count DESC
        LIMIT $limit
        """

        results = self.graph.run(
            query,
            concept_name=concept_name,
            min_relevance=min_relevance,
            limit=limit
        ).data()

        return [{"paper": dict(r["p"]), "relevance": r["relevance"]} for r in results]

    def get_method_papers(self, method_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get papers using a specific method.

        Args:
            method_name: Method name
            limit: Maximum results

        Returns:
            List of papers
        """
        query = """
        MATCH (p:Paper)-[r:USES_METHOD]->(m:Method {name: $method_name})
        RETURN p, r.confidence as confidence
        ORDER BY r.confidence DESC, p.year DESC
        LIMIT $limit
        """

        results = self.graph.run(query, method_name=method_name, limit=limit).data()
        return [{"paper": dict(r["p"]), "confidence": r["confidence"]} for r in results]

    def get_related_concepts(
        self,
        concept_name: str,
        min_similarity: float = 0.5,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept.

        Args:
            concept_name: Concept name
            min_similarity: Minimum similarity threshold
            limit: Maximum results

        Returns:
            List of related concepts with similarity scores
        """
        query = """
        MATCH (c1:Concept {name: $concept_name})-[r:RELATED_TO]-(c2:Concept)
        WHERE r.similarity >= $min_similarity
        RETURN c2, r.similarity as similarity
        ORDER BY r.similarity DESC
        LIMIT $limit
        """

        results = self.graph.run(
            query,
            concept_name=concept_name,
            min_similarity=min_similarity,
            limit=limit
        ).data()

        return [{"concept": dict(r["c2"]), "similarity": r["similarity"]} for r in results]

    def find_related_papers(
        self,
        paper_id: str,
        max_hops: int = 2,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find papers related to a given paper through the knowledge graph.

        Uses multiple relationship types: citations, shared concepts, shared methods.

        Args:
            paper_id: Paper ID
            max_hops: Maximum graph hops
            limit: Maximum results

        Returns:
            List of related papers with relationship paths
        """
        query = f"""
        MATCH path = (p1:Paper {{id: $paper_id}})-[*1..{max_hops}]-(p2:Paper)
        WHERE p1 <> p2
        WITH p2, length(path) as distance, collect(distinct type(relationships(path)[0])) as rel_types
        RETURN p2, distance, rel_types
        ORDER BY distance, p2.citation_count DESC
        LIMIT $limit
        """

        results = self.graph.run(query, paper_id=paper_id, limit=limit).data()
        return [
            {
                "paper": dict(r["p2"]),
                "distance": r["distance"],
                "relationship_types": r["rel_types"]
            }
            for r in results
        ]

    def get_concept_cooccurrence(
        self,
        concept_name: str,
        min_papers: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find concepts that co-occur with a given concept in papers.

        Args:
            concept_name: Concept name
            min_papers: Minimum number of shared papers

        Returns:
            List of co-occurring concepts with counts
        """
        query = """
        MATCH (c1:Concept {name: $concept_name})<-[:DISCUSSES]-(p:Paper)-[:DISCUSSES]->(c2:Concept)
        WHERE c1 <> c2
        WITH c2, count(distinct p) as cooccurrence_count
        WHERE cooccurrence_count >= $min_papers
        RETURN c2, cooccurrence_count
        ORDER BY cooccurrence_count DESC
        """

        results = self.graph.run(
            query,
            concept_name=concept_name,
            min_papers=min_papers
        ).data()

        return [{"concept": dict(r["c2"]), "count": r["cooccurrence_count"]} for r in results]

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics.

        Returns:
            Dictionary with node and relationship counts
        """
        stats = {}

        # Node counts
        for label in ["Paper", "Author", "Concept", "Method"]:
            count = self.graph.run(f"MATCH (n:{label}) RETURN count(n) as count").data()[0]["count"]
            stats[f"{label.lower()}_count"] = count

        # Relationship counts
        for rel_type in ["CITES", "AUTHORED", "DISCUSSES", "USES_METHOD", "RELATED_TO"]:
            count = self.graph.run(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            ).data()[0]["count"]
            stats[f"{rel_type.lower()}_count"] = count

        return stats

    def clear_graph(self):
        """Clear all nodes and relationships from the graph."""
        self.graph.run("MATCH (n) DETACH DELETE n")
        logger.warning("Cleared all graph data")


# Singleton instance
_knowledge_graph: Optional[KnowledgeGraph] = None
_knowledge_graph_lock = threading.Lock()


def get_knowledge_graph(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    reset: bool = False
) -> KnowledgeGraph:
    """
    Get or create the singleton knowledge graph instance.

    Args:
        uri: Neo4j URI
        user: Neo4j username
        password: Neo4j password
        database: Neo4j database name
        reset: Whether to reset the singleton

    Returns:
        KnowledgeGraph instance
    """
    global _knowledge_graph
    if _knowledge_graph is None or reset:
        with _knowledge_graph_lock:
            if _knowledge_graph is None or reset:
                _knowledge_graph = KnowledgeGraph(
                    uri=uri,
                    user=user,
                    password=password,
                    database=database
                )
    return _knowledge_graph


def reset_knowledge_graph():
    """Reset the singleton knowledge graph (useful for testing)."""
    global _knowledge_graph
    _knowledge_graph = None
