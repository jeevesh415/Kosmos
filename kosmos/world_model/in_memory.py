"""
In-memory world model implementation for testing.

This module provides an in-memory implementation of WorldModelStorage
that can be used for testing without requiring Neo4j.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kosmos.world_model.interface import EntityManager, WorldModelStorage
from kosmos.world_model.models import Annotation, Entity, Relationship


class InMemoryWorldModel(WorldModelStorage, EntityManager):
    """
    In-memory implementation of WorldModelStorage for testing.

    This provides the same interface as Neo4jWorldModel but stores
    all data in dictionaries, eliminating the need for Neo4j in tests.
    """

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._annotations: Dict[str, List[Annotation]] = {}

    def add_entity(self, entity: Entity, merge: bool = True) -> str:
        """Add entity to the in-memory store."""
        entity_id = entity.id or str(uuid.uuid4())
        entity.id = entity_id

        if merge and entity_id in self._entities:
            existing = self._entities[entity_id]
            merged_props = {**existing.properties, **entity.properties}
            existing.properties = merged_props
            if entity.confidence > existing.confidence:
                existing.confidence = entity.confidence
            existing.updated_at = datetime.now()
        else:
            if entity.created_at is None:
                entity.created_at = datetime.now()
            entity.updated_at = datetime.now()
            self._entities[entity_id] = entity

        return entity_id

    def get_entity(self, entity_id: str, project: Optional[str] = None) -> Optional[Entity]:
        """Retrieve entity by ID with optional project filter."""
        entity = self._entities.get(entity_id)
        if entity and project is not None and entity.project != project:
            return None
        return entity

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> None:
        """Update entity properties."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found: {entity_id}")
        entity = self._entities[entity_id]
        entity.properties.update(updates)
        entity.updated_at = datetime.now()

    def delete_entity(self, entity_id: str) -> None:
        """Delete entity and all its relationships."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found: {entity_id}")
        del self._entities[entity_id]

        # Remove related relationships
        to_remove = [
            r_id
            for r_id, r in self._relationships.items()
            if r.source_id == entity_id or r.target_id == entity_id
        ]
        for r_id in to_remove:
            del self._relationships[r_id]

        # Remove annotations
        if entity_id in self._annotations:
            del self._annotations[entity_id]

    def add_relationship(self, relationship: Relationship) -> str:
        """Add relationship between two entities."""
        rel_id = relationship.id or str(uuid.uuid4())
        relationship.id = rel_id
        self._relationships[rel_id] = relationship
        return rel_id

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Retrieve relationship by ID."""
        return self._relationships.get(relationship_id)

    def query_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "outgoing",
        max_depth: int = 1,
    ) -> List[Entity]:
        """Query entities related to a given entity."""
        related = []
        for rel in self._relationships.values():
            if direction == "incoming" and rel.target_id == entity_id:
                if relationship_type is None or rel.type == relationship_type:
                    if rel.source_id in self._entities:
                        related.append(self._entities[rel.source_id])
            elif direction == "outgoing" and rel.source_id == entity_id:
                if relationship_type is None or rel.type == relationship_type:
                    if rel.target_id in self._entities:
                        related.append(self._entities[rel.target_id])
            elif direction == "both":
                if rel.source_id == entity_id or rel.target_id == entity_id:
                    if relationship_type is None or rel.type == relationship_type:
                        other_id = (
                            rel.target_id if rel.source_id == entity_id else rel.source_id
                        )
                        if other_id in self._entities:
                            related.append(self._entities[other_id])
        return related

    def export_graph(self, filepath: str, project: Optional[str] = None) -> None:
        """Export knowledge graph to file."""
        entities = list(self._entities.values())
        relationships = list(self._relationships.values())

        if project:
            entities = [e for e in entities if e.project == project]
            entity_ids = {e.id for e in entities}
            relationships = [
                r
                for r in relationships
                if r.source_id in entity_ids and r.target_id in entity_ids
            ]

        export_data = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source": "kosmos",
            "mode": "in_memory",
            "project": project,
            "statistics": self.get_statistics(project),
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships],
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    def import_graph(
        self, filepath: str, clear: bool = False, project: Optional[str] = None
    ) -> None:
        """Import knowledge graph from file."""
        if clear:
            self.reset(project)

        with open(filepath, "r") as f:
            data = json.load(f)

        for entity_data in data.get("entities", []):
            entity = Entity.from_dict(entity_data)
            if project:
                entity.project = project
            self.add_entity(entity, merge=True)

        for rel_data in data.get("relationships", []):
            relationship = Relationship.from_dict(rel_data)
            self.add_relationship(relationship)

    def get_statistics(self, project: Optional[str] = None) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        entities = list(self._entities.values())
        if project:
            entities = [e for e in entities if e.project == project]

        entity_types: Dict[str, int] = {}
        for e in entities:
            entity_types[e.type] = entity_types.get(e.type, 0) + 1

        rel_types: Dict[str, int] = {}
        for r in self._relationships.values():
            rel_types[r.type] = rel_types.get(r.type, 0) + 1

        projects = list({e.project for e in self._entities.values() if e.project})

        return {
            "entity_count": len(entities),
            "relationship_count": len(self._relationships),
            "entity_types": entity_types,
            "relationship_types": rel_types,
            "projects": projects,
        }

    def reset(self, project: Optional[str] = None) -> None:
        """Clear all knowledge graph data."""
        if project:
            to_remove = [
                e_id for e_id, e in self._entities.items() if e.project == project
            ]
            for e_id in to_remove:
                self.delete_entity(e_id)
        else:
            self._entities.clear()
            self._relationships.clear()
            self._annotations.clear()

    def close(self) -> None:
        """Close connections and cleanup resources."""
        pass  # No resources to close for in-memory implementation

    # EntityManager methods

    def verify_entity(self, entity_id: str, verified_by: str) -> None:
        """Mark entity as manually verified."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found: {entity_id}")
        self._entities[entity_id].verified = True
        self._entities[entity_id].properties["verified_by"] = verified_by
        self._entities[entity_id].properties["verified_at"] = datetime.now().isoformat()

    def add_annotation(self, entity_id: str, annotation: Annotation) -> None:
        """Add annotation to entity."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found: {entity_id}")
        if entity_id not in self._annotations:
            self._annotations[entity_id] = []
        self._annotations[entity_id].append(annotation)

    def get_annotations(self, entity_id: str) -> List[Annotation]:
        """Get all annotations for an entity."""
        return self._annotations.get(entity_id, [])
