"""
Postprocessor para priorizar documentos provenientes de archivos internos.
Coloca primero los nodos cuyo nombre de archivo empiece con "unemi_".
"""
import logging
from typing import Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field

logger = logging.getLogger(__name__)


class RoleBasedPostprocessor(BaseNodePostprocessor):
    """
    Postprocessor que prioriza documentos con prefijo "unemi_".
    Mantiene todos los nodos, pero pone primero los internos.
    """
    
    user_role: Optional[str] = Field(default=None, description="Rol del usuario")
    
    def __init__(self, user_role: Optional[str] = None, **kwargs):
        super().__init__(user_role=user_role or None, **kwargs)
        logger.debug(f"RoleBasedPostprocessor inicializado con rol: {user_role}")
    
    def _get_file_name(self, node: NodeWithScore) -> Optional[str]:
        """Extrae el nombre del archivo desde los metadatos del nodo."""
        # Los metadatos pueden estar en node.metadata o node.node.metadata
        metadata = None
        
        # Intentar acceder directamente desde node.metadata
        if hasattr(node, 'metadata') and node.metadata:
            metadata = node.metadata
        # Si no, intentar desde node.node.metadata
        elif hasattr(node, 'node') and node.node is not None:
            metadata = getattr(node.node, 'metadata', None)
        
        if not metadata:
            return None
        
        # Si metadata es un dict, usar get
        if isinstance(metadata, dict):
            file_name = metadata.get('file_name') or metadata.get('filename') or metadata.get('file')
        else:
            # Si es otro tipo, intentar acceder como atributo
            file_name = getattr(metadata, 'file_name', None) or getattr(metadata, 'filename', None)
        
        if file_name:
            return str(file_name)
        
        return None
    
    def _should_prioritize(self, node: NodeWithScore) -> bool:
        """Determina si un nodo debe ser priorizado (prefijo unemi_)."""
        file_name = self._get_file_name(node)
        if not file_name:
            return False
        
        should_prioritize = file_name.lower().startswith("unemi_")
        if should_prioritize:
            logger.debug(f"Priorizando documento: {file_name}")
        
        return should_prioritize
    
    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        """
        Reordena los nodos colocando primero los que provienen de archivos "unemi_".
        No descarta el resto: solo cambia el orden de prioridad.
        """
        if not nodes:
            return nodes
        
        prioritized = []
        non_prioritized = []
        
        for node in nodes:
            if self._should_prioritize(node):
                prioritized.append(node)
            else:
                non_prioritized.append(node)
        
        if prioritized:
            logger.info(
                f"RoleBasedPostprocessor: Priorizados {len(prioritized)} documentos "
                f"de {len(nodes)} totales (rol={self.user_role})"
            )
            return prioritized + non_prioritized
        
        return nodes

