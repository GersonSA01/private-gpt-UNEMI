from dataclasses import dataclass
from typing import TYPE_CHECKING
import logging

from injector import inject, singleton
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    SimilarityPostprocessor,
)
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.node_store.role_based_postprocessor import RoleBasedPostprocessor
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None


class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None


@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ChatEngineInput":
        # Hacer una copia para no modificar la lista original
        messages_copy = list(messages)
        
        # Detect if there is a system message, extract the last message and chat history
        system_message = (
            messages_copy[0]
            if len(messages_copy) > 0 and messages_copy[0].role == MessageRole.SYSTEM
            else None
        )
        last_message = (
            messages_copy[-1]
            if len(messages_copy) > 0 and messages_copy[-1].role == MessageRole.USER
            else None
        )
        # Remove from messages list the system message and last message,
        # if they exist. The rest is the chat history.
        if system_message:
            messages_copy.pop(0)
        if last_message:
            messages_copy.pop(-1)
        chat_history = messages_copy if len(messages_copy) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
        )


@singleton
class ChatService:
    settings: Settings

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=llm_component.llm,
            embed_model=embedding_component.embedding_model,
            show_progress=True,
        )

    def _expand_query(self, query: str) -> list[str]:
        """Expande la consulta usando el LLM para generar variaciones y sinónimos.
        
        Esto ayuda a encontrar información incluso si las palabras no coinciden exactamente
        con las del documento.
        """
        try:
            expansion_prompt = f"""Dada la siguiente consulta del usuario, genera 2-3 variaciones o reformulaciones que puedan ayudar a encontrar información relacionada en documentos. 
Incluye sinónimos, términos relacionados, y formas alternativas de expresar lo mismo.

Consulta original: {query}

Genera solo las variaciones, una por línea, sin numeración ni explicaciones:"""
            
            # Usar el LLM para generar variaciones
            response = self.llm_component.llm.complete(expansion_prompt)
            variations = [query]  # Incluir la consulta original
            
            if response and response.text:
                # Dividir por líneas y limpiar
                for line in response.text.strip().split('\n'):
                    line = line.strip()
                    # Remover numeración si existe (1., 2., etc.)
                    if line and not line.startswith('Consulta'):
                        # Limpiar prefijos comunes
                        for prefix in ['-', '•', '1.', '2.', '3.', '*']:
                            if line.startswith(prefix):
                                line = line[len(prefix):].strip()
                        if line and len(line) > 5:  # Solo agregar si tiene contenido sustancial
                            variations.append(line)
            
            # Limitar a máximo 4 variaciones (original + 3)
            variations = variations[:4]
            logger.debug(f"Consulta expandida: {variations}")
            return variations
        except Exception as e:
            logger.warning(f"Error al expandir consulta: {e}, usando consulta original")
            return [query]

    def _chat_engine(
        self,
        system_prompt: str | None = None,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
        user_role: str | None = None,
    ) -> BaseChatEngine:
        settings = self.settings
        if use_context:
            # Crear un retriever con expansión de consulta
            base_retriever = self.vector_store_component.get_retriever(
                index=self.index,
                context_filter=context_filter,
                similarity_top_k=self.settings.rag.similarity_top_k,
            )
            
            # Crear un retriever que expande la consulta antes de buscar
            class QueryExpansionRetriever:
                """Retriever que expande la consulta usando el LLM antes de buscar."""
                def __init__(self, base_retriever, expand_query_func, llm):
                    self.base_retriever = base_retriever
                    self.expand_query = expand_query_func
                    self.llm = llm
                
                def retrieve(self, query_bundle):
                    """Retrieve con expansión de consulta."""
                    # Expandir la consulta
                    query_str = str(query_bundle.query_str) if hasattr(query_bundle, 'query_str') else str(query_bundle)
                    expanded_queries = self.expand_query(query_str)
                    
                    # Hacer búsquedas con todas las variaciones
                    all_nodes = []
                    seen_node_ids = set()
                    
                    for expanded_query in expanded_queries:
                        try:
                            # Crear un nuevo query bundle con la consulta expandida
                            from llama_index.core.schema import QueryBundle
                            query_bundle_expanded = QueryBundle(query_str=expanded_query)
                            nodes = self.base_retriever.retrieve(query_bundle_expanded)
                            
                            # Agregar nodos únicos, excluyendo archivos temporales (tmp)
                            for node in nodes:
                                # Filtrar archivos que empiezan con "tmp"
                                node_metadata = getattr(node.node, 'metadata', {}) if hasattr(node, 'node') else {}
                                file_name = node_metadata.get('file_name', '')
                                if file_name.lower().startswith('tmp'):
                                    logger.debug(f"Filtrando archivo temporal: {file_name}")
                                    continue
                                
                                node_id = node.node.node_id if hasattr(node.node, 'node_id') else id(node.node)
                                if node_id not in seen_node_ids:
                                    all_nodes.append(node)
                                    seen_node_ids.add(node_id)
                        except Exception as e:
                            logger.debug(f"Error en búsqueda con variación '{expanded_query}': {e}")
                            continue
                    
                    # Ordenar por score y limitar
                    all_nodes.sort(key=lambda x: x.score if hasattr(x, 'score') and x.score is not None else 0, reverse=True)
                    top_k = getattr(self.base_retriever, 'similarity_top_k', 10)
                    return all_nodes[:top_k]
                
                def __getattr__(self, name):
                    # Delegar otros atributos al retriever base
                    return getattr(self.base_retriever, name)
            
            # Usar el retriever con expansión si está habilitado
            if settings.rag.query_expansion:
                vector_index_retriever = QueryExpansionRetriever(
                    base_retriever,
                    self._expand_query,
                    self.llm_component.llm
                )
            else:
                vector_index_retriever = base_retriever
            node_postprocessors: list[BaseNodePostprocessor] = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
            ]
            
            # Priorizar documentos internos (prefijo unemi_) antes del resto
            role_postprocessor = RoleBasedPostprocessor(user_role=user_role)
            node_postprocessors.append(role_postprocessor)
            logger.info("RoleBasedPostprocessor habilitado: se priorizarán archivos 'unemi_'")
            
            if settings.rag.similarity_value:
                node_postprocessors.append(
                    SimilarityPostprocessor(
                        similarity_cutoff=settings.rag.similarity_value
                    )
                )

            if settings.rag.rerank.enabled:
                rerank_postprocessor = SentenceTransformerRerank(
                    model=settings.rag.rerank.model, top_n=settings.rag.rerank.top_n
                )
                node_postprocessors.append(rerank_postprocessor)

            return ContextChatEngine.from_defaults(
                system_prompt=system_prompt,
                retriever=vector_index_retriever,
                llm=self.llm_component.llm,  # Takes no effect at the moment
                node_postprocessors=node_postprocessors,
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=system_prompt,
                llm=self.llm_component.llm,
            )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
        user_role: str | None = None,
    ) -> CompletionGen:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        # Asegurar que el mensaje no esté vacío
        if not last_message or not last_message.strip():
            raise ValueError("El mensaje no puede estar vacío para generar embeddings")

        # Loggear solo el prompt limpio (system + user), sin contexto de documentos
        logger.info("=" * 50)
        logger.info("** PROMPT ENVIADO AL LLM (STREAMING, sin contexto de documentos): **")
        if system_prompt:
            # Mostrar solo las primeras 500 caracteres del system prompt para no saturar
            system_preview = system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt
            logger.info(f"system: {system_preview}")
        if last_message:
            logger.info(f"user: {last_message}")
        if chat_history:
            logger.info(f"chat_history: {len(chat_history)} mensajes previos")
        logger.info("=" * 50)

        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
            user_role=user_role,
        )
        streaming_response = chat_engine.stream_chat(
            message=last_message.strip(),
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]
        completion_gen = CompletionGen(
            response=streaming_response.response_gen, sources=sources
        )
        return completion_gen

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
        user_role: str | None = None,
    ) -> Completion:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        # Loggear solo el prompt limpio (system + user), sin contexto de documentos
        logger.info("=" * 50)
        logger.info("** PROMPT ENVIADO AL LLM (sin contexto de documentos): **")
        if system_prompt:
            # Mostrar solo las primeras 500 caracteres del system prompt para no saturar
            system_preview = system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt
            logger.info(f"system: {system_preview}")
        if last_message:
            logger.info(f"user: {last_message}")
        if chat_history:
            logger.info(f"chat_history: {len(chat_history)} mensajes previos")
        logger.info("=" * 50)

        chat_engine = self._chat_engine(
            system_prompt=system_prompt,
            use_context=use_context,
            context_filter=context_filter,
            user_role=user_role,
        )
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(response=wrapped_response.response, sources=sources)
        
        # Loggear la respuesta
        logger.info("=" * 50)
        logger.info("** RESPUESTA DEL LLM: **")
        response_preview = completion.response[:500] + "..." if len(completion.response) > 500 else completion.response
        logger.info(f"response: {response_preview}")
        logger.info(f"sources: {len(sources)} documentos encontrados")
        logger.info("=" * 50)
        
        return completion
