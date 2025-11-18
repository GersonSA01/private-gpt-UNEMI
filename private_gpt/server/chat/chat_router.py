import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


def parse_json_response(response_text: str) -> dict[str, any]:
    """
    Intenta parsear una respuesta JSON del modelo.
    
    Esta función centraliza la decisión de has_information en PrivateGPT.
    Retorna un dict con:
    - has_information_llm: lo que vino del JSON del modelo (o None si no se pudo parsear)
    - has_information_final: decisión final de PrivateGPT (siempre bool)
    - response_text: texto de respuesta limpio
    - fuentes_from_json: fuentes si el modelo las incluyó en el JSON
    
    El modelo puede responder en múltiples formatos:
    1. JSON válido: {"has_information": true/false, "response": "..."}
    2. Texto plano con has_information: "has_information: false\n..."
    3. Texto plano sin has_information explícito
    """
    if not response_text:
        logger.info(
            "PGPT has_information decision",
            extra={
                "has_info_llm": None,
                "has_info_final": False,
                "reason": "empty_response"
            },
        )
        return {
            "has_information_llm": None,
            "has_information_final": False,
            "response": "",
            "fuentes_from_json": []
        }
    
    response_text = response_text.strip()
    has_info_llm: bool | None = None
    has_info_final: bool = False
    response_clean = response_text
    fuentes_from_json: list = []
    
    # 1. Intentar parsear como JSON completo
    try:
        # Buscar el inicio del JSON (primer {)
        start_idx = response_text.find('{')
        if start_idx != -1:
            # Buscar el final del JSON balanceando las llaves
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if brace_count == 0:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "has_information" in parsed:
                    has_info_llm = parsed.get("has_information", False)
                    response_clean = parsed.get("response", "")
                    if not response_clean and has_info_llm:
                        # Si tiene información pero no hay response, usar el texto después del JSON
                        response_clean = response_text[end_idx:].strip() or response_text
                    elif not response_clean:
                        # Si no hay información y no hay response, usar string vacío o texto después del JSON
                        response_clean = response_text[end_idx:].strip() or ""
                    fuentes_from_json = parsed.get("fuentes", []) or []
                    
                    # Decisión final: confiar en el LLM si viene explícito
                    has_info_final = bool(has_info_llm)
                    
                    logger.info(
                        "PGPT has_information decision",
                        extra={
                            "has_info_llm": has_info_llm,
                            "has_info_final": has_info_final,
                            "reason": "json_complete"
                        },
                    )
                    return {
                        "has_information_llm": has_info_llm,
                        "has_information_final": has_info_final,
                        "response": response_clean,
                        "fuentes_from_json": fuentes_from_json
                    }
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.debug(f"No se pudo parsear JSON completo: {e}")
    
    # 2. Intentar extraer JSON parcial (buscar { ... "has_information" ... } en cualquier parte)
    try:
        json_match = re.search(r'\{[^{}]*"has_information"[^{}]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "has_information" in parsed:
                has_info_llm = parsed.get("has_information", False)
                response_clean = parsed.get("response", "")
                if not response_clean:
                    # Usar el texto después del JSON
                    json_end = json_match.end()
                    response_clean = response_text[json_end:].strip() or response_text
                fuentes_from_json = parsed.get("fuentes", []) or []
                
                # Decisión final: confiar en el LLM si viene explícito
                has_info_final = bool(has_info_llm)
                
                logger.info(
                    "PGPT has_information decision",
                    extra={
                        "has_info_llm": has_info_llm,
                        "has_info_final": has_info_final,
                        "reason": "json_partial"
                    },
                )
                return {
                    "has_information_llm": has_info_llm,
                    "has_information_final": has_info_final,
                    "response": response_clean,
                    "fuentes_from_json": fuentes_from_json
                }
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.debug(f"No se pudo parsear JSON parcial: {e}")
    
    # 3. Detectar patrón "has_information: false" o "has_information=false" en texto plano
    has_info_patterns = [
        r'has_information\s*:\s*(true|false)',
        r'has_information\s*=\s*(true|false)',
        r'"has_information"\s*:\s*(true|false)',
        r'"has_information"\s*=\s*(true|false)',
    ]
    
    for pattern in has_info_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            value_str = match.group(1).lower()
            has_info_llm = value_str in ["true", "1", "yes", "sí", "si"]
            
            # Extraer el texto después de has_information como response
            match_end = match.end()
            next_newline = response_text.find('\n', match_end)
            if next_newline != -1:
                response_clean = response_text[next_newline:].strip()
            else:
                response_clean = response_text[match_end:].strip()
                # Limpiar si empieza con ":" o "="
                response_clean = re.sub(r'^[:=]\s*', '', response_clean).strip()
            
            if not response_clean or len(response_clean) < 10:
                response_clean = response_text
            
            # Decisión final: confiar en el patrón detectado
            has_info_final = bool(has_info_llm)
            
            logger.info(
                "PGPT has_information decision",
                extra={
                    "has_info_llm": has_info_llm,
                    "has_info_final": has_info_final,
                    "reason": "pattern_detected"
                },
            )
            return {
                "has_information_llm": has_info_llm,
                "has_information_final": has_info_final,
                "response": response_clean,
                "fuentes_from_json": []
            }
    
    # 4. Fallback: usar heurísticas suaves si no hay decisión del LLM
    no_info_phrases = [
        "no se encuentra", "no hay información",
        "no contiene información", "no está disponible",
        "no tengo información", "no encuentro información",
        "no puedo ayudarte", "no puedo proporcionar",
        "te sugiero que te pongas en contacto",
        "contacta directamente", "lo siento, no",
        "disculpa, no", "lamentablemente no encontré",
        "no pude localizar", "no pude encontrar",
        "no encontré información específica",
        "Lo siento"
    ]
    response_lower = response_text.lower()
    has_no_info_phrase = any(phrase in response_lower for phrase in no_info_phrases)
    is_too_short = len(response_text.strip()) < 50
    
    # Heurística suave: si no hay frase negativa y no es muy corta, asumir que hay información
    has_info_final = not (has_no_info_phrase or is_too_short)
    
    logger.info(
        "PGPT has_information decision",
        extra={
            "has_info_llm": None,
            "has_info_final": has_info_final,
            "reason": "heuristic_fallback",
            "has_no_info_phrase": has_no_info_phrase,
            "is_too_short": is_too_short
        },
    )
    
    return {
        "has_information_llm": None,
        "has_information_final": has_info_final,
        "response": response_text,
        "fuentes_from_json": []
    }


from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
    to_openai_response,
    to_openai_sse_stream,
)
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.utils.auth import authenticated

chat_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class ChatBody(BaseModel):
    messages: list[OpenAIMessage]
    use_context: bool = False
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    stream: bool = False
    session_context: dict | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a rapper. Always answer with a rap.",
                        },
                        {
                            "role": "user",
                            "content": "How do you fry an egg?",
                        },
                    ],
                    "stream": False,
                    "use_context": True,
                    "include_sources": True,
                    "context_filter": {
                        "docs_ids": ["c202d5e6-7b69-4869-81cc-dd574ee8ee11"]
                    },
                }
            ]
        }
    }


@chat_router.post(
    "/chat/completions",
    response_model=None,
    responses={200: {"model": OpenAICompletion}},
    tags=["Contextual Completions"],
    openapi_extra={
        "x-fern-streaming": {
            "stream-condition": "stream",
            "response": {"$ref": "#/components/schemas/OpenAICompletion"},
            "response-stream": {"$ref": "#/components/schemas/OpenAICompletion"},
        }
    },
)
def chat_completion(
    request: Request, body: ChatBody
) -> OpenAICompletion | StreamingResponse:
    """Given a list of messages comprising a conversation, return a response.

    Optionally include an initial `role: system` message to influence the way
    the LLM answers.

    If `use_context` is set to `true`, the model will use context coming
    from the ingested documents to create the response. The documents being used can
    be filtered using the `context_filter` and passing the document IDs to be used.
    Ingested documents IDs can be found using `/ingest/list` endpoint. If you want
    all ingested documents to be used, remove `context_filter` altogether.

    When using `'include_sources': true`, the API will return the source Chunks used
    to create the response, which come from the context provided.

    When using `'stream': true`, the API will return data chunks following [OpenAI's
    streaming model](https://platform.openai.com/docs/api-reference/chat/streaming):
    ```
    {"id":"12345","object":"completion.chunk","created":1694268190,
    "model":"private-gpt","choices":[{"index":0,"delta":{"content":"Hello"},
    "finish_reason":null}]}
    ```
    """
    service = request.state.injector.get(ChatService)
    all_messages = [
        ChatMessage(content=m.content, role=MessageRole(m.role)) for m in body.messages
    ]
    
    # Extraer user_role del session_context si está disponible
    user_role = None
    if body.session_context and isinstance(body.session_context, dict):
        user_role = body.session_context.get("user_role")
    
    if body.stream:
        completion_gen = service.stream_chat(
            messages=all_messages,
            use_context=body.use_context,
            context_filter=body.context_filter,
            user_role=user_role,
        )
        
        # Para streaming, las fuentes se procesan al final en yield_deltas del UI
        # Pero aquí también podemos procesarlas si es necesario
        # Por ahora, pasamos las fuentes directamente para que se incluyan en el stream
        return StreamingResponse(
            to_openai_sse_stream(
                completion_gen.response,
                completion_gen.sources if body.include_sources else None,
            ),
            media_type="text/event-stream",
        )
    else:
        completion = service.chat(
            messages=all_messages,
            use_context=body.use_context,
            context_filter=body.context_filter,
            user_role=user_role,
        )
        
        # Procesar la respuesta JSON y asegurar que las fuentes estén correctas
        response_text = completion.response if completion.response else ""
        parsed_response = parse_json_response(response_text)
        clean_response = parsed_response.get("response", response_text)
        # Usar has_information_final de PrivateGPT como la verdad
        has_info = parsed_response.get("has_information_final", False)
        
        # Preparar lista de fuentes
        sources_list = []
        if completion.sources and body.include_sources:
            # Ordenar fuentes por score (mayor a menor) para obtener las más relevantes
            sorted_sources = sorted(
                completion.sources,
                key=lambda s: s.score if hasattr(s, 'score') and s.score is not None else 0,
                reverse=True
            )
            
            # Tomar solo las top 5 fuentes más relevantes (donde realmente se encontró la información)
            top_sources = sorted_sources[:5]
            
            used_files = set()
            for source in top_sources:
                if source.document and source.document.doc_metadata:
                    file_name = source.document.doc_metadata.get("file_name", "Unknown")
                    page_label = source.document.doc_metadata.get("page_label", "Unknown")
                    file_key = f"{file_name}-{page_label}"
                    if file_key not in used_files:
                        sources_list.append({
                            "archivo": file_name,  # Nombre completo del archivo desde metadata
                            "pagina": str(page_label)  # Asegurar que sea string
                        })
                        used_files.add(file_key)
            
            # Ordenar por página (menor a mayor) y eliminar duplicados
            sources_list.sort(key=lambda x: int(x["pagina"]) if x["pagina"].isdigit() else 999)
            logger.debug(f"Fuentes procesadas: {sources_list}")
        
        # SIEMPRE reconstruir el JSON con has_information (tanto si es True como False)
        import json as json_module
        response_json = json_module.dumps({
            "has_information": has_info,
            "response": clean_response,
            "fuentes": sources_list
        }, ensure_ascii=False)
        clean_response = response_json
        logger.debug(f"Respuesta final con has_information={has_info} y {len(sources_list)} fuentes: {clean_response[:200]}...")
        
        return to_openai_response(
            clean_response, completion.sources if body.include_sources else None
        )


class PriorityChatBody(BaseModel):
    messages: list[OpenAIMessage]
    priority_patterns: list[str] = Field(
        default=["unemi"],
        description="Lista de patrones para identificar archivos prioritarios. Se buscará primero en archivos que contengan estos patrones en el nombre.",
        examples=[["unemi", "reglamento"]]
    )
    fallback_to_all: bool = Field(
        default=True,
        description="Si True, buscará en el resto de archivos si no encuentra información relevante en los prioritarios."
    )
    min_response_length: int = Field(
        default=30,
        description="Longitud mínima de respuesta para considerar que se encontró información relevante. Reducido para ser más tolerante a errores ortográficos."
    )
    include_sources: bool = True
    stream: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "¿Cómo justificar una falta?",
                        },
                    ],
                    "priority_patterns": ["unemi"],
                    "fallback_to_all": True,
                    "stream": False,
                }
            ]
        }
    }


@chat_router.post(
    "/chat/completions/priority",
    response_model=None,
    responses={200: {"model": OpenAICompletion}},
    tags=["Contextual Completions"],
    openapi_extra={
        "x-fern-streaming": {
            "stream-condition": "stream",
            "response": {"$ref": "#/components/schemas/OpenAICompletion"},
            "response-stream": {"$ref": "#/components/schemas/OpenAICompletion"},
        }
    },
)
def priority_chat_completion(
    request: Request, body: PriorityChatBody
) -> OpenAICompletion | StreamingResponse:
    """Chat con búsqueda prioritaria en archivos específicos.
    
    Este endpoint busca primero en archivos que coincidan con los patrones especificados
    en `priority_patterns`. Si no encuentra información relevante, buscará en el resto
    de archivos (si `fallback_to_all` es True).
    
    Los patrones se buscan en el nombre del archivo (case-insensitive). Por ejemplo,
    si `priority_patterns=["unemi"]`, buscará primero en todos los archivos que contengan
    "unemi" en su nombre.
    
    Una respuesta se considera relevante si:
    - Tiene fuentes (sources) asociadas
    - La longitud de la respuesta es mayor a `min_response_length`
    
    La respuesta incluye información sobre los documentos consultados:
    - `consulted_documents`: Lista de todos los documentos que se consultaron en la búsqueda
    - `used_documents`: Lista de documentos que fueron realmente usados en la respuesta (aparecen en sources)
    - `unused_documents`: Lista de documentos consultados pero no usados en la respuesta
    """
    chat_service = request.state.injector.get(ChatService)
    ingest_service = request.state.injector.get(IngestService)
    
    # Obtener todos los documentos
    all_docs = ingest_service.list_ingested()
    
    if not all_docs:
        raise HTTPException(status_code=400, detail="No hay documentos ingestionados")
    
    # Organizar documentos por prioridad
    doc_groups = {}
    remaining_docs = all_docs.copy()
    
    for i, pattern in enumerate(body.priority_patterns):
        matching = [
            doc for doc in remaining_docs
            if doc.doc_metadata and pattern.lower() in doc.doc_metadata.get("file_name", "").lower()
        ]
        doc_groups[f"priority_{i}"] = {
            "pattern": pattern,
            "ids": [doc.doc_id for doc in matching]
        }
        remaining_docs = [doc for doc in remaining_docs if doc not in matching]
        logger.info(
            f"Patrón '{pattern}': {len(matching)} archivos encontrados. "
            f"Archivos: {[doc.doc_metadata.get('file_name', 'Unknown') if doc.doc_metadata else 'Unknown' for doc in matching]}"
        )
    
    # Convertir mensajes
    all_messages = [
        ChatMessage(content=m.content, role=MessageRole(m.role)) for m in body.messages
    ]
    
    # Buscar en orden de prioridad
    for group_name, group_data in doc_groups.items():
        if not group_data["ids"]:
            logger.debug(f"Grupo {group_name} vacío, saltando")
            continue
        
        logger.info(
            f"Buscando en archivos prioritarios '{group_data['pattern']}': "
            f"{len(group_data['ids'])} documentos"
        )
        context_filter = ContextFilter(docs_ids=group_data["ids"])
        
        if body.stream:
            completion_gen = chat_service.stream_chat(
                messages=all_messages,
                use_context=True,
                context_filter=context_filter,
            )
            # Para streaming, retornamos directamente (no verificamos relevancia)
            response = StreamingResponse(
                to_openai_sse_stream(
                    completion_gen.response,
                    completion_gen.sources if body.include_sources else None,
                ),
                media_type="text/event-stream",
            )
            # Agregar header para indicar el ámbito de búsqueda
            response.headers["X-Search-Scope"] = group_data["pattern"]
            return response
        else:
            completion = chat_service.chat(
                messages=all_messages,
                use_context=True,
                context_filter=context_filter,
            )
            
            # Verificar si la respuesta es relevante
            has_sources = completion.sources and len(completion.sources) > 0
            response_text = completion.response if completion.response else ""
            
            # Parsear respuesta JSON para obtener has_information
            parsed_response = parse_json_response(response_text)
            has_information = parsed_response.get("has_information", False)
            clean_response = parsed_response.get("response", response_text)
            
            # La respuesta es relevante si tiene sources, el modelo indica que hay información
            # y la respuesta tiene longitud suficiente
            is_relevant = (
                has_sources 
                and has_information
                and len(clean_response) >= body.min_response_length
            )
            
            logger.info(
                f"Búsqueda en '{group_data['pattern']}': "
                f"has_sources={has_sources}, has_information={has_information}, "
                f"length={len(clean_response)}, is_relevant={is_relevant}"
            )
            
            if is_relevant:
                # Respuesta relevante encontrada en archivos prioritarios
                # Asegurar que las fuentes tengan el nombre completo correcto del metadata
                parsed_response = parse_json_response(completion.response if completion.response else "")
                # Siempre reemplazar las fuentes con las del metadata para asegurar nombres completos correctos
                # Solo incluir las fuentes más relevantes (mejor score), no todas las consultadas
                if completion.sources:
                    # Ordenar fuentes por score (mayor a menor) para obtener las más relevantes
                    sorted_sources = sorted(
                        completion.sources,
                        key=lambda s: s.score if hasattr(s, 'score') and s.score is not None else 0,
                        reverse=True
                    )
                    
                    # Tomar solo las top 5 fuentes más relevantes (donde realmente se encontró la información)
                    top_sources = sorted_sources[:5]
                    
                    used_files = set()
                    sources_list = []
                    for source in top_sources:
                        if source.document and source.document.doc_metadata:
                            file_name = source.document.doc_metadata.get("file_name", "Unknown")
                            page_label = source.document.doc_metadata.get("page_label", "Unknown")
                            file_key = f"{file_name}-{page_label}"
                            if file_key not in used_files:
                                sources_list.append({
                                    "archivo": file_name,  # Nombre completo del archivo desde metadata
                                    "pagina": str(page_label)  # Asegurar que sea string
                                })
                                used_files.add(file_key)
                    # Reemplazar las fuentes del modelo con las correctas del metadata
                    parsed_response["fuentes"] = sources_list
                    # Reconstruir clean_response con fuentes correctas en el JSON
                    import json as json_module
                    response_with_sources = json_module.dumps({
                        "has_information": parsed_response.get("has_information", True),
                        "response": clean_response,
                        "fuentes": sources_list
                    }, ensure_ascii=False)
                    clean_response = response_with_sources
                
                # Usar la respuesta limpia (sin el JSON wrapper si estaba)
                openai_response = to_openai_response(
                    clean_response,
                    completion.sources if body.include_sources else None,
                )
                # Agregar información del ámbito de búsqueda y documentos consultados
                if isinstance(openai_response, dict):
                    # Obtener información de los documentos consultados
                    consulted_docs = [
                        doc for doc in all_docs if doc.doc_id in group_data["ids"]
                    ]
                    # Obtener documentos realmente usados (de los sources)
                    used_doc_ids = set()
                    if completion.sources:
                        for source in completion.sources:
                            if source.document and source.document.doc_id:
                                used_doc_ids.add(source.document.doc_id)
                    
                    used_docs = [
                        doc for doc in consulted_docs if doc.doc_id in used_doc_ids
                    ]
                    unused_docs = [
                        doc for doc in consulted_docs if doc.doc_id not in used_doc_ids
                    ]
                    
                    openai_response["search_scope"] = group_data["pattern"]
                    openai_response["priority_files_count"] = len(group_data["ids"])
                    openai_response["consulted_documents"] = [
                        {
                            "doc_id": doc.doc_id,
                            "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                        }
                        for doc in consulted_docs
                    ]
                    openai_response["used_documents"] = [
                        {
                            "doc_id": doc.doc_id,
                            "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                        }
                        for doc in used_docs
                    ]
                    openai_response["unused_documents"] = [
                        {
                            "doc_id": doc.doc_id,
                            "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                        }
                        for doc in unused_docs
                    ]
                return openai_response
            else:
                # Si no es relevante, continuar al siguiente grupo o fallback
                logger.debug(
                    f"Respuesta no relevante en {group_data['pattern']}: "
                    f"has_sources={has_sources}, has_information={has_information}, "
                    f"length={len(clean_response)}"
                )
                continue
    
    # Fallback: buscar en el resto de archivos
    if body.fallback_to_all:
        # Si hay documentos restantes, buscar solo en ellos
        # Si no hay documentos restantes pero hay documentos prioritarios que no dieron resultado,
        # buscar en todos los documentos
        if remaining_docs:
            other_ids = [doc.doc_id for doc in remaining_docs]
            logger.debug(f"Fallback: buscando en {len(other_ids)} archivos adicionales")
        else:
            # Si no hay documentos restantes, buscar en todos (puede que todos sean prioritarios)
            other_ids = [doc.doc_id for doc in all_docs]
            logger.debug(f"Fallback: buscando en todos los {len(other_ids)} archivos")
        
        context_filter = ContextFilter(docs_ids=other_ids) if other_ids else None
        
        if body.stream:
            completion_gen = chat_service.stream_chat(
                messages=all_messages,
                use_context=True,
                context_filter=context_filter,
            )
            response = StreamingResponse(
                to_openai_sse_stream(
                    completion_gen.response,
                    completion_gen.sources if body.include_sources else None,
                ),
                media_type="text/event-stream",
            )
            response.headers["X-Search-Scope"] = "fallback"
            return response
        else:
            completion = chat_service.chat(
                messages=all_messages,
                use_context=True,
                context_filter=context_filter,
            )
            # Limpiar respuesta JSON si viene en ese formato
            response_text = completion.response if completion.response else ""
            parsed_response = parse_json_response(response_text)
            clean_response = parsed_response.get("response", response_text)
            
            openai_response = to_openai_response(
                clean_response,
                completion.sources if body.include_sources else None,
            )
            if isinstance(openai_response, dict):
                # Obtener información de los documentos consultados en fallback
                consulted_docs = [
                    doc for doc in all_docs if doc.doc_id in other_ids
                ]
                # Obtener documentos realmente usados
                used_doc_ids = set()
                if completion.sources:
                    for source in completion.sources:
                        if source.document and source.document.doc_id:
                            used_doc_ids.add(source.document.doc_id)
                
                used_docs = [
                    doc for doc in consulted_docs if doc.doc_id in used_doc_ids
                ]
                unused_docs = [
                    doc for doc in consulted_docs if doc.doc_id not in used_doc_ids
                ]
                
                openai_response["search_scope"] = "fallback"
                openai_response["fallback_files_count"] = len(other_ids)
                openai_response["consulted_documents"] = [
                    {
                        "doc_id": doc.doc_id,
                        "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                    }
                    for doc in consulted_docs
                ]
                openai_response["used_documents"] = [
                    {
                        "doc_id": doc.doc_id,
                        "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                    }
                    for doc in used_docs
                ]
                openai_response["unused_documents"] = [
                    {
                        "doc_id": doc.doc_id,
                        "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                    }
                    for doc in unused_docs
                ]
            return openai_response
    
    # Si no hay fallback y no se encontró nada, retornar última búsqueda
    if doc_groups:
        last_group = list(doc_groups.values())[-1]
        context_filter = ContextFilter(docs_ids=last_group["ids"])
        completion = chat_service.chat(
            messages=all_messages,
            use_context=True,
            context_filter=context_filter,
        )
        # Limpiar respuesta JSON si viene en ese formato
        response_text = completion.response if completion.response else ""
        parsed_response = parse_json_response(response_text)
        clean_response = parsed_response.get("response", response_text)
        
        openai_response = to_openai_response(
            clean_response,
            completion.sources if body.include_sources else None,
        )
        if isinstance(openai_response, dict):
            # Obtener información de los documentos consultados
            consulted_docs = [
                doc for doc in all_docs if doc.doc_id in last_group["ids"]
            ]
            # Obtener documentos realmente usados
            used_doc_ids = set()
            if completion.sources:
                for source in completion.sources:
                    if source.document and source.document.doc_id:
                        used_doc_ids.add(source.document.doc_id)
            
            used_docs = [
                doc for doc in consulted_docs if doc.doc_id in used_doc_ids
            ]
            unused_docs = [
                doc for doc in consulted_docs if doc.doc_id not in used_doc_ids
            ]
            
            openai_response["search_scope"] = "priority_only"
            openai_response["consulted_documents"] = [
                {
                    "doc_id": doc.doc_id,
                    "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                }
                for doc in consulted_docs
            ]
            openai_response["used_documents"] = [
                {
                    "doc_id": doc.doc_id,
                    "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                }
                for doc in used_docs
            ]
            openai_response["unused_documents"] = [
                {
                    "doc_id": doc.doc_id,
                    "file_name": doc.doc_metadata.get("file_name", "Unknown") if doc.doc_metadata else "Unknown"
                }
                for doc in unused_docs
            ]
        return openai_response
    
    # Si no hay documentos prioritarios, buscar en todos
    if body.stream:
        completion_gen = chat_service.stream_chat(
            messages=all_messages,
            use_context=True,
            context_filter=None,
        )
        return StreamingResponse(
            to_openai_sse_stream(
                completion_gen.response,
                completion_gen.sources if body.include_sources else None,
            ),
            media_type="text/event-stream",
        )
    else:
        completion = chat_service.chat(
            messages=all_messages,
            use_context=True,
            context_filter=None,
        )
        # Limpiar respuesta JSON si viene en ese formato
        response_text = completion.response if completion.response else ""
        parsed_response = parse_json_response(response_text)
        clean_response = parsed_response.get("response", response_text)
        
        return to_openai_response(
            clean_response, completion.sources if body.include_sources else None
        )
