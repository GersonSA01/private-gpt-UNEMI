# Solución: Error de Dimensiones de Embeddings (14315, 14348)

## Problema
Error: `operands could not be broadcast together with shapes (14315,) (14348,)`

Este error indica que hay documentos en Qdrant con dimensiones de embeddings **muy diferentes**:
- Algunos con **14315** dimensiones
- Otros con **14348** dimensiones
- Cuando deberían tener **384** dimensiones

## Causa
Los documentos fueron indexados con un **modelo de embeddings diferente** o hay **datos corruptos** en Qdrant.

**Posibles causas:**
1. Documentos indexados con un modelo anterior (antes de cambiar la configuración)
2. Documentos indexados con un modelo diferente (quizás Gemini embeddings que tienen más dimensiones)
3. Datos corruptos en Qdrant
4. Mezcla de documentos con diferentes modelos de embeddings

## Solución

### 1. Limpiar Completamente Qdrant ✅
```bash
# Detener el servicio
docker compose down

# Eliminar Qdrant
Remove-Item -Recurse -Force .\local_data\private_gpt\qdrant

# Eliminar stores
Remove-Item -Force .\local_data\private_gpt\docstore.json
Remove-Item -Force .\local_data\private_gpt\index_store.json
Remove-Item -Force .\local_data\private_gpt\graph_store.json
Remove-Item -Force .\local_data\private_gpt\image__vector_store.json

# Reiniciar el servicio
docker compose up -d
```

### 2. Verificar la Configuración ✅
Verifica que `settings-docker.yaml` tenga la configuración correcta:

```yaml
embedding:
  mode: ${PGPT_EMBED_MODE:huggingface}
  embed_dim: 384

huggingface:
  embedding_hf_model_name: ${PGPT_EMBEDDING_HF_MODEL_NAME:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}
  access_token: ${HF_TOKEN:-}
  trust_remote_code: ${PGPT_HF_TRUST_REMOTE_CODE:false}
```

### 3. Re-indexar Todos los Documentos ✅
Después de limpiar Qdrant, **debes re-indexar todos los documentos**:

1. Ve a la UI (http://localhost:8001)
2. Sube todos los PDFs nuevamente
3. Espera a que termine la indexación
4. Verifica que no haya errores

## Verificación

### Verificar que el Modelo Correcto se Está Usando
```bash
# Ver logs del contenedor
docker compose logs private-gpt-gemini | Select-String -Pattern "Initializing the embedding model"
```

Deberías ver:
```
Initializing the embedding model in mode=huggingface
```

### Verificar los Documentos Indexados
```bash
# Usar la API para verificar
curl http://localhost:8001/v1/ingest/list
```

### Verificar las Dimensiones
El modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` debería generar embeddings de **384 dimensiones**.

## Prevención

### 1. **Siempre Limpiar Qdrant al Cambiar el Modelo de Embeddings**
- Si cambias el modelo de embeddings, **siempre limpia Qdrant primero**
- No mezcles documentos indexados con diferentes modelos

### 2. **Verificar la Configuración Antes de Indexar**
- Verifica que `embed_dim` coincida con las dimensiones del modelo
- Verifica que el modelo correcto esté configurado

### 3. **No Mezclar Modelos**
- **NO** indexes documentos con diferentes modelos de embeddings
- Si necesitas cambiar el modelo, limpia Qdrant y re-indexa todo

### 4. **Usar el Mismo Modelo para Todos los Documentos**
- Asegúrate de que todos los documentos se indexen con el mismo modelo
- Verifica la configuración antes de indexar nuevos documentos

## Notas Importantes

### Dimensiones de Modelos Comunes
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: **384** dimensiones
- `nomic-ai/nomic-embed-text-v1.5`: **768** dimensiones
- `models/embedding-001` (Gemini): **768** dimensiones
- `text-embedding-ada-002` (OpenAI): **1536** dimensiones

### Las Dimensiones 14315 y 14348
Estas dimensiones **NO corresponden a ningún modelo estándar** de embeddings. Esto sugiere que:
- Los documentos fueron indexados con un modelo diferente o corrupto
- Hay algún problema con cómo se están generando los embeddings
- Los datos en Qdrant están corruptos

**Solución:** Limpiar Qdrant completamente y re-indexar todos los documentos con el modelo correcto (384 dimensiones).

## Próximos Pasos

1. ✅ **Qdrant limpiado completamente**
2. ✅ **Servicio reiniciado**
3. ⏳ **Re-indexar todos los PDFs** (debes hacerlo manualmente)
4. ⏳ **Verificar que no haya errores** después de re-indexar

## Verificación Final

Después de re-indexar, verifica que:
- ✅ No hay errores en los logs
- ✅ Los documentos aparecen en la lista
- ✅ Puedes hacer preguntas sobre los documentos
- ✅ No hay errores de dimensiones

## Si el Problema Persiste

Si después de limpiar Qdrant y re-indexar todavía tienes problemas:

1. **Verifica los logs del servidor:**
```bash
docker compose logs -f private-gpt-gemini
```

2. **Verifica la configuración:**
```bash
# Ver la configuración activa
docker compose exec private-gpt-gemini cat /home/worker/app/settings-docker.yaml
```

3. **Verifica que el modelo correcto se esté usando:**
```bash
docker compose logs private-gpt-gemini | Select-String -Pattern "embedding"
```

4. **Verifica las dimensiones del modelo:**
   - El modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` debería generar embeddings de **384 dimensiones**
   - Si las dimensiones son diferentes, verifica la configuración

## Referencias

- Modelo: [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Dimensiones: **384**
- Vector Store: **Qdrant**

