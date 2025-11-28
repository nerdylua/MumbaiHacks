from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.request import Request
from app.models.response import Response, ProductionResponse, HealthResponse
from app.services.preprocessors.unified_processor import UnifiedDocumentProcessor
from app.services.retrievers.retrieval_service import RetrievalService
from app.services.retrievers.structure_aware_retrieval_service import TrueStructureAwareRetrievalService
from app.services.vector_stores.vector_store_factory import VectorStoreFactory
from app.services.logging.supabase_logger import supabase_logger
from app.providers.factory import LLMProviderFactory
from app.config.settings import settings
from app.services.pipelines.pipeline_manager import PipelineManager
import time
import uuid
import hashlib
from typing import Union, Optional

router = APIRouter()

vector_store = VectorStoreFactory.create_vector_store(settings)

llm_provider = LLMProviderFactory.create_provider(
    settings.DEFAULT_LLM_PROVIDER, 
    settings
)

document_processor = UnifiedDocumentProcessor(
    vector_store=vector_store,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
)

# Initialize both retrieval services for different pipeline types
traditional_retrieval_service = RetrievalService(
    vector_store=vector_store,
    llm_provider=llm_provider
)

structure_aware_retrieval_service = TrueStructureAwareRetrievalService(
    vector_store=vector_store,
    llm_provider=llm_provider
)

async def log_request_background(
    document_url: str,
    questions: list,
    answers: list,
    processing_time: float,
    document_metadata: dict,
    raw_response: dict,
    success: bool,
    error_message: Optional[str] = None
):
    try:
        await supabase_logger.log_api_request(
            document_url=document_url,
            questions=questions,
            answers=answers,
            processing_time=processing_time,
            document_metadata=document_metadata,
            raw_response=raw_response,
            success=success,
            error_message=error_message
        )
    except Exception as e:
        print(f"Background logging failed: {e}")

@router.post("/run", response_model=Union[Response, ProductionResponse])
async def run_rag(
    request: Request,
    background_tasks: BackgroundTasks):
    start_time = time.time()
    
    document_id = hashlib.sha256(request.documents.encode()).hexdigest()[:16]
    answers = []
    document_metadata = {}
    raw_response = {}
    success = True
    error_message = None
    
    try:
        # Get pipeline type from validated request (validator ensures it's supported)
        pipeline_type = request.processing_mode
        
        print(f"Using {pipeline_type} RAG processing: {PipelineManager.get_pipeline_description(pipeline_type)}")
        
        answers, document_metadata, raw_response = await PipelineManager.execute_pipeline(
            pipeline_type=pipeline_type,
            document_id=document_id,
            document_url=request.documents,
            questions=request.questions,
            k=request.k,
            vector_store=vector_store,
            document_processor=document_processor,
            traditional_retrieval_service=traditional_retrieval_service,
            structure_aware_retrieval_service=structure_aware_retrieval_service,
            settings=settings
        )

        processing_time = time.time() - start_time

        background_tasks.add_task(
            log_request_background,
            document_url=request.documents,
            questions=request.questions,
            answers=answers,
            processing_time=processing_time,
            document_metadata=document_metadata,
            raw_response=raw_response,
            success=success
        )

        if settings.ENVIRONMENT.lower() == "production":
            return ProductionResponse(
                success=True,
                answers=answers
            )
        else:
            return Response(
                success=True,
                answers=answers,
                processing_time=processing_time,
                document_metadata=document_metadata,
                raw_response=raw_response,
            )
        
    except HTTPException:
        processing_time = time.time() - start_time
        
        background_tasks.add_task(
            log_request_background,
            document_url=request.documents,
            questions=request.questions,
            answers=answers,
            processing_time=processing_time,
            document_metadata=document_metadata,
            raw_response=raw_response,
            success=False,
            error_message=error_message
        )
        
        raise
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        
        background_tasks.add_task(
            log_request_background,
            document_url=request.documents,
            questions=request.questions,
            answers=answers,
            processing_time=processing_time,
            document_metadata=document_metadata,
            raw_response=raw_response,
            success=False,
            error_message=error_message
        )
        
        raise HTTPException(status_code=500, detail=error_message)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        if hasattr(vector_store, 'aget_document_count'):
            doc_count = await vector_store.aget_document_count()
        else:
            doc_count = vector_store.get_document_count()
        
        return HealthResponse(
            status="healthy",
            vector_store=vector_store.store_type,
            llm_provider=llm_provider.provider_name,
            document_count=doc_count
        )
        
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            vector_store=vector_store.store_type,
            llm_provider=llm_provider.provider_name
        )
